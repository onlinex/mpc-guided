"""Unit tests for RolloutAccumulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.legacy.dynamics.episode_store import EpisodeStore
from src.legacy.dynamics.rollout_accumulator import RolloutAccumulator

DEVICE = torch.device("cpu")
VISUAL_DIM = 4
PROPRIO_DIM = 2
ACTION_DIM = 3


def _store():
    return EpisodeStore(
        capacity_transitions=200,
        visual_dim=VISUAL_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        seed=0,
    )


def _v(val):
    return np.full(VISUAL_DIM, val, dtype=np.float32)


def _p(val):
    return np.full(PROPRIO_DIM, val, dtype=np.float32)


def _a(val):
    return np.full(ACTION_DIM, val, dtype=np.float32)


def _roll(acc, n, *, start=0.0):
    """Drive an n-step episode whose state[t] = start + t."""
    acc.start_episode(_v(start), _p(start))
    for t in range(n):
        acc.append_step(_a(start + t), _v(start + t + 1), _p(start + t + 1))


def test_happy_path_flushes_one_episode():
    store = _store()
    acc = RolloutAccumulator(store)
    _roll(acc, 5)
    acc.finish_episode()
    assert store.num_episodes == 1
    assert store.num_transitions == 5
    batch = store.sample(64, horizon=1, context=1, device=DEVICE)
    # state[t] = t, so state[t+1] - state[t] == 1.
    diffs = (batch.visual_future[:, 0, 0] - batch.visual_context[:, -1, 0]).numpy()
    assert np.allclose(diffs, 1.0)


def test_two_episodes_back_to_back():
    store = _store()
    acc = RolloutAccumulator(store)
    _roll(acc, 4, start=0.0)
    acc.finish_episode()
    _roll(acc, 6, start=100.0)
    acc.finish_episode()
    assert store.num_episodes == 2
    assert store.num_transitions == 10


def test_pinned_flag_propagates():
    store = _store()
    acc = RolloutAccumulator(store)
    _roll(acc, 4)
    acc.finish_episode(pinned=True)
    assert store.num_pinned_episodes == 1


def test_append_before_start_raises():
    acc = RolloutAccumulator(_store())
    with pytest.raises(RuntimeError, match="before start_episode"):
        acc.append_step(_a(0), _v(0), _p(0))


def test_finish_with_no_episode_raises():
    acc = RolloutAccumulator(_store())
    with pytest.raises(RuntimeError, match="no in-progress"):
        acc.finish_episode()


def test_finish_zero_step_episode_raises():
    acc = RolloutAccumulator(_store())
    acc.start_episode(_v(0), _p(0))
    with pytest.raises(RuntimeError, match="zero-step"):
        acc.finish_episode()


def test_start_while_in_progress_raises():
    acc = RolloutAccumulator(_store())
    _roll(acc, 2)
    with pytest.raises(RuntimeError, match="in progress"):
        acc.start_episode(_v(0), _p(0))


def test_discard_drops_in_progress():
    store = _store()
    acc = RolloutAccumulator(store)
    _roll(acc, 3)
    assert acc.in_progress
    acc.discard()
    assert not acc.in_progress
    assert store.num_episodes == 0
    # After discard a fresh episode can begin without error.
    _roll(acc, 2)
    acc.finish_episode()
    assert store.num_episodes == 1


def test_steps_and_in_progress_accounting():
    acc = RolloutAccumulator(_store())
    assert not acc.in_progress and acc.steps == 0
    acc.start_episode(_v(0), _p(0))
    assert acc.in_progress and acc.steps == 0
    acc.append_step(_a(0), _v(1), _p(1))
    assert acc.steps == 1
    acc.append_step(_a(1), _v(2), _p(2))
    assert acc.steps == 2
    acc.finish_episode()
    assert not acc.in_progress and acc.steps == 0
