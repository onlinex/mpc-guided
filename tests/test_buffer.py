"""Unit tests for TransitionReplayBuffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dynamics.buffer import TransitionReplayBuffer


def _make_buffer(capacity=8, visual_dim=4, proprio_dim=2, action_dim=3):
    return TransitionReplayBuffer(
        capacity,
        visual_dim=visual_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        seed=0,
    )


def _add_n(buf, n, *, offset=0):
    for i in range(n):
        buf.add(
            np.full(buf.visual_dim, i + offset, dtype=np.float32),
            np.full(buf.proprio_dim, i + offset, dtype=np.float32),
            np.full(buf.action_dim, i + offset, dtype=np.float32),
            np.full(buf.visual_dim, i + offset + 0.5, dtype=np.float32),
            np.full(buf.proprio_dim, i + offset + 0.5, dtype=np.float32),
        )


def test_add_and_size():
    buf = _make_buffer(capacity=4)
    assert len(buf) == 0
    _add_n(buf, 3)
    assert len(buf) == 3
    assert not buf.full


def test_capacity_caps_size():
    buf = _make_buffer(capacity=4)
    _add_n(buf, 10)
    assert len(buf) == 4
    assert buf.full


def test_sample_returns_correct_shapes():
    buf = _make_buffer(capacity=8, visual_dim=5, proprio_dim=2, action_dim=3)
    _add_n(buf, 8)
    batch = buf.sample(4, device=torch.device("cpu"))
    assert batch.visual.shape == (4, 5)
    assert batch.proprio.shape == (4, 2)
    assert batch.action.shape == (4, 3)
    assert batch.next_visual.shape == (4, 5)
    assert batch.next_proprio.shape == (4, 2)


def test_sample_empty_raises():
    buf = _make_buffer()
    with pytest.raises(ValueError):
        buf.sample(1, device=torch.device("cpu"))


def test_pin_protects_initial_entries():
    buf = _make_buffer(capacity=4)
    _add_n(buf, 2)  # entries 0, 1
    buf.pin_current_contents()
    assert buf.pinned == 2

    # Overwrite many times — pinned entries 0, 1 must survive.
    _add_n(buf, 20, offset=100)

    visuals = buf._visual.copy()  # noqa: SLF001
    pinned_vals = {float(visuals[i, 0]) for i in range(2)}
    assert pinned_vals == {0.0, 1.0}


def test_sample_expert_returns_only_pinned_region():
    buf = _make_buffer(capacity=8)
    _add_n(buf, 3)  # values 0, 1, 2 -> pinned
    buf.pin_current_contents()
    _add_n(buf, 5, offset=100)  # values 100..104 -> on-policy

    seen = set()
    for _ in range(50):
        batch = buf.sample_expert(4, device=torch.device("cpu"))
        for v in batch.visual[:, 0].tolist():
            seen.add(float(v))
    assert seen <= {0.0, 1.0, 2.0}, f"expert sample leaked into on-policy: {seen}"


def test_sample_on_policy_returns_only_unpinned_region():
    buf = _make_buffer(capacity=8)
    _add_n(buf, 3)
    buf.pin_current_contents()
    _add_n(buf, 4, offset=100)  # 100, 101, 102, 103

    seen = set()
    for _ in range(50):
        batch = buf.sample_on_policy(4, device=torch.device("cpu"))
        for v in batch.visual[:, 0].tolist():
            seen.add(float(v))
    assert seen <= {100.0, 101.0, 102.0, 103.0}, f"on-policy sample leaked into expert: {seen}"


def test_sample_on_policy_is_none_when_empty():
    buf = _make_buffer(capacity=8)
    _add_n(buf, 3)
    buf.pin_current_contents()
    assert buf.sample_on_policy(4, device=torch.device("cpu")) is None


def test_sample_expert_is_none_when_unpinned():
    buf = _make_buffer(capacity=8)
    _add_n(buf, 3)
    # never pinned
    assert buf.sample_expert(4, device=torch.device("cpu")) is None


def test_add_validates_shapes():
    buf = _make_buffer(visual_dim=4)
    with pytest.raises(ValueError):
        buf.add(
            np.zeros(3, dtype=np.float32),  # wrong visual dim
            np.zeros(buf.proprio_dim, dtype=np.float32),
            np.zeros(buf.action_dim, dtype=np.float32),
            np.zeros(buf.visual_dim, dtype=np.float32),
            np.zeros(buf.proprio_dim, dtype=np.float32),
        )
