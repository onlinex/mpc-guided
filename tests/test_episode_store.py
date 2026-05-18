"""Unit tests for EpisodeStore."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dynamics.episode_store import EpisodeStore

DEVICE = torch.device("cpu")


def _make_store(capacity=100, visual_dim=4, proprio_dim=2, action_dim=3, seed=0):
    return EpisodeStore(
        capacity,
        visual_dim=visual_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        seed=seed,
    )


def _make_episode(T, *, marker, visual_dim=4, proprio_dim=2, action_dim=3):
    """Episode encoding the marker in col 0 and the timestep in col 1.

    Lets tests assert windows are (a) from one episode (col 0 constant) and
    (b) contiguous (col 1 strictly increments by 1). Integer values survive
    float32 cleanly, unlike ``marker + t / 1000``.
    """
    visual = np.zeros((T + 1, visual_dim), dtype=np.float32)
    proprio = np.zeros((T + 1, proprio_dim), dtype=np.float32)
    action = np.zeros((T, action_dim), dtype=np.float32)
    visual[:, 0] = marker
    proprio[:, 0] = marker
    action[:, 0] = marker
    visual[:, 1] = np.arange(T + 1, dtype=np.float32)
    if proprio_dim >= 2:
        proprio[:, 1] = np.arange(T + 1, dtype=np.float32)
    if action_dim >= 2:
        action[:, 1] = np.arange(T, dtype=np.float32)
    return visual, proprio, action


# ---- basic mutation / accounting ----


def test_empty_store():
    store = _make_store()
    assert store.num_episodes == 0
    assert store.num_transitions == 0
    assert store.sample(4, device=DEVICE) is None


def test_add_episode_counts():
    store = _make_store(capacity=100)
    v, p, a = _make_episode(10, marker=1.0)
    store.add_episode(v, p, a, pinned=False)
    assert store.num_episodes == 1
    assert store.num_transitions == 10
    assert store.num_pinned_episodes == 0
    assert store.num_on_policy_transitions == 10


def test_pinned_counts():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(5, marker=1.0), pinned=True)
    store.add_episode(*_make_episode(7, marker=2.0), pinned=False)
    assert store.num_pinned_episodes == 1
    assert store.num_pinned_transitions == 5
    assert store.num_on_policy_transitions == 7
    assert store.num_transitions == 12


# ---- eviction ----


def test_eviction_drops_oldest_non_pinned():
    store = _make_store(capacity=15)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    store.add_episode(*_make_episode(10, marker=2.0), pinned=False)
    # Capacity 15, only counts non-pinned; second add (20 total) evicts first (marker=1).
    assert store.num_episodes == 1
    # Verify it's the second one that remains by checking the marker.
    batch = store.sample(64, horizon=1, context=1, device=DEVICE)
    seen = set(batch.visual_context[:, -1, 0].tolist())
    assert all(v >= 2.0 for v in seen), f"old episode leaked after eviction: {seen}"


def test_pinned_never_evicts():
    store = _make_store(capacity=5)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=True)
    # Add many on-policy episodes; pinned must survive.
    for k in range(5):
        store.add_episode(*_make_episode(8, marker=100.0 + k), pinned=False)
    assert store.num_pinned_episodes == 1
    assert store.num_pinned_transitions == 10
    # On-policy region is capped.
    assert store.num_on_policy_transitions <= 5 + 8  # last add may overshoot until next evict


def test_eviction_handles_overshoot():
    # Single non-pinned episode larger than capacity should still be retained
    # (we don't split episodes), and a subsequent add should drop it.
    store = _make_store(capacity=5)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    assert store.num_episodes == 1
    store.add_episode(*_make_episode(3, marker=2.0), pinned=False)
    # First (10) evicted because adding 3 made total 13 > cap 5.
    assert store.num_episodes == 1
    batch = store.sample(8, device=DEVICE)
    assert all(v >= 2.0 for v in batch.visual_context[:, -1, 0].tolist())


# ---- sampling: shapes ----


def test_sample_default_shapes():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    batch = store.sample(4, device=DEVICE)
    assert batch.visual_context.shape == (4, 1, store.visual_dim)
    assert batch.proprio_context.shape == (4, 1, store.proprio_dim)
    assert batch.action.shape == (4, 1, store.action_dim)
    assert batch.visual_future.shape == (4, 1, store.visual_dim)
    assert batch.proprio_future.shape == (4, 1, store.proprio_dim)


def test_sample_horizon_shapes():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(20, marker=1.0), pinned=False)
    batch = store.sample(8, horizon=5, context=3, device=DEVICE)
    assert batch.visual_context.shape == (8, 3, store.visual_dim)
    assert batch.action.shape == (8, 5, store.action_dim)
    assert batch.visual_future.shape == (8, 5, store.visual_dim)


# ---- sampling: contiguity & no-cross-boundary ----


def test_window_is_contiguous_within_episode():
    """Every window must come from one episode (same integer marker) and be contiguous."""
    store = _make_store(capacity=1000)
    for k in range(5):
        store.add_episode(*_make_episode(15, marker=float(k * 100)), pinned=False)
    batch = store.sample(128, horizon=4, context=3, device=DEVICE)
    markers = np.concatenate(
        [batch.visual_context[:, :, 0].numpy(), batch.visual_future[:, :, 0].numpy()], axis=1
    )
    timesteps = np.concatenate(
        [batch.visual_context[:, :, 1].numpy(), batch.visual_future[:, :, 1].numpy()], axis=1
    )
    assert np.all(markers == markers[:, :1]), "window crossed an episode boundary"
    diffs = np.diff(timesteps, axis=1)
    assert np.all(diffs == 1.0), f"window not contiguous: diffs={diffs}"


def test_short_episode_excluded_when_horizon_too_large():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(3, marker=1.0), pinned=False)   # T=3
    store.add_episode(*_make_episode(20, marker=2.0), pinned=False)  # T=20
    # horizon=5, context=1 needs T >= 5 → first excluded.
    batch = store.sample(64, horizon=5, context=1, device=DEVICE)
    markers = batch.visual_context[:, -1, 0].numpy()
    assert np.all(np.floor(markers) == 2.0), "short episode leaked into horizon=5 sample"


def test_returns_none_when_no_valid_window():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(3, marker=1.0), pinned=False)
    # horizon=10 > any episode length
    assert store.sample(4, horizon=10, context=1, device=DEVICE) is None


def test_context_excludes_early_anchors():
    """With context=3, the anchor t must satisfy t >= 2 (need frames t-2, t-1, t)."""
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    batch = store.sample(256, horizon=1, context=3, device=DEVICE)
    # Last context frame = state at t. Anchor t ranges over [context-1, T-horizon] = [2, 9].
    ts = batch.visual_context[:, -1, 1].numpy().astype(int)
    assert ts.min() >= 2
    assert ts.max() <= 9


# ---- sampling: source filtering ----


def test_source_expert_returns_only_pinned():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(8, marker=1.0), pinned=True)
    store.add_episode(*_make_episode(8, marker=100.0), pinned=False)
    seen = set()
    for _ in range(40):
        batch = store.sample(8, source="expert", device=DEVICE)
        for v in batch.visual_context[:, -1, 0].tolist():
            seen.add(int(np.floor(v)))
    assert seen == {1}, f"expert source leaked on-policy: {seen}"


def test_source_on_policy_returns_only_non_pinned():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(8, marker=1.0), pinned=True)
    store.add_episode(*_make_episode(8, marker=100.0), pinned=False)
    seen = set()
    for _ in range(40):
        batch = store.sample(8, source="on_policy", device=DEVICE)
        for v in batch.visual_context[:, -1, 0].tolist():
            seen.add(int(np.floor(v)))
    assert seen == {100}, f"on_policy source leaked expert: {seen}"


def test_source_returns_none_when_empty():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(8, marker=1.0), pinned=True)
    assert store.sample(4, source="on_policy", device=DEVICE) is None
    store2 = _make_store(capacity=100)
    store2.add_episode(*_make_episode(8, marker=100.0), pinned=False)
    assert store2.sample(4, source="expert", device=DEVICE) is None


# ---- index cache invalidation ----


def test_index_cache_invalidates_on_add():
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(8, marker=1.0), pinned=False)
    batch1 = store.sample(64, device=DEVICE)
    seen1 = {int(np.floor(v)) for v in batch1.visual_context[:, -1, 0].tolist()}
    assert seen1 == {1}
    store.add_episode(*_make_episode(8, marker=2.0), pinned=False)
    seen2: set[int] = set()
    for _ in range(40):
        batch = store.sample(64, device=DEVICE)
        seen2 |= {int(np.floor(v)) for v in batch.visual_context[:, -1, 0].tolist()}
    assert seen2 == {1, 2}, f"new episode not reachable after add: {seen2}"


def test_index_cache_invalidates_on_evict():
    store = _make_store(capacity=10)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    # Prime cache.
    store.sample(8, device=DEVICE)
    # Force eviction of episode 1.
    store.add_episode(*_make_episode(10, marker=2.0), pinned=False)
    seen: set[int] = set()
    for _ in range(40):
        batch = store.sample(8, device=DEVICE)
        seen |= {int(np.floor(v)) for v in batch.visual_context[:, -1, 0].tolist()}
    assert seen == {2}, f"evicted episode still sampled: {seen}"


# ---- validation ----


def test_add_validates_action_shape():
    store = _make_store(action_dim=3)
    v, p, _ = _make_episode(5, marker=1.0)
    bad_action = np.zeros((5, 2), dtype=np.float32)  # wrong action_dim
    with pytest.raises(ValueError):
        store.add_episode(v, p, bad_action, pinned=False)


def test_add_validates_state_length():
    store = _make_store(visual_dim=4, proprio_dim=2, action_dim=3)
    v, p, a = _make_episode(5, marker=1.0)  # v,p length 6; a length 5
    bad_visual = v[:-1]  # length 5, should be 6
    with pytest.raises(ValueError):
        store.add_episode(bad_visual, p, a, pinned=False)


def test_add_rejects_empty_episode():
    store = _make_store(visual_dim=4, proprio_dim=2, action_dim=3)
    with pytest.raises(ValueError):
        store.add_episode(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            pinned=False,
        )


def test_invalid_capacity():
    with pytest.raises(ValueError):
        EpisodeStore(0, visual_dim=4, proprio_dim=2, action_dim=3)


def test_invalid_sample_params():
    store = _make_store()
    store.add_episode(*_make_episode(5, marker=1.0), pinned=False)
    with pytest.raises(ValueError):
        store.sample(0, device=DEVICE)
    with pytest.raises(ValueError):
        store.sample(4, horizon=0, device=DEVICE)
    with pytest.raises(ValueError):
        store.sample(4, context=0, device=DEVICE)


# ---- semantic equivalence with old single-step view ----


def test_single_step_action_matches_state_transition():
    """For horizon=1, action[i, 0] should be the action that takes context[-1] to future[0]."""
    store = _make_store(capacity=100)
    store.add_episode(*_make_episode(10, marker=1.0), pinned=False)
    batch = store.sample(32, horizon=1, context=1, device=DEVICE)
    # col 0 = marker (constant per episode), col 1 = timestep.
    # action[t] is taken AT state t, so action col 1 must match context's last col 1,
    # and the future state must be exactly one timestep later.
    assert torch.equal(batch.action[:, 0, 1], batch.visual_context[:, -1, 1])
    assert torch.equal(
        batch.visual_future[:, 0, 1],
        batch.visual_context[:, -1, 1] + 1.0,
    )
