"""Episodic replay store for dynamics training.

Replaces flat transition buffers with a list of per-episode arrays. This makes
two things first-class:

  * **Multi-step rollouts** — sample a future window ``[t+1 … t+H]`` from a
    single episode for compounding-error-aware dynamics training.
  * **Frame-stacked context** — sample a past window ``[t-k+1 … t]`` for
    encoders that consume short history. No consumer uses ``context > 1`` yet,
    but the buffer supports it so the wiring is one CLI flag away.

Eviction is at episode granularity: when total transitions across non-pinned
episodes exceed the capacity budget, the oldest non-pinned episode is dropped
whole. Pinned (expert) episodes never evict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Source = Literal["any", "expert", "on_policy"]


@dataclass(frozen=True)
class Episode:
    """One contiguous trajectory.

    ``visual`` / ``proprio`` hold ``T+1`` states; ``action`` holds the ``T``
    actions that take state ``t`` to state ``t+1``.
    """

    visual: np.ndarray   # (T+1, visual_dim)
    proprio: np.ndarray  # (T+1, proprio_dim)
    action: np.ndarray   # (T,   action_dim)
    pinned: bool
    episode_id: int

    @property
    def num_transitions(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class WindowBatch:
    """A batch of temporal windows sampled from episodes.

    Indexing convention for a single row, given anchor timestep ``t``:

      * ``visual_context[:, -1]``  == state at ``t``
      * ``visual_context[:,  0]``  == state at ``t - context + 1``
      * ``action[:,  0]``          == action taken at ``t``
      * ``visual_future[:, -1]``   == state at ``t + horizon``
    """

    visual_context: torch.Tensor   # (B, context, visual_dim)
    proprio_context: torch.Tensor  # (B, context, proprio_dim)
    action: torch.Tensor           # (B, horizon, action_dim)
    visual_future: torch.Tensor    # (B, horizon, visual_dim)
    proprio_future: torch.Tensor   # (B, horizon, proprio_dim)


class EpisodeStore:
    """Episode-keyed replay store with pinning and (horizon, context) sampling."""

    def __init__(
        self,
        capacity_transitions: int,
        *,
        visual_dim: int,
        proprio_dim: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        if capacity_transitions < 1:
            raise ValueError(f"capacity_transitions must be >= 1, got {capacity_transitions}")
        self.capacity_transitions = int(capacity_transitions)
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self._episodes: list[Episode] = []
        self._next_episode_id = 0
        self._rng = np.random.default_rng(seed)
        # Cache: (horizon, context, source) -> (N, 2) array of (episode_idx, t_anchor)
        self._index_cache: dict[tuple[int, int, str], np.ndarray] = {}

    # ---- introspection ----

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_pinned_episodes(self) -> int:
        return sum(1 for e in self._episodes if e.pinned)

    @property
    def num_transitions(self) -> int:
        return sum(e.num_transitions for e in self._episodes)

    @property
    def num_pinned_transitions(self) -> int:
        return sum(e.num_transitions for e in self._episodes if e.pinned)

    @property
    def num_on_policy_transitions(self) -> int:
        return sum(e.num_transitions for e in self._episodes if not e.pinned)

    # ---- mutation ----

    def add_episode(
        self,
        visual: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        *,
        pinned: bool,
    ) -> None:
        """Append an episode. Pinned episodes never evict."""
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        self._check_episode_shapes(visual, proprio, action)

        ep = Episode(
            visual=visual,
            proprio=proprio,
            action=action,
            pinned=bool(pinned),
            episode_id=self._next_episode_id,
        )
        self._episodes.append(ep)
        self._next_episode_id += 1
        self._index_cache.clear()
        if not pinned:
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop oldest non-pinned episodes until under budget.

        Always keeps the most-recently-added non-pinned episode, even if it
        alone exceeds capacity — otherwise a single oversized rollout would
        wipe the on-policy region entirely.
        """
        cap = self.capacity_transitions
        while self.num_on_policy_transitions > cap:
            non_pinned_idxs = [i for i, e in enumerate(self._episodes) if not e.pinned]
            if len(non_pinned_idxs) <= 1:
                return
            self._episodes.pop(non_pinned_idxs[0])
            self._index_cache.clear()

    # ---- sampling ----

    def sample(
        self,
        batch_size: int,
        *,
        horizon: int = 1,
        context: int = 1,
        source: Source = "any",
        device: torch.device,
    ) -> WindowBatch | None:
        """Sample ``batch_size`` windows. Returns ``None`` if no valid window exists."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if context < 1:
            raise ValueError(f"context must be >= 1, got {context}")

        valid = self._valid_anchors(horizon=horizon, context=context, source=source)
        if valid.shape[0] == 0:
            return None
        choice = self._rng.integers(0, valid.shape[0], size=batch_size)
        picks = valid[choice]
        return self._gather(picks, horizon=horizon, context=context, device=device)

    def _valid_anchors(self, *, horizon: int, context: int, source: Source) -> np.ndarray:
        key = (horizon, context, source)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        rows: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            if source == "expert" and not ep.pinned:
                continue
            if source == "on_policy" and ep.pinned:
                continue
            T = ep.num_transitions
            lo = context - 1            # need context past frames including t
            hi = T - horizon            # need horizon actions starting at t
            if hi < lo:
                continue
            for t in range(lo, hi + 1):
                rows.append((ep_idx, t))
        anchors = (
            np.asarray(rows, dtype=np.int64)
            if rows
            else np.empty((0, 2), dtype=np.int64)
        )
        self._index_cache[key] = anchors
        return anchors

    def _gather(
        self,
        picks: np.ndarray,
        *,
        horizon: int,
        context: int,
        device: torch.device,
    ) -> WindowBatch:
        B = picks.shape[0]
        vc = np.empty((B, context, self.visual_dim), dtype=np.float32)
        pc = np.empty((B, context, self.proprio_dim), dtype=np.float32)
        ac = np.empty((B, horizon, self.action_dim), dtype=np.float32)
        vf = np.empty((B, horizon, self.visual_dim), dtype=np.float32)
        pf = np.empty((B, horizon, self.proprio_dim), dtype=np.float32)
        for i in range(B):
            ep_idx, t = int(picks[i, 0]), int(picks[i, 1])
            ep = self._episodes[ep_idx]
            vc[i] = ep.visual[t - context + 1 : t + 1]
            pc[i] = ep.proprio[t - context + 1 : t + 1]
            ac[i] = ep.action[t : t + horizon]
            vf[i] = ep.visual[t + 1 : t + horizon + 1]
            pf[i] = ep.proprio[t + 1 : t + horizon + 1]
        return WindowBatch(
            visual_context=torch.as_tensor(vc, device=device),
            proprio_context=torch.as_tensor(pc, device=device),
            action=torch.as_tensor(ac, device=device),
            visual_future=torch.as_tensor(vf, device=device),
            proprio_future=torch.as_tensor(pf, device=device),
        )

    # ---- validation ----

    def _check_episode_shapes(
        self, visual: np.ndarray, proprio: np.ndarray, action: np.ndarray
    ) -> None:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(
                f"action must be (T, {self.action_dim}), got {tuple(action.shape)}"
            )
        T = action.shape[0]
        if T < 1:
            raise ValueError("episode must contain at least one transition")
        expected_state_shape = (T + 1,)
        if visual.ndim != 2 or visual.shape != (T + 1, self.visual_dim):
            raise ValueError(
                f"visual must be {(*expected_state_shape, self.visual_dim)}, "
                f"got {tuple(visual.shape)}"
            )
        if proprio.ndim != 2 or proprio.shape != (T + 1, self.proprio_dim):
            raise ValueError(
                f"proprio must be {(*expected_state_shape, self.proprio_dim)}, "
                f"got {tuple(proprio.shape)}"
            )
