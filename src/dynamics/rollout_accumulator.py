"""Streaming adapter that turns per-step env transitions into episodes.

Env collection produces transitions one at a time, but ``EpisodeStore`` is
episode-keyed. This class buffers the in-progress episode in lists and flushes
it to the store on ``finish_episode``.

Convention: an episode has ``T+1`` states and ``T`` actions. Callers seed the
first state with ``start_episode``, then call ``append_step`` per env step
with ``(action, next_visual, next_proprio)``.
"""

from __future__ import annotations

import numpy as np

from src.dynamics.episode_store import EpisodeStore


class RolloutAccumulator:
    def __init__(self, store: EpisodeStore) -> None:
        self._store = store
        self._visual: list[np.ndarray] = []
        self._proprio: list[np.ndarray] = []
        self._action: list[np.ndarray] = []

    @property
    def in_progress(self) -> bool:
        return len(self._visual) > 0

    @property
    def steps(self) -> int:
        """Number of completed transitions in the in-progress episode."""
        return len(self._action)

    def start_episode(self, visual: np.ndarray, proprio: np.ndarray) -> None:
        if self.in_progress:
            raise RuntimeError(
                "start_episode called while an episode is in progress; "
                "call finish_episode or discard first"
            )
        self._visual.append(np.asarray(visual, dtype=np.float32).reshape(-1))
        self._proprio.append(np.asarray(proprio, dtype=np.float32).reshape(-1))

    def append_step(
        self,
        action: np.ndarray,
        next_visual: np.ndarray,
        next_proprio: np.ndarray,
    ) -> None:
        if not self.in_progress:
            raise RuntimeError("append_step called before start_episode")
        self._action.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._visual.append(np.asarray(next_visual, dtype=np.float32).reshape(-1))
        self._proprio.append(np.asarray(next_proprio, dtype=np.float32).reshape(-1))

    def finish_episode(self, *, pinned: bool = False) -> None:
        """Flush the in-progress episode to the store and reset."""
        if not self.in_progress:
            raise RuntimeError("finish_episode called with no in-progress episode")
        if not self._action:
            raise RuntimeError("finish_episode called with zero-step episode")
        visual = np.stack(self._visual, axis=0)
        proprio = np.stack(self._proprio, axis=0)
        action = np.stack(self._action, axis=0)
        self._reset()
        self._store.add_episode(visual, proprio, action, pinned=pinned)

    def discard(self) -> None:
        """Drop the in-progress episode without flushing."""
        self._reset()

    def _reset(self) -> None:
        self._visual.clear()
        self._proprio.clear()
        self._action.clear()
