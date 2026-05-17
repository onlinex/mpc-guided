"""Replay buffer for one-step visual+proprio dynamics transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DynamicsBatch:
    """A sampled batch of one-step transition data with separated streams."""

    visual: torch.Tensor      # (B, visual_dim)
    proprio: torch.Tensor     # (B, proprio_dim)
    action: torch.Tensor      # (B, action_dim)
    next_visual: torch.Tensor  # (B, visual_dim)
    next_proprio: torch.Tensor  # (B, proprio_dim)


class TransitionReplayBuffer:
    """Fixed-size circular buffer for ``(visual, proprio, action, next_visual, next_proprio)`` transitions.

    Visual and proprioceptive components are stored in separate arrays so the
    dynamics model can treat them as distinct inputs/outputs (and so we can
    later add e.g. a proprio encoder without re-plumbing the buffer).
    """

    def __init__(
        self,
        capacity: int,
        *,
        visual_dim: int,
        proprio_dim: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self._visual = np.empty((capacity, visual_dim), dtype=np.float32)
        self._proprio = np.empty((capacity, proprio_dim), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._next_visual = np.empty((capacity, visual_dim), dtype=np.float32)
        self._next_proprio = np.empty((capacity, proprio_dim), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._size = 0
        self._pinned = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    @property
    def pinned(self) -> int:
        return self._pinned

    def pin_current_contents(self) -> None:
        """Mark all currently-stored transitions as never-evictable.

        Subsequent ``add`` calls cycle through ``[pinned, capacity)`` only, so
        these entries survive forever. Pinned entries still count toward
        ``__len__`` and are sampled like any other transition.
        """
        if self._size > self.capacity - 1:
            raise ValueError(
                f"cannot pin {self._size} entries: need at least 1 unpinned slot "
                f"(capacity={self.capacity})"
            )
        self._pinned = self._size
        self._cursor = self._size % self.capacity

    def add(
        self,
        visual: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        next_visual: np.ndarray,
        next_proprio: np.ndarray,
    ) -> None:
        visual = self._check(visual, self.visual_dim, "visual")
        proprio = self._check(proprio, self.proprio_dim, "proprio")
        action = self._check(action, self.action_dim, "action")
        next_visual = self._check(next_visual, self.visual_dim, "next_visual")
        next_proprio = self._check(next_proprio, self.proprio_dim, "next_proprio")

        self._visual[self._cursor] = visual
        self._proprio[self._cursor] = proprio
        self._actions[self._cursor] = action
        self._next_visual[self._cursor] = next_visual
        self._next_proprio[self._cursor] = next_proprio
        next_cursor = self._cursor + 1
        if next_cursor >= self.capacity:
            next_cursor = self._pinned
        self._cursor = next_cursor
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> DynamicsBatch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return self._gather(indices, device)

    def sample_expert(self, batch_size: int, device: torch.device) -> DynamicsBatch | None:
        """Sample only from the pinned (expert) region. Returns None if empty."""
        if self._pinned == 0:
            return None
        indices = self._rng.integers(0, self._pinned, size=batch_size)
        return self._gather(indices, device)

    def sample_on_policy(self, batch_size: int, device: torch.device) -> DynamicsBatch | None:
        """Sample only from the un-pinned (on-policy) region. Returns None if empty."""
        on_policy_size = self._size - self._pinned
        if on_policy_size <= 0:
            return None
        indices = self._pinned + self._rng.integers(0, on_policy_size, size=batch_size)
        return self._gather(indices, device)

    def _gather(self, indices: np.ndarray, device: torch.device) -> DynamicsBatch:
        return DynamicsBatch(
            visual=torch.as_tensor(self._visual[indices], device=device),
            proprio=torch.as_tensor(self._proprio[indices], device=device),
            action=torch.as_tensor(self._actions[indices], device=device),
            next_visual=torch.as_tensor(self._next_visual[indices], device=device),
            next_proprio=torch.as_tensor(self._next_proprio[indices], device=device),
        )

    @staticmethod
    def _check(arr: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape != (expected_dim,):
            raise ValueError(f"{name} shape must be {(expected_dim,)}, got {arr.shape}")
        return arr
