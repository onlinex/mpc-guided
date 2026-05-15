"""Replay buffer for one-step visual dynamics transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DynamicsBatch:
    """A sampled batch of one-step transition data."""

    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor


class TransitionReplayBuffer:
    """Fixed-size circular buffer for ``(state, action, next_state)`` transitions."""

    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int,
        action_dim: int,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state shape must be {(self.state_dim,)}, got {state.shape}")
        if action.shape != (self.action_dim,):
            raise ValueError(f"action shape must be {(self.action_dim,)}, got {action.shape}")
        if next_state.shape != (self.state_dim,):
            raise ValueError(
                f"next_state shape must be {(self.state_dim,)}, got {next_state.shape}"
            )

        self._states[self._cursor] = state
        self._actions[self._cursor] = action
        self._next_states[self._cursor] = next_state
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> DynamicsBatch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return DynamicsBatch(
            state=torch.as_tensor(self._states[indices], device=device),
            action=torch.as_tensor(self._actions[indices], device=device),
            next_state=torch.as_tensor(self._next_states[indices], device=device),
        )
