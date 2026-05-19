"""Fixed-capacity FIFO ring buffer of raw (state, action_chunk, next_state) entries.

Each entry represents one ``chunk_size``-step open-loop slice through the env:
``state`` is the snapshot before the chunk, ``action_chunk`` is the
``(chunk_size, action_dim)`` sequence actually executed, and ``next_state`` is
the snapshot ``chunk_size`` env steps later. Populated by env rollouts and
sampled into the dynamics-model training step. Stores RAW (unnormalized)
states — normalization is the caller's job at sample time.
"""

from __future__ import annotations

import numpy as np


class OnlineBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 1,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, chunk_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action_chunk: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        i = self._head
        self.states[i] = state
        self.actions[i] = action_chunk
        self.next_states[i] = next_state
        self._head = (i + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n <= 0 or self._size == 0:
            empty_s = np.zeros((0, self.state_dim), dtype=np.float32)
            empty_a = np.zeros((0, self.chunk_size, self.action_dim), dtype=np.float32)
            return empty_s, empty_a, empty_s.copy()
        idx = rng.integers(0, self._size, size=n)
        return self.states[idx].copy(), self.actions[idx].copy(), self.next_states[idx].copy()
