"""State-based BC actor.

Plain MLP, ReLU activations, raw (unsquashed) output. This is the canonical
ManiSkill state-based BC architecture and matches the reference baseline in
train_bc_baseline.py byte-for-byte. Kept here so train.py doesn't need to
import from a sibling top-level script.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """state_dim -> 256 -> ReLU -> 256 -> ReLU -> action_dim."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
