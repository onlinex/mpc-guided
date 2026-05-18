"""Minimal forward dynamics model. Predicts next state from (state, action).

Trained alongside the BC actor on the same per-episode dataset but with its
own optimizer — gradients are isolated, so this is a pure passenger that
doesn't influence policy learning. Useful as a diagnostic baseline (how
predictable is the env transition from state?) and as a starting point for
later experiments that actually use it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """``(state, action) -> next_state``.

    Same MLP shape as ``src.actor.Actor`` for consistency: two 256-wide
    hidden layers with ReLU, raw linear output (no squashing).
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
