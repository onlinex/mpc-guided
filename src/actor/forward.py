"""Forward dynamics model with a self-surprise head.

Shared trunk, two heads:
- ``state_head``: predicts next state from (state, action) — the dynamics signal
  that trains the trunk + state_head and drives the actor's H-step rollout.
- ``surprise_head``: predicts the per-sample MSE of the state_head's own next-state
  prediction. Its input is the trunk's hidden activation *detached*, so the head's
  loss only updates its own weights — it can never tug the predictor toward
  minimizing its own self-reported error. softplus keeps the prediction >= 0.

The head is a passive estimator: at training time the target is the detached
per-sample MSE of ``state_head``; at actor-rollout time we read its prediction
to log how surprised the model expects to be along the imagined trajectory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    """``(state, action) -> (next_state, predicted_surprise)``.

    Trunk is 2x256 ReLU. ``state_head`` is a raw linear to ``state_dim``;
    ``surprise_head`` is a single softplus scalar.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden, state_dim)
        self.surprise_head = nn.Linear(hidden, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([state, action], dim=-1))
        next_state = self.state_head(h)
        pred_surprise = F.softplus(self.surprise_head(h.detach())).squeeze(-1)
        return next_state, pred_surprise
