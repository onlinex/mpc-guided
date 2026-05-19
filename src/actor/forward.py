"""Forward dynamics model with a self-modeling surprise head.

Shared trunk, two heads:
- ``state_head``: predicts next state from (state, action) — the dynamics signal
  that trains the trunk + state_head and drives the actor's H-step rollout.
- ``surprise_head``: a non-negative scalar (softplus) estimating the per-sample
  L2 prediction error of ``state_head``. Trained against the detached actual
  MSE; the trunk's hidden activation is detached before the surprise head, so
  the head's loss only updates its own weights and never tugs the predictor.

The head outputs raw error in real units. For an interpretable [0, 1] view —
"is this more surprising than the recent training-time average?" — call
``normalize_actual(raw)`` which sigmoid-standardizes against EMA buffers
``surprise_mean`` / ``surprise_sq_mean``. The buffers live in ``state_dict``,
so a checkpoint is self-contained: load it and the normalization is already
calibrated, no external state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    """``(state, action) -> (next_state, surprise)``.

    Trunk is 2x256 ReLU. ``state_head`` is a raw linear to ``state_dim``;
    ``surprise_head`` is a single softplus scalar — non-negative estimate of
    the per-sample L2 prediction error in real units.

    ``surprise_momentum`` is the EMA "new weight" (so effective averaging window
    is ~``1 / momentum`` updates). Default 1e-4 → settles on the last ~10k
    dynamics-step updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        surprise_momentum: float = 1e-4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.surprise_momentum = float(surprise_momentum)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden, state_dim)
        self.surprise_head = nn.Linear(hidden, 1)
        # EMA of actual per-sample MSE and its second moment. Init to (0, 1)
        # so an untrained model normalizes to ~sigmoid(actual) until the EMA
        # warms up over the first few thousand updates.
        self.register_buffer("surprise_mean", torch.zeros(()))
        self.register_buffer("surprise_sq_mean", torch.ones(()))

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([state, action], dim=-1))
        next_state = self.state_head(h)
        surprise = F.softplus(self.surprise_head(h.detach())).squeeze(-1)
        return next_state, surprise

    @torch.no_grad()
    def update_surprise_stats(self, actual: torch.Tensor) -> None:
        """EMA-update running stats from a batch of detached per-sample MSE."""
        m = self.surprise_momentum
        batch_mean = actual.mean().detach()
        batch_sq_mean = actual.pow(2).mean().detach()
        self.surprise_mean.mul_(1 - m).add_(m * batch_mean)
        self.surprise_sq_mean.mul_(1 - m).add_(m * batch_sq_mean)

    def normalize_actual(self, actual: torch.Tensor) -> torch.Tensor:
        """Sigmoid-standardize raw surprise against the EMA.

        Returns a [0, 1] reading where 0.5 ≈ "as surprised as the recent
        training-time average," >0.5 = unusually high, <0.5 = unusually low.
        Same call works on a single raw scalar from the head or on an actual
        per-sample MSE — they live in the same units.
        """
        var = (self.surprise_sq_mean - self.surprise_mean.pow(2)).clamp_min(1e-8)
        return torch.sigmoid((actual - self.surprise_mean) / var.sqrt())
