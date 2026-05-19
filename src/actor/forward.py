"""Forward dynamics model with a self-modeling surprise head.

Shared trunk, two heads:
- ``state_head``: predicts next state from (state, action) — the dynamics
  signal that trains the trunk + state_head and drives the actor's H-step
  rollout.
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

    Trunk is a single ``hidden``-wide layer; ``state_head`` is a 2-layer MLP
    (so most of the predictive capacity lives in the specialized head, not
    the shared encoder); ``surprise_head`` is a softplus scalar from a
    2-layer MLP — non-negative estimate of the per-sample L2 prediction
    error in real units.

    ``surprise_momentum`` is the EMA "new weight" (so effective averaging window
    is ~``1 / momentum`` updates). Default 1e-3 → settles on the last ~1k
    dynamics-step updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        surprise_momentum: float = 1e-3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.surprise_momentum = float(surprise_momentum)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
        )
        # 2-layer head holds the bulk of next-state prediction capacity,
        # rather than burying it in a wide shared trunk that the surprise
        # head can't read into (h is detached before the surprise head).
        self.state_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.surprise_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # EMA of actual per-sample MSE and its second moment. Init to (0, 1)
        # so an untrained model normalizes to ~sigmoid(actual) until the EMA
        # warms up over the first few thousand updates.
        self.register_buffer("surprise_mean", torch.zeros(()))
        self.register_buffer("surprise_sq_mean", torch.ones(()))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        detach_surprise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``detach_surprise=True`` (default) cuts the gradient path from the
        surprise output back through the trunk — used during the dynamics step
        so the head's loss can't tug the predictor. Pass ``False`` from the
        actor rollout if you want surprise to be a *signal the actor can
        optimize through* (e.g., maximize-surprise regularizer); the path then
        becomes ``surprise -> trunk -> action -> actor params``.
        """
        h = self.trunk(torch.cat([state, action], dim=-1))
        next_state = self.state_head(h)
        h_for_surp = h.detach() if detach_surprise else h
        surprise = F.softplus(self.surprise_head(h_for_surp)).squeeze(-1)
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
