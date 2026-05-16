"""Deterministic state-conditioned policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.actor.config import ActorConfig
from src.networks import MLP


class Actor(nn.Module):
    """Deterministic policy mapping states to bounded actions.

    The MLP predicts pre-tanh action means; outputs are squashed through tanh
    and linearly scaled into the env's action bounds. Exploration noise (e.g.
    AR(1)/OU) is applied externally by the env-collection loop.
    """

    def __init__(
        self,
        config: ActorConfig,
        *,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {config.action_dim}")
        if config.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {config.state_dim}")
        action_low = action_low.to(dtype=torch.float32).reshape(-1)
        action_high = action_high.to(dtype=torch.float32).reshape(-1)
        if action_low.shape != (config.action_dim,):
            raise ValueError(
                f"action_low shape must be {(config.action_dim,)}, got {action_low.shape}"
            )
        if action_high.shape != (config.action_dim,):
            raise ValueError(
                f"action_high shape must be {(config.action_dim,)}, got {action_high.shape}"
            )
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise ValueError("Actor requires finite action bounds")
        if not torch.all(action_high > action_low):
            raise ValueError("every action_high entry must be greater than action_low")

        self.config = config
        self.net = MLP(
            config.state_dim,
            config.action_dim,
            config.hidden_dims,
            activation_name=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"state must be rank-2 (B, state_dim), got shape {tuple(state.shape)}")
        if state.shape[1] != self.state_dim:
            raise ValueError(f"state dim must be {self.state_dim}, got {state.shape[1]}")
        raw = self.net(state)
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        return action.clamp(self.action_low, self.action_high)
