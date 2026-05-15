"""State-conditioned stochastic actor used for environment data collection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.actor.config import ActorSample, StochasticActorConfig
from src.networks import MLP


class StochasticActor(nn.Module):
    """Squashed Gaussian policy mapping R3M states to robot actions.

    The network predicts a Gaussian in pre-tanh action space. Samples are
    squashed through tanh and linearly scaled into the ManiSkill action bounds.
    This keeps rollout actions valid while preserving a simple stochastic
    policy interface that can be trained later.
    """

    def __init__(
        self,
        config: StochasticActorConfig,
        *,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {config.action_dim}")
        if config.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {config.state_dim}")
        if config.log_std_min >= config.log_std_max:
            raise ValueError(
                f"log_std_min must be < log_std_max, got "
                f"{config.log_std_min} >= {config.log_std_max}"
            )
        action_low = action_low.to(dtype=torch.float32).reshape(-1)
        action_high = action_high.to(dtype=torch.float32).reshape(-1)
        if action_low.shape != (config.action_dim,):
            raise ValueError(f"action_low shape must be {(config.action_dim,)}, got {action_low.shape}")
        if action_high.shape != (config.action_dim,):
            raise ValueError(
                f"action_high shape must be {(config.action_dim,)}, got {action_high.shape}"
            )
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise ValueError("StochasticActor requires finite action bounds")
        if not torch.all(action_high > action_low):
            raise ValueError("every action_high entry must be greater than action_low")

        self.config = config
        self.net = MLP(
            config.state_dim,
            2 * config.action_dim,
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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_state(state)
        raw_mean, raw_log_std = self.net(state).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        return raw_mean, log_std

    def sample(self, state: torch.Tensor, *, deterministic: bool = False) -> ActorSample:
        raw_mean, log_std = self(state)
        std = log_std.exp()
        if deterministic:
            raw_action = raw_mean
        else:
            raw_action = raw_mean + std * torch.randn_like(std)
        action = self._scale_action(torch.tanh(raw_action))
        mean = self._scale_action(torch.tanh(raw_mean))
        action = action.clamp(self.action_low, self.action_high)
        return ActorSample(
            action=action,
            mean=mean,
            std=std,
            raw_mean=raw_mean,
            log_std=log_std,
        )

    def _scale_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        return unit_action * self.action_scale + self.action_bias

    def _validate_state(self, state: torch.Tensor) -> None:
        if state.ndim != 2:
            raise ValueError(f"state must be rank-2 (B, state_dim), got shape {tuple(state.shape)}")
        if state.shape[1] != self.state_dim:
            raise ValueError(f"state dim must be {self.state_dim}, got {state.shape[1]}")
