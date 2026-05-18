"""Deterministic policy with separated visual + proprio inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.legacy.actor.config import ActorConfig
from src.networks import MLP


class Actor(nn.Module):
    """Deterministic policy mapping ``(visual, proprio)`` to bounded actions.

    The two input streams are concatenated internally to drive a single MLP
    whose pre-tanh output is squashed and linearly scaled into the env's
    action bounds. Exploration noise (e.g. AR(1)/OU) is applied externally
    by the env-collection loop.
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
        if config.visual_dim < 1:
            raise ValueError(f"visual_dim must be >= 1, got {config.visual_dim}")
        if config.proprio_dim < 0:
            raise ValueError(f"proprio_dim must be >= 0, got {config.proprio_dim}")
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
            config.visual_dim + config.proprio_dim,
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
    def visual_dim(self) -> int:
        return self.config.visual_dim

    @property
    def proprio_dim(self) -> int:
        return self.config.proprio_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        for name, tensor, expected in (
            ("visual", visual, self.visual_dim),
            ("proprio", proprio, self.proprio_dim),
        ):
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must be rank-2 (B, {expected}), got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != expected:
                raise ValueError(f"{name} dim must be {expected}, got {tensor.shape[1]}")
        if visual.shape[0] != proprio.shape[0]:
            raise ValueError(
                f"visual and proprio batch sizes differ: {visual.shape[0]} vs {proprio.shape[0]}"
            )
        x = torch.cat([visual, proprio], dim=-1)
        raw = self.net(x)
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        return action.clamp(self.action_low, self.action_high)
