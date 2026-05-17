"""Forward dynamics model with separated visual + proprio streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.networks import MLP


@dataclass(frozen=True)
class ForwardDynamicsConfig:
    """Architecture settings for a one-step (visual, proprio) -> next forward model."""

    action_dim: int
    visual_dim: int = 512
    proprio_dim: int = 0
    hidden_dims: tuple[int, ...] = (1024, 1024)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0


class ForwardDynamicsModel(nn.Module):
    """Predict next ``(visual, proprio)`` from current ``(visual, proprio, action)``.

    The two input streams are concatenated internally to drive a single residual
    MLP, but the public API keeps them separate so the model can later wrap
    each with its own encoder (e.g. a compressive head on proprio) without
    callers needing to change.
    """

    def __init__(self, config: ForwardDynamicsConfig) -> None:
        super().__init__()
        if config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {config.action_dim}")
        if config.visual_dim < 1:
            raise ValueError(f"visual_dim must be >= 1, got {config.visual_dim}")
        if config.proprio_dim < 0:
            raise ValueError(f"proprio_dim must be >= 0, got {config.proprio_dim}")
        self.config = config
        in_dim = config.visual_dim + config.proprio_dim + config.action_dim
        out_dim = config.visual_dim + config.proprio_dim
        self.net = MLP(
            in_dim,
            out_dim,
            config.hidden_dims,
            activation_name=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    @property
    def visual_dim(self) -> int:
        return self.config.visual_dim

    @property
    def proprio_dim(self) -> int:
        return self.config.proprio_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def forward(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(visual, proprio, action)
        x = torch.cat([visual, proprio, action], dim=-1)
        delta = self.net(x)
        delta_visual = delta[:, : self.visual_dim]
        delta_proprio = delta[:, self.visual_dim :]
        return visual + delta_visual, proprio + delta_proprio

    def _validate(self, visual: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> None:
        for name, tensor, expected in (
            ("visual", visual, self.visual_dim),
            ("proprio", proprio, self.proprio_dim),
            ("action", action, self.action_dim),
        ):
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must be rank-2 (B, {expected}), got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != expected:
                raise ValueError(f"{name} dim must be {expected}, got {tensor.shape[1]}")
        if not (visual.shape[0] == proprio.shape[0] == action.shape[0]):
            raise ValueError(
                f"batch sizes differ: visual={visual.shape[0]} proprio={proprio.shape[0]} "
                f"action={action.shape[0]}"
            )
