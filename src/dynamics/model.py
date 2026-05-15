"""Forward dynamics model for R3M visual states and robot actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.networks import MLP


@dataclass(frozen=True)
class ForwardDynamicsConfig:
    """Architecture settings for a one-step visual forward model."""

    action_dim: int
    state_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
    predict_delta: bool = True


class ForwardDynamicsModel(nn.Module):
    """Predict the next R3M feature state from current feature state and action.

    The public output is always the predicted next state. Internally the model
    can predict a residual delta, which is usually easier to learn for R3M
    feature transitions.
    """

    def __init__(self, config: ForwardDynamicsConfig) -> None:
        super().__init__()
        if config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {config.action_dim}")
        if config.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {config.state_dim}")
        self.config = config
        self.net = MLP(
            config.state_dim + config.action_dim,
            config.state_dim,
            config.hidden_dims,
            activation_name=config.activation,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(state, action)
        model_out = self.net(torch.cat([state, action], dim=-1))
        if self.config.predict_delta:
            return state + model_out
        return model_out

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(state, action)
        prediction = self.net(torch.cat([state, action], dim=-1))
        if self.config.predict_delta:
            return prediction
        return prediction - state

    def _validate_inputs(self, state: torch.Tensor, action: torch.Tensor) -> None:
        if state.ndim != 2:
            raise ValueError(f"state must be rank-2 (B, state_dim), got shape {tuple(state.shape)}")
        if action.ndim != 2:
            raise ValueError(
                f"action must be rank-2 (B, action_dim), got shape {tuple(action.shape)}"
            )
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"state and action batch sizes differ: {state.shape[0]} vs {action.shape[0]}"
            )
        if state.shape[1] != self.state_dim:
            raise ValueError(f"state dim must be {self.state_dim}, got {state.shape[1]}")
        if action.shape[1] != self.action_dim:
            raise ValueError(f"action dim must be {self.action_dim}, got {action.shape[1]}")
