"""Configuration and value types for actor modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StochasticActorConfig:
    """Configuration for a squashed Gaussian actor over bounded actions."""

    action_dim: int
    state_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 0.5


@dataclass(frozen=True)
class ActorSample:
    """A sampled bounded action and diagnostics from the actor distribution."""

    action: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    raw_mean: torch.Tensor
    log_std: torch.Tensor
