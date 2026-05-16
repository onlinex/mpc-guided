"""Configuration types for the actor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorConfig:
    """Configuration for a deterministic actor over bounded actions."""

    action_dim: int
    state_dim: int = 512
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
