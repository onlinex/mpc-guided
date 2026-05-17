"""Configuration types for the actor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorConfig:
    """Configuration for a deterministic actor over bounded actions.

    The actor accepts ``(visual, proprio)`` as separate inputs (concatenated
    internally) so we can later replace the raw proprio with a learned encoder
    without changing callers.
    """

    action_dim: int
    visual_dim: int = 512
    proprio_dim: int = 0
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
