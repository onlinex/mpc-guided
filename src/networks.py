"""Shared neural network building blocks."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def activation(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "silu":
            return nn.SiLU()
        case "gelu":
            return nn.GELU()
        case "tanh":
            return nn.Tanh()
        case _:
            raise ValueError(f"Unsupported activation={name!r}")


class MLP(nn.Module):
    """Simple configurable MLP for policy and dynamics heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        *,
        activation_name: str,
        layer_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation(activation_name))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
