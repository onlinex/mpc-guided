"""State-based BC actor.

Plain MLP, ReLU activations, raw (unsquashed) output. This is the canonical
ManiSkill state-based BC architecture and matches the reference baseline in
train_bc_baseline.py byte-for-byte. Kept here so train.py doesn't need to
import from a sibling top-level script.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """state_dim -> 256 -> ReLU -> 256 -> ReLU -> latent_dim -> chunk_size * action_dim.

    A linear bottleneck (``latent_dim``, no activation) sits between the
    encoder and the per-chunk decoder. The actor has to compress its plan
    into ``latent_dim`` numbers before expanding to ``chunk_size`` actions,
    which acts as a regularizer against emitting ``chunk_size * action_dim``
    independent scalars. Output reshaped to ``(B, chunk_size, action_dim)``;
    callers index ``out[:, 0]`` for the next action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(state)
        out = self.decoder(latent)
        return out.view(*state.shape[:-1], self.chunk_size, self.action_dim)
