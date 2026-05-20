"""State-based BC actor.

Plain MLP, ReLU activations, raw (unsquashed) output. This is the canonical
ManiSkill state-based BC architecture and matches the reference baseline in
train_bc_baseline.py byte-for-byte. Kept here so train.py doesn't need to
import from a sibling top-level script.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.actor.temporal import TemporalConv1d


class Actor(nn.Module):
    """state_dim -> encoder -> latent (the plan bottleneck) -> chunk decoder
    -> (chunk_size, action_dim).

    The encoder is a 2-layer 256-wide MLP that compresses state into a
    ``latent_dim`` plan (linear bottleneck, no activation). The decoder
    expands that plan into a length-``chunk_size`` action sequence:

      latent (B, latent_dim)
        -> broadcast + positional embedding -> (B, N, latent_dim)
        -> TemporalConv1d(k) + ReLU                 (smoothness prior)
        -> per-step shared Linear(latent_dim, action_dim)

    The temporal conv shares weights across positions, so the decoder pays
    a representational cost to make neighbouring actions wildly different
    — a real smoothness prior, not just a loss penalty.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 1,
        latent_dim: int = 32,
        conv_kernel: int = 3,
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
        # Same plan, different time slot: positional embedding is the only
        # source of position-specific structure before the conv mixes
        # neighbours. Init small so all positions start ~at the latent and
        # training spreads them apart as needed.
        self.pos_embed = nn.Parameter(torch.zeros(chunk_size, latent_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.temporal = TemporalConv1d(latent_dim, latent_dim, conv_kernel)
        self.head = nn.Linear(latent_dim, action_dim)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """state -> latent. Split out so callers can perturb the latent
        (e.g., exploration noise in the bottleneck) before decoding."""
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent (B, latent_dim) -> chunk (B, chunk_size, action_dim)."""
        seq = latent.unsqueeze(1) + self.pos_embed  # (B, N, latent_dim)
        seq = F.relu(self.temporal(seq))
        return self.head(seq)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))
