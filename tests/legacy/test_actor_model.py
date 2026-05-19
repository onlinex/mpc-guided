"""Unit tests for the Actor module."""

from __future__ import annotations

import torch

from src.legacy.actor import Actor, ActorConfig


def _make(action_dim=4, visual_dim=6, proprio_dim=3, low=-1.0, high=1.0):
    return Actor(
        ActorConfig(
            action_dim=action_dim,
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            hidden_dims=(16, 16),
        ),
        action_low=torch.full((action_dim,), low, dtype=torch.float32),
        action_high=torch.full((action_dim,), high, dtype=torch.float32),
    )


def test_output_shape():
    actor = _make()
    visual = torch.randn(7, actor.config.visual_dim)
    proprio = torch.randn(7, actor.config.proprio_dim)
    out = actor(visual, proprio)
    assert out.shape == (7, actor.config.action_dim)


def test_output_within_bounds():
    actor = _make(low=-0.5, high=0.5)
    visual = torch.randn(64, actor.config.visual_dim) * 100  # extreme input
    proprio = torch.randn(64, actor.config.proprio_dim) * 100
    out = actor(visual, proprio)
    assert (out >= -0.5 - 1e-6).all() and (out <= 0.5 + 1e-6).all()


def test_deterministic():
    actor = _make()
    actor.eval()
    visual = torch.randn(4, actor.config.visual_dim)
    proprio = torch.randn(4, actor.config.proprio_dim)
    with torch.no_grad():
        a = actor(visual, proprio)
        b = actor(visual, proprio)
    torch.testing.assert_close(a, b)
