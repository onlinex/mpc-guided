"""Unit tests for ForwardDynamicsModel."""

from __future__ import annotations

import pytest
import torch

from src.dynamics.model import ForwardDynamicsConfig, ForwardDynamicsModel


def _make(visual_dim=8, proprio_dim=3, action_dim=4):
    return ForwardDynamicsModel(
        ForwardDynamicsConfig(
            action_dim=action_dim,
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            hidden_dims=(16, 16),
        )
    )


def test_forward_shape():
    model = _make()
    visual = torch.randn(5, model.visual_dim)
    proprio = torch.randn(5, model.proprio_dim)
    action = torch.randn(5, model.action_dim)
    out_visual, out_proprio = model(visual, proprio, action)
    assert out_visual.shape == visual.shape
    assert out_proprio.shape == proprio.shape


def test_starts_at_identity():
    """With zero-init delta head, pred = (visual, proprio) exactly at init.

    This is the invariant the recent dynamics fix added. If this test fails,
    the model is no longer starting at identity and visual_identity_loss_ratio
    will start far from 1.0.
    """
    model = _make()
    visual = torch.randn(7, model.visual_dim)
    proprio = torch.randn(7, model.proprio_dim)
    action = torch.randn(7, model.action_dim)
    out_visual, out_proprio = model(visual, proprio, action)
    torch.testing.assert_close(out_visual, visual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_proprio, proprio, atol=0.0, rtol=0.0)


def test_validates_input_shape():
    model = _make()
    with pytest.raises(ValueError):
        model(
            torch.randn(5, model.visual_dim + 1),
            torch.randn(5, model.proprio_dim),
            torch.randn(5, model.action_dim),
        )


def test_validates_batch_alignment():
    model = _make()
    with pytest.raises(ValueError):
        model(
            torch.randn(5, model.visual_dim),
            torch.randn(4, model.proprio_dim),
            torch.randn(5, model.action_dim),
        )
