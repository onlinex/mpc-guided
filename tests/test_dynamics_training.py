"""Tests for DynamicsTrainer multi-step rollout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.dynamics.episode_store import EpisodeStore
from src.dynamics.model import ForwardDynamicsConfig, ForwardDynamicsModel
from src.dynamics.training import DynamicsTrainer, DynamicsTrainerConfig

DEVICE = torch.device("cpu")
VISUAL_DIM = 8
PROPRIO_DIM = 4
ACTION_DIM = 3


def _trainer(*, rollout_discount=1.0, proprio_loss_weight=1.0, lr=0.0, seed=0):
    """A trainer with lr=0 by default so train_step is a pure loss computation."""
    torch.manual_seed(seed)
    model = ForwardDynamicsModel(
        ForwardDynamicsConfig(
            action_dim=ACTION_DIM,
            visual_dim=VISUAL_DIM,
            proprio_dim=PROPRIO_DIM,
            hidden_dims=(32, 32),
        )
    )
    cfg = DynamicsTrainerConfig(
        lr=lr,
        weight_decay=0.0,
        grad_clip_norm=None,
        proprio_loss_weight=proprio_loss_weight,
        rollout_discount=rollout_discount,
    )
    return DynamicsTrainer(model, cfg)


def _store_with_episode(T):
    store = EpisodeStore(
        capacity_transitions=200,
        visual_dim=VISUAL_DIM,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        seed=0,
    )
    rng = np.random.default_rng(0)
    visual = rng.normal(size=(T + 1, VISUAL_DIM)).astype(np.float32)
    proprio = rng.normal(size=(T + 1, PROPRIO_DIM)).astype(np.float32)
    action = rng.normal(size=(T, ACTION_DIM)).astype(np.float32) * 0.1
    store.add_episode(visual, proprio, action, pinned=True)
    return store


def test_horizon_one_matches_manual_single_step_mse():
    """train_step with H=1 must equal a hand-computed (mse(visual) + w * mse(proprio))."""
    trainer = _trainer(proprio_loss_weight=0.5)
    store = _store_with_episode(20)
    batch = store.sample(8, horizon=1, device=DEVICE)
    metrics = trainer.train_step(batch)

    # Recompute by hand.
    trainer.model.eval()
    with torch.no_grad():
        pred_v, pred_p = trainer.model(
            batch.visual_context[:, -1], batch.proprio_context[:, -1], batch.action[:, 0]
        )
        vl = F.mse_loss(pred_v, batch.visual_future[:, 0])
        pl = F.mse_loss(pred_p, batch.proprio_future[:, 0])
    # The trainer's reported loss reflects the model BEFORE the update; with lr=0 we can
    # just reproduce it from the same params.
    assert abs(metrics["visual_loss"] - float(vl)) < 1e-6
    assert abs(metrics["proprio_loss"] - float(pl)) < 1e-6
    assert abs(metrics["loss"] - float(vl + 0.5 * pl)) < 1e-6


def test_multi_step_rolls_predictions_back_in():
    """The H-step loss must match a manual rollout that feeds predictions back as state."""
    trainer = _trainer(proprio_loss_weight=1.0)
    store = _store_with_episode(20)
    H = 4
    batch = store.sample(8, horizon=H, device=DEVICE)
    metrics = trainer.train_step(batch)

    # Manual rollout with the same (post-zero-init, untrained) model parameters.
    trainer.model.eval()
    with torch.no_grad():
        v = batch.visual_context[:, -1]
        p = batch.proprio_context[:, -1]
        vl_sum, pl_sum = 0.0, 0.0
        for h in range(H):
            pv, pp = trainer.model(v, p, batch.action[:, h])
            vl_sum += float(F.mse_loss(pv, batch.visual_future[:, h]))
            pl_sum += float(F.mse_loss(pp, batch.proprio_future[:, h]))
            v, p = pv, pp
        vl_expected = vl_sum / H
        pl_expected = pl_sum / H

    assert abs(metrics["visual_loss"] - vl_expected) < 1e-5
    assert abs(metrics["proprio_loss"] - pl_expected) < 1e-5


def test_rollout_discount_weights_first_step_more():
    """With gamma<1, later steps contribute less; gamma=0.5 puts ~67% weight on step 0 at H=2."""
    trainer = _trainer(rollout_discount=0.5)
    store = _store_with_episode(20)
    batch = store.sample(8, horizon=2, device=DEVICE)
    metrics = trainer.train_step(batch)

    trainer.model.eval()
    with torch.no_grad():
        v = batch.visual_context[:, -1]
        p = batch.proprio_context[:, -1]
        v0, p0 = trainer.model(v, p, batch.action[:, 0])
        v1, p1 = trainer.model(v0, p0, batch.action[:, 1])
        vl0 = float(F.mse_loss(v0, batch.visual_future[:, 0]))
        vl1 = float(F.mse_loss(v1, batch.visual_future[:, 1]))
        # weights 1.0, 0.5; total 1.5
        vl_expected = (vl0 + 0.5 * vl1) / 1.5

    assert abs(metrics["visual_loss"] - vl_expected) < 1e-5


def test_multi_step_loss_decreases_on_overfit():
    """One small episode + many updates should drive multi-step loss down."""
    trainer = _trainer(lr=3e-3)
    store = _store_with_episode(8)
    H = 3
    batch = store.sample(64, horizon=H, device=DEVICE)
    first_loss = trainer.train_step(batch)["loss"]
    for _ in range(300):
        trainer.train_step(batch)
    final_loss = trainer.train_step(batch)["loss"]
    assert final_loss < first_loss * 0.5, f"loss did not decrease enough: {first_loss} -> {final_loss}"
