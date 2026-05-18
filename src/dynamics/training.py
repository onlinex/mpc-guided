"""Training utilities for forward dynamics models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.dynamics.episode_store import WindowBatch
from src.dynamics.model import ForwardDynamicsModel


@dataclass(frozen=True)
class DynamicsTrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 10.0
    proprio_loss_weight: float = 1.0
    rollout_discount: float = 1.0  # per-step weight in multi-step loss; 1.0 = uniform.


class DynamicsTrainer:
    """Owns optimization for a ``ForwardDynamicsModel`` with split losses.

    ``train_step`` accepts any rollout horizon: it rolls the model forward
    ``H`` steps from the last context frame, feeds its own predictions back
    in as state (teacher-forced actions only), and averages per-step MSE.
    ``H=1`` reduces exactly to the one-step loss used pre-migration.
    """

    def __init__(
        self,
        model: ForwardDynamicsModel,
        config: DynamicsTrainerConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def train_step(self, batch: WindowBatch) -> dict[str, float]:
        self.model.train()
        visual = batch.visual_context[:, -1]
        proprio = batch.proprio_context[:, -1]
        horizon = batch.action.shape[1]
        gamma = self.config.rollout_discount

        visual_loss = visual.new_zeros(())
        proprio_loss = visual.new_zeros(())
        weight_total = 0.0
        for h in range(horizon):
            pred_visual, pred_proprio = self.model(visual, proprio, batch.action[:, h])
            w = gamma ** h
            visual_loss = visual_loss + w * F.mse_loss(pred_visual, batch.visual_future[:, h])
            proprio_loss = proprio_loss + w * F.mse_loss(pred_proprio, batch.proprio_future[:, h])
            weight_total += w
            visual, proprio = pred_visual, pred_proprio  # rolled-out, NOT teacher-forced

        visual_loss = visual_loss / weight_total
        proprio_loss = proprio_loss / weight_total
        loss = visual_loss + self.config.proprio_loss_weight * proprio_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "visual_loss": float(visual_loss.detach().cpu()),
            "proprio_loss": float(proprio_loss.detach().cpu()),
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def eval_step(self, batch: WindowBatch) -> dict[str, float]:
        visual, proprio, action, next_visual, next_proprio = _single_step(batch)
        self.model.eval()
        pred_visual, pred_proprio = self.model(visual, proprio, action)
        visual_loss = F.mse_loss(pred_visual, next_visual)
        proprio_loss = F.mse_loss(pred_proprio, next_proprio)
        loss = visual_loss + self.config.proprio_loss_weight * proprio_loss
        cosine = F.cosine_similarity(pred_visual, next_visual, dim=-1).mean()
        identity_visual = F.mse_loss(visual, next_visual)
        identity_proprio = F.mse_loss(proprio, next_proprio)
        return {
            "loss": float(loss.cpu()),
            "visual_loss": float(visual_loss.cpu()),
            "proprio_loss": float(proprio_loss.cpu()),
            "visual_cosine": float(cosine.cpu()),
            "visual_identity_loss_ratio": float(
                (visual_loss / identity_visual.clamp(min=1e-12)).cpu()
            ),
            "proprio_identity_loss_ratio": float(
                (proprio_loss / identity_proprio.clamp(min=1e-12)).cpu()
            ),
        }

    def _clip_grad_norm(self) -> float:
        if self.config.grad_clip_norm is None:
            return self._grad_norm()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.grad_clip_norm,
        )
        return float(norm.detach().cpu())

    def _grad_norm(self) -> float:
        norms = [
            p.grad.detach().norm(2)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.stack(norms).norm(2).cpu())


def _single_step(
    batch: WindowBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse a (context=1, horizon=1) WindowBatch into (s, p, a, s', p') tensors."""
    if batch.visual_context.shape[1] != 1 or batch.visual_future.shape[1] != 1:
        raise ValueError(
            f"single-step trainer expects context=1, horizon=1; got context="
            f"{batch.visual_context.shape[1]}, horizon={batch.visual_future.shape[1]}"
        )
    return (
        batch.visual_context[:, 0],
        batch.proprio_context[:, 0],
        batch.action[:, 0],
        batch.visual_future[:, 0],
        batch.proprio_future[:, 0],
    )
