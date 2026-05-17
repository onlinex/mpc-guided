"""Training utilities for forward dynamics models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.dynamics.buffer import DynamicsBatch
from src.dynamics.model import ForwardDynamicsModel


@dataclass(frozen=True)
class DynamicsTrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 10.0
    proprio_loss_weight: float = 1.0


class DynamicsTrainer:
    """Owns optimization for a ``ForwardDynamicsModel`` with split losses."""

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

    def train_step(self, batch: DynamicsBatch) -> dict[str, float]:
        self.model.train()
        pred_visual, pred_proprio = self.model(batch.visual, batch.proprio, batch.action)
        visual_loss = F.mse_loss(pred_visual, batch.next_visual)
        proprio_loss = F.mse_loss(pred_proprio, batch.next_proprio)
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
    def eval_step(self, batch: DynamicsBatch) -> dict[str, float]:
        self.model.eval()
        pred_visual, pred_proprio = self.model(batch.visual, batch.proprio, batch.action)
        visual_loss = F.mse_loss(pred_visual, batch.next_visual)
        proprio_loss = F.mse_loss(pred_proprio, batch.next_proprio)
        loss = visual_loss + self.config.proprio_loss_weight * proprio_loss
        cosine = F.cosine_similarity(pred_visual, batch.next_visual, dim=-1).mean()
        identity_visual = F.mse_loss(batch.visual, batch.next_visual)
        identity_proprio = F.mse_loss(batch.proprio, batch.next_proprio)
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
