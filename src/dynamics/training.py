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


class DynamicsTrainer:
    """Owns optimization for a ``ForwardDynamicsModel``."""

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
        pred_next_state = self.model(batch.state, batch.action)
        loss = F.mse_loss(pred_next_state, batch.next_state)
        delta_loss = F.mse_loss(
            pred_next_state - batch.state,
            batch.next_state - batch.state,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def eval_step(self, batch: DynamicsBatch) -> dict[str, float]:
        self.model.eval()
        pred_next_state = self.model(batch.state, batch.action)
        loss = F.mse_loss(pred_next_state, batch.next_state)
        delta = pred_next_state - batch.state
        target_delta = batch.next_state - batch.state
        delta_loss = F.mse_loss(delta, target_delta)
        cosine = F.cosine_similarity(pred_next_state, batch.next_state, dim=-1).mean()
        return {
            "loss": float(loss.cpu()),
            "delta_loss": float(delta_loss.cpu()),
            "cosine": float(cosine.cpu()),
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
