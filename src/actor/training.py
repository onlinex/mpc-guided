"""Actor training against expert video frame pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.actor.model import Actor
from src.backbone import encode_images
from src.datasets.video_pairs import VideoFramePairSampler
from src.dynamics.model import ForwardDynamicsModel


@dataclass(frozen=True)
class VideoActorTrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 10.0
    batch_size: int = 32


class VideoActorTrainer:
    """Optimizes an actor to reach goal R3M latents through learned dynamics."""

    def __init__(
        self,
        *,
        actor: Actor,
        dynamics: ForwardDynamicsModel,
        backbone: torch.nn.Module,
        sampler: VideoFramePairSampler,
        config: VideoActorTrainerConfig,
        device: torch.device,
    ) -> None:
        if config.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

        self.actor = actor
        self.dynamics = dynamics
        self.backbone = backbone
        self.sampler = sampler
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def train_step(self) -> dict[str, float]:
        self.actor.train()
        self.dynamics.eval()
        self.backbone.eval()

        batch = self.sampler.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            start_state = encode_images(self.backbone, batch.start_rgb, self.device)
            goal_state = encode_images(self.backbone, batch.goal_rgb, self.device)

        gaps = batch.frame_gaps.to(self.device).long()
        pred_state, action_abs_mean = self._rollout(start_state, gaps)
        loss = F.mse_loss(pred_state, goal_state)
        cosine = F.cosine_similarity(pred_state, goal_state, dim=-1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "cosine": float(cosine.detach().cpu()),
            "action_abs_mean": action_abs_mean,
            "frame_gap_mean": float(batch.frame_gaps.mean().detach().cpu()),
            "grad_norm": grad_norm,
        }

    def _rollout(
        self,
        start_state: torch.Tensor,
        gaps: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        dynamics_requires_grad = [p.requires_grad for p in self.dynamics.parameters()]
        for p in self.dynamics.parameters():
            p.requires_grad_(False)
        try:
            max_gap = int(gaps.max().item())
            state = start_state
            states = [state]
            action_abs_means = []
            for _ in range(max_gap):
                action = self.actor(state)
                action_abs_means.append(action.detach().abs().mean())
                state = self.dynamics(state, action)
                states.append(state)
            trajectory = torch.stack(states, dim=1)  # (B, max_gap+1, D)
            gather_idx = gaps.view(-1, 1, 1).expand(-1, 1, trajectory.shape[-1])
            pred_state = trajectory.gather(1, gather_idx).squeeze(1)
            action_abs_mean = float(torch.stack(action_abs_means).mean().cpu())
            return pred_state, action_abs_mean
        finally:
            for p, requires_grad in zip(self.dynamics.parameters(), dynamics_requires_grad, strict=True):
                p.requires_grad_(requires_grad)

    def _clip_grad_norm(self) -> float:
        if self.config.grad_clip_norm is None:
            return self._grad_norm()
        norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.config.grad_clip_norm,
        )
        return float(norm.detach().cpu())

    def _grad_norm(self) -> float:
        norms = [
            p.grad.detach().norm(2)
            for p in self.actor.parameters()
            if p.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.stack(norms).norm(2).cpu())
