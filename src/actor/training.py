"""Actor training against expert video frame pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.actor.model import Actor
from src.backbone import encode_images
from src.datasets.video_pairs import VideoFramePairSampler
from src.dynamics.buffer import DynamicsBatch
from src.dynamics.model import ForwardDynamicsModel


@dataclass(frozen=True)
class VideoActorTrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 10.0
    batch_size: int = 32
    bc_input_noise_std: float = 0.0  # Gaussian noise on (visual, proprio) during BC only.


class VideoActorTrainer:
    """Optimizes an actor to reach goal visual latents through learned dynamics.

    Rolls the actor forward through frozen dynamics for ``gap`` steps and
    matches only the visual (R3M) component of the predicted state against the
    R3M encoding of the goal frame. Proprio is propagated through the rollout
    but is not part of the loss — goals are visual frames, not full robot states.
    """

    def __init__(
        self,
        *,
        actor: Actor,
        dynamics: ForwardDynamicsModel,
        backbone: torch.nn.Module | None,
        sampler: VideoFramePairSampler,
        config: VideoActorTrainerConfig,
        device: torch.device,
    ) -> None:
        if config.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")
        if backbone is None and not sampler.has_state:
            raise ValueError(
                "VideoActorTrainer needs either a visual backbone or a sampler with "
                "per-frame privileged state; got neither"
            )

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
        if self.backbone is not None:
            self.backbone.eval()

        batch = self.sampler.sample(self.config.batch_size, self.device)
        if self.backbone is None:
            start_visual = batch.start_state.to(self.device)
            goal_visual = batch.goal_state.to(self.device)
        else:
            with torch.no_grad():
                start_visual = encode_images(self.backbone, batch.start_rgb, self.device)
                goal_visual = encode_images(self.backbone, batch.goal_rgb, self.device)
        start_proprio = batch.start_proprio.to(self.device)

        gaps = batch.frame_gaps.to(self.device).long()
        pred_visual, action_abs_mean = self._rollout(start_visual, start_proprio, gaps)
        loss = F.mse_loss(pred_visual, goal_visual)
        cosine = F.cosine_similarity(pred_visual, goal_visual, dim=-1).mean()

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

    def bc_train_step(self, batch: DynamicsBatch) -> dict[str, float]:
        """Direct behavioral-cloning: MSE(actor(state, proprio), expert_action).

        Bypasses the dynamics rollout entirely. Optional Gaussian noise on the
        input (state, proprio) creates synthetic OOD samples each step — a
        cheap DART-style augmentation that helps the actor recover from the
        small state deviations its own closed-loop control introduces.
        """
        self.actor.train()
        visual = batch.visual
        proprio = batch.proprio
        noise_std = self.config.bc_input_noise_std
        if noise_std > 0.0:
            visual = visual + torch.randn_like(visual) * noise_std
            proprio = proprio + torch.randn_like(proprio) * noise_std
        pred_action = self.actor(visual, proprio)
        loss = F.mse_loss(pred_action, batch.action)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "cosine": float("nan"),
            "action_abs_mean": float(pred_action.detach().abs().mean().cpu()),
            "action_label_abs_mean": float(batch.action.detach().abs().mean().cpu()),
            "frame_gap_mean": float("nan"),
            "grad_norm": grad_norm,
        }

    def _rollout(
        self,
        start_visual: torch.Tensor,
        start_proprio: torch.Tensor,
        gaps: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        dynamics_requires_grad = [p.requires_grad for p in self.dynamics.parameters()]
        for p in self.dynamics.parameters():
            p.requires_grad_(False)
        try:
            max_gap = int(gaps.max().item())
            visual, proprio = start_visual, start_proprio
            visuals = [visual]
            action_abs_means = []
            for _ in range(max_gap):
                action = self.actor(visual, proprio)
                action_abs_means.append(action.detach().abs().mean())
                visual, proprio = self.dynamics(visual, proprio, action)
                visuals.append(visual)
            trajectory = torch.stack(visuals, dim=1)  # (B, max_gap+1, visual_dim)
            gather_idx = gaps.view(-1, 1, 1).expand(-1, 1, trajectory.shape[-1])
            pred_visual = trajectory.gather(1, gather_idx).squeeze(1)
            action_abs_mean = float(torch.stack(action_abs_means).mean().cpu())
            return pred_visual, action_abs_mean
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
