"""Run simple ManiSkill rollouts without any learned algorithm code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from src.legacy.actor import Actor, ActorConfig
from src.backbone import build_backbone
from src.observations import encode_observation, extract_proprio
from src.rollout import rollout
from src.utils import OUNoise, pick_device


ENV_ID = "PickCube-v1"


@dataclass(frozen=True)
class DemoConfig:
    env_id: str = ENV_ID
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str | None = "human"
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"
    camera_width: int = 224
    camera_height: int = 224
    policy: str = "random"
    seed: int = 42
    episodes: int = 5
    max_steps: int = 200
    action_chunk_size: int = 1
    checkpoint: str | None = None
    r3m_model_id: str = "resnet18"
    device: str = "auto"
    stochastic: bool = False
    exploration_ou_theta: float = 0.15
    exploration_ou_std: float = 0.5


class SpaceActor:
    """Minimal actor over a continuous Gymnasium Box action space."""

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        policy: str,
        action_chunk_size: int,
        seed: int,
    ):
        self.action_space = action_space
        self.policy = policy
        self.action_chunk_size = action_chunk_size
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, _obs) -> np.ndarray:
        if self.policy == "zero":
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            return np.repeat(action.reshape(1, -1), self.action_chunk_size, axis=0)
        return np.stack([self._sample_action() for _ in range(self.action_chunk_size)], axis=0)

    def _sample_action(self) -> np.ndarray:
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        if np.isfinite(low).all() and np.isfinite(high).all():
            return self.rng.uniform(low, high).astype(np.float32)
        return np.asarray(self.action_space.sample(), dtype=np.float32)


class CheckpointActor:
    """Wraps a trained ``Actor`` plus an R3M backbone for env rollouts.

    The actor is always queried deterministically. When ``ou_noise`` is provided
    (``--stochastic`` mode), AR(1) exploration noise is added to the action and
    clipped to the action bounds — matching how training collection drives the
    env.
    """

    def __init__(
        self,
        *,
        actor: Actor,
        backbone: torch.nn.Module,
        env: gym.Env,
        device: torch.device,
        camera_uid: str,
        action_chunk_size: int,
        ou_noise: OUNoise | None,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        self.actor = actor
        self.backbone = backbone
        self.env = env
        self.device = device
        self.camera_uid = camera_uid
        self.action_chunk_size = action_chunk_size
        self.ou_noise = ou_noise
        self.action_low = action_low
        self.action_high = action_high

    def reset(self) -> None:
        if self.ou_noise is not None:
            self.ou_noise.reset()

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        visual_np = encode_observation(self.backbone, obs, self.device, self.camera_uid)
        proprio_np = extract_proprio(self.env)
        visual = torch.as_tensor(visual_np, device=self.device).reshape(1, -1)
        proprio = torch.as_tensor(proprio_np, device=self.device).reshape(1, -1)
        action = self.actor(visual, proprio).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.ou_noise is not None:
            action = np.clip(
                action + self.ou_noise.sample(), self.action_low, self.action_high
            ).astype(np.float32)
        return np.repeat(action.reshape(1, -1), self.action_chunk_size, axis=0)


def load_actor_from_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> Actor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor_config = ActorConfig(**ckpt["actor_config"])
    state_dict = ckpt["actor_state_dict"]
    placeholder_action_low = torch.full((actor_config.action_dim,), -1.0, dtype=torch.float32)
    placeholder_action_high = torch.full((actor_config.action_dim,), 1.0, dtype=torch.float32)
    actor = Actor(
        actor_config,
        action_low=placeholder_action_low,
        action_high=placeholder_action_high,
    ).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def parse_args() -> DemoConfig:
    defaults = DemoConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default=defaults.env_id)
    p.add_argument("--obs-mode", choices=["state", "state_dict", "rgb"], default=defaults.obs_mode)
    p.add_argument("--control-mode", default=defaults.control_mode)
    p.add_argument("--render-mode", choices=["none", "human", "rgb_array", "sensors"], default="human")
    p.add_argument("--sim-backend", default=defaults.sim_backend)
    p.add_argument("--render-backend", default=defaults.render_backend)
    p.add_argument("--camera-width", type=int, default=defaults.camera_width)
    p.add_argument("--camera-height", type=int, default=defaults.camera_height)
    p.add_argument("--policy", choices=["random", "zero"], default=defaults.policy)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--episodes", type=int, default=defaults.episodes)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--action-chunk-size", type=int, default=defaults.action_chunk_size)
    p.add_argument(
        "--checkpoint",
        default=defaults.checkpoint,
        help="Path to a trained actor checkpoint (.pt). When set, overrides --policy and forces obs_mode=rgb.",
    )
    p.add_argument("--r3m-model-id", default=defaults.r3m_model_id)
    p.add_argument("--device", default=defaults.device)
    p.add_argument(
        "--stochastic",
        action="store_true",
        default=defaults.stochastic,
        help="Add OU exploration noise on top of the actor's mean action. Default is deterministic.",
    )
    p.add_argument(
        "--exploration-ou-theta",
        type=float,
        default=defaults.exploration_ou_theta,
        help="AR(1)/OU decorrelation rate for --stochastic playback.",
    )
    p.add_argument(
        "--exploration-ou-std",
        type=float,
        default=defaults.exploration_ou_std,
        help="Stationary std of the AR(1) exploration noise for --stochastic playback.",
    )
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be >= 1.")
    if args.max_steps < 1:
        p.error("--max-steps must be >= 1.")
    if args.action_chunk_size < 1:
        p.error("--action-chunk-size must be >= 1.")
    obs_mode = args.obs_mode
    if args.checkpoint is not None:
        obs_mode = "rgb"
    if obs_mode == "rgb" and args.render_backend == "none":
        p.error("--render-backend none is incompatible with --obs-mode rgb / --checkpoint.")
    return DemoConfig(
        **{
            **vars(args),
            "obs_mode": obs_mode,
            "render_mode": None if args.render_mode == "none" else args.render_mode,
        }
    )


def make_env(cfg: DemoConfig) -> gym.Env:
    kwargs = {
        "obs_mode": cfg.obs_mode,
        "control_mode": cfg.control_mode,
        "sim_backend": cfg.sim_backend,
        "render_backend": cfg.render_backend,
    }
    if cfg.render_mode is not None:
        kwargs["render_mode"] = cfg.render_mode
    if cfg.obs_mode == "rgb":
        kwargs["sensor_configs"] = {
            "base_camera": {"width": cfg.camera_width, "height": cfg.camera_height}
        }
    return gym.make(cfg.env_id, **kwargs)


def run(cfg: DemoConfig) -> None:
    env = make_env(cfg)
    try:
        action_space = getattr(env, "single_action_space", env.action_space)
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"Expected Box action space, got {type(action_space).__name__}.")
        if cfg.checkpoint is not None:
            device = pick_device(cfg.device)
            backbone = build_backbone(device, model_id=cfg.r3m_model_id)
            loaded_actor = load_actor_from_checkpoint(cfg.checkpoint, device=device)
            action_dim = int(np.prod(action_space.shape))
            ou_noise = (
                None
                if not cfg.stochastic or cfg.exploration_ou_std <= 0.0
                else OUNoise(
                    action_dim=action_dim,
                    theta=cfg.exploration_ou_theta,
                    stationary_std=cfg.exploration_ou_std,
                    seed=cfg.seed,
                )
            )
            actor = CheckpointActor(
                actor=loaded_actor,
                backbone=backbone,
                env=env,
                device=device,
                camera_uid="base_camera",
                action_chunk_size=cfg.action_chunk_size,
                ou_noise=ou_noise,
                action_low=np.asarray(action_space.low, dtype=np.float32).reshape(-1),
                action_high=np.asarray(action_space.high, dtype=np.float32).reshape(-1),
            )
            policy_label = f"checkpoint:{cfg.checkpoint}"
            if ou_noise is not None:
                policy_label += f" (OU theta={cfg.exploration_ou_theta} std={cfg.exploration_ou_std})"
        else:
            actor = SpaceActor(
                action_space,
                policy=cfg.policy,
                action_chunk_size=cfg.action_chunk_size,
                seed=cfg.seed,
            )
            policy_label = cfg.policy

        print(
            f"Env: {cfg.env_id} | obs_mode={cfg.obs_mode} | policy={policy_label} | "
            f"sim_backend={cfg.sim_backend} | render_backend={cfg.render_backend}",
            flush=True,
        )

        def log_episode(ep: int, steps: int, ret: float, success: bool) -> None:
            print(f"episode {ep}: steps={steps} return={ret:.2f} success={success}", flush=True)

        metrics = rollout(
            env,
            actor,
            cfg.episodes,
            max_steps=cfg.max_steps,
            action_chunk_size=cfg.action_chunk_size,
            seed=cfg.seed,
            render=cfg.render_mode is not None,
            log_episode=log_episode,
        )
        print(
            f"success_rate={metrics['success_rate']:.0%} mean_return={metrics['mean_return']:.2f}",
            flush=True,
        )
    finally:
        env.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
