"""Run simple ManiSkill rollouts without any learned algorithm code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from src.rollout import rollout


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
    seed: int = 0
    episodes: int = 5
    max_steps: int = 200
    action_chunk_size: int = 1


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
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be >= 1.")
    if args.max_steps < 1:
        p.error("--max-steps must be >= 1.")
    if args.action_chunk_size < 1:
        p.error("--action-chunk-size must be >= 1.")
    if args.obs_mode == "rgb" and args.render_backend == "none":
        p.error("--render-backend none is incompatible with --obs-mode rgb.")
    return DemoConfig(
        **{
            **vars(args),
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
        actor = SpaceActor(
            action_space,
            policy=cfg.policy,
            action_chunk_size=cfg.action_chunk_size,
            seed=cfg.seed,
        )

        print(
            f"Env: {cfg.env_id} | obs_mode={cfg.obs_mode} | policy={cfg.policy} | "
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
