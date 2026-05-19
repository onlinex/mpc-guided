"""Load a train.py checkpoint and play it in the SAPIEN GUI.

Mirrors play_bc_baseline.py but imports the active Actor from src.actor and
honors --normalize-states from the checkpoint's saved args (recomputing stats
from the dataset if needed).

Usage:
  uv run python play.py \\
    --checkpoint runs/bc/<RUN>/checkpoints/best_eval_success_at_end.pt \\
    --episodes 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from src.actor import Actor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--sim-backend", default="physx_cpu")
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gui", action="store_true",
                   help="Run without the SAPIEN viewer (just print success/return).")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="If set, also save per-episode mp4s via RecordEpisode.")
    return p.parse_args()


def _load_normalization(ckpt_args: dict, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover obs_mean/obs_std the trainer used. Returns (0, 1) if disabled."""
    if not ckpt_args.get("normalize_states", False):
        return np.zeros(state_dim, dtype=np.float32), np.ones(state_dim, dtype=np.float32)
    from src.bc import StateBCDataset
    ds = StateBCDataset(
        ckpt_args["dataset_dir"],
        device=torch.device("cpu"),
        num_demos=ckpt_args.get("num_demos"),
        normalize_states=True,
    )
    return ds.stats.mean.astype(np.float32), ds.stats.std.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_mode = "rgb_array" if args.no_gui else "human"
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        reward_mode="sparse",
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        render_mode=render_mode,
    )
    if args.video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode
        args.video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(args.video_dir), save_trajectory=False)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    actor = Actor(state_dim, action_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_mean, obs_std = _load_normalization(ckpt.get("args", {}), state_dim)
    print(
        f"Loaded {args.checkpoint} "
        f"(state_dim={state_dim} action_dim={action_dim} "
        f"normalize={(obs_std != 1).any()})"
    )

    try:
        successes, returns = [], []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            ep_return, succeeded, steps = 0.0, False, 0
            while True:
                steps += 1
                if not args.no_gui:
                    env.render()
                state = np.asarray(obs, dtype=np.float32).reshape(-1)
                state = (state - obs_mean) / obs_std
                state_t = torch.from_numpy(state).to(device).unsqueeze(0)
                with torch.no_grad():
                    action = actor(state_t).squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                succeeded = succeeded or bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                if not args.no_gui:
                    time.sleep(0.02)
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break
            successes.append(int(succeeded))
            returns.append(ep_return)
            print(f"ep {ep}: success={succeeded} return={ep_return:.2f} steps={steps}")
        print(
            f"\nSummary: success={np.mean(successes):.2f} "
            f"return={np.mean(returns):.2f} over {args.episodes} eps"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
