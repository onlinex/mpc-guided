"""Faithful port of the ManiSkill state-based BC baseline.

Mirrors examples/baselines/bc/bc.py from the ManiSkill repo. The implementation
is intentionally minimal — plain MLP, raw output, single-action MSE — so that
results can be compared directly against the reported baseline numbers.

Run order:
  1. Download RL demos (one-time):
     uv run python -m mani_skill.utils.download_demo PickCube-v1

  2. Convert to state-based pd_joint_delta_pos h5:
     uv run python -m mani_skill.trajectory.replay_trajectory \\
       --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.h5 \\
       --use-first-env-state -c pd_joint_delta_pos -o state \\
       --save-traj --num-procs 10 -b cpu

  3. Train:
     uv run python train_bc_baseline.py \\
       --demo-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.cpu.h5 \\
       --total-iters 50000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    demo_path: str = "~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.cpu.h5"
    control_mode: str = "pd_joint_delta_pos"
    num_demos: int | None = None
    total_iters: int = 50_000
    batch_size: int = 1024
    lr: float = 3e-4
    normalize_states: bool = False
    seed: int = 42
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 100
    log_freq: int = 1000
    eval_freq: int = 1000
    num_eval_episodes: int = 100
    save_freq: int | None = None
    log_dir: str = "runs/bc-baseline"
    run_name: str | None = None
    cuda: bool = True


def parse_args() -> Args:
    d = Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default=d.env_id)
    p.add_argument("--demo-path", default=d.demo_path)
    p.add_argument("--control-mode", default=d.control_mode)
    p.add_argument("--num-demos", type=int, default=d.num_demos)
    p.add_argument("--total-iters", type=int, default=d.total_iters)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--normalize-states", action=argparse.BooleanOptionalAction, default=d.normalize_states)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--sim-backend", default=d.sim_backend)
    p.add_argument("--max-episode-steps", type=int, default=d.max_episode_steps)
    p.add_argument("--log-freq", type=int, default=d.log_freq)
    p.add_argument("--eval-freq", type=int, default=d.eval_freq)
    p.add_argument("--num-eval-episodes", type=int, default=d.num_eval_episodes)
    p.add_argument("--save-freq", type=int, default=d.save_freq)
    p.add_argument("--log-dir", default=d.log_dir)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--no-cuda", dest="cuda", action="store_false")
    p.set_defaults(cuda=d.cuda)
    return Args(**vars(p.parse_args()))


class IterationBasedBatchSampler(BatchSampler):
    """Resamples from an underlying BatchSampler until num_iterations is reached.

    Vendored from the NVIDIA DeepLearningExamples repo (same source as upstream).
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def _load_h5_group(g):
    out = {}
    for k in g.keys():
        if isinstance(g[k], h5py.Dataset):
            out[k] = g[k][:]
        else:
            out[k] = _load_h5_group(g[k])
    return out


class ManiSkillDataset(Dataset):
    """Loads the h5 produced by replay_trajectory ``-o state``.

    Drops the last (terminal) obs of each episode since there's no matching
    action. Optionally normalizes all observations to zero mean / unit std
    (single scalar mean/std across all dims, matching upstream behavior).
    """

    def __init__(
        self,
        dataset_file: str,
        device: torch.device,
        load_count: int | None = None,
        normalize_states: bool = False,
    ) -> None:
        dataset_file = os.path.expanduser(dataset_file)
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        with open(json_path, "r") as f:
            self.json_data = json.load(f)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]

        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes from {dataset_file}")

        obs_list, action_list, done_list = [], [], []
        for eps_id in tqdm(range(load_count), desc="bc/load-h5", unit="ep"):
            eps = self.episodes[eps_id]
            traj = _load_h5_group(self.data[f"traj_{eps['episode_id']}"])
            obs_list.append(traj["obs"][:-1])
            action_list.append(traj["actions"])
            done_list.append(traj["success"].reshape(-1, 1))

        self.observations = np.vstack(obs_list).astype(np.float32)
        self.actions = np.vstack(action_list).astype(np.float32)
        self.dones = np.vstack(done_list)
        assert self.observations.shape[0] == self.actions.shape[0]
        self.device = device

        self.state_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[1])

        if normalize_states:
            mean, std = self.get_state_stats()
            self.observations = (self.observations - mean) / std
            self._mean, self._std = mean, std
        else:
            self._mean, self._std = 0.0, 1.0

    def get_state_stats(self):
        return float(self.observations.mean()), float(self.observations.std())

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float().to(self.device)
        action = torch.from_numpy(self.actions[idx]).float().to(self.device)
        done = torch.from_numpy(self.dones[idx]).to(self.device)
        return obs, action, done


class Actor(nn.Module):
    """state_dim -> 256 -> ReLU -> 256 -> ReLU -> action_dim. No squash, no LN."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def evaluate(
    *,
    env: gym.Env,
    actor: Actor,
    device: torch.device,
    num_episodes: int,
    seed_base: int,
    obs_mean: float,
    obs_std: float,
) -> dict[str, float]:
    """Sequentially run ``num_episodes`` episodes and aggregate success metrics."""
    success_once_list, success_at_end_list, returns, lengths = [], [], [], []
    actor.eval()
    with torch.no_grad():
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + i)
            ep_return, succeeded_once, succeeded_end = 0.0, False, False
            steps = 0
            while True:
                steps += 1
                state = np.asarray(obs, dtype=np.float32).reshape(-1)
                if obs_std != 1.0 or obs_mean != 0.0:
                    state = (state - obs_mean) / obs_std
                action = actor(torch.from_numpy(state).to(device).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                success_flag = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                succeeded_once = succeeded_once or success_flag
                succeeded_end = success_flag
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                    np.asarray(truncated).reshape(-1)[0]
                ):
                    break
            success_once_list.append(float(succeeded_once))
            success_at_end_list.append(float(succeeded_end))
            returns.append(ep_return)
            lengths.append(steps)
    actor.train()
    return {
        "success_once": float(np.mean(success_once_list)),
        "success_at_end": float(np.mean(success_at_end_list)),
        "episode_return": float(np.mean(returns)),
        "episode_length": float(np.mean(lengths)),
    }


def main() -> None:
    args = parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.env_id}-{timestamp}-seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Validate control_mode matches the dataset's.
    demo_path = os.path.expanduser(args.demo_path)
    json_path = demo_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        demo_info = json.load(f)
    ds_kwargs = demo_info["env_info"]["env_kwargs"]
    if "control_mode" in ds_kwargs:
        ds_cm = ds_kwargs["control_mode"]
    else:
        ds_cm = demo_info["episodes"][0]["control_mode"]
    if ds_cm != args.control_mode:
        raise SystemExit(
            f"Control mode mismatch: dataset={ds_cm} args={args.control_mode}"
        )

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        reward_mode="sparse",
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
    )

    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
        normalize_states=args.normalize_states,
    )
    print(
        f"Dataset: {len(ds)} transitions, state_dim={ds.state_dim} action_dim={ds.action_dim}"
    )

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)

    state_dim = ds.state_dim
    action_dim = ds.action_dim
    actor = Actor(state_dim, action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)

    print(f"Device: {device} | run_dir: {run_dir}")
    print(f"Actor: state_dim={state_dim} action_dim={action_dim} | params={sum(p.numel() for p in actor.parameters())}")

    best_eval = defaultdict(float)
    try:
        for iteration, batch in enumerate(tqdm(dataloader, total=args.total_iters, desc="bc/train")):
            obs, action, _ = batch
            pred = actor(obs)
            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.log_freq == 0:
                writer.add_scalar("losses/total_loss", float(loss.detach().cpu()), iteration)
                writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], iteration)

            if iteration % args.eval_freq == 0:
                metrics = evaluate(
                    env=env,
                    actor=actor,
                    device=device,
                    num_episodes=args.num_eval_episodes,
                    seed_base=args.seed * 1000 + iteration,
                    obs_mean=ds._mean,  # noqa: SLF001
                    obs_std=ds._std,    # noqa: SLF001
                )
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}", v, iteration)
                print(
                    f"iter={iteration} loss={float(loss):.4f} "
                    f"success_once={metrics['success_once']:.3f} "
                    f"success_at_end={metrics['success_at_end']:.3f} "
                    f"return={metrics['episode_return']:.2f}"
                )
                for k in ("success_once", "success_at_end"):
                    if metrics[k] > best_eval[k]:
                        best_eval[k] = metrics[k]
                        torch.save(
                            {"actor": actor.state_dict(), "iter": iteration, "metrics": metrics},
                            ckpt_dir / f"best_eval_{k}.pt",
                        )

            if args.save_freq is not None and iteration % args.save_freq == 0:
                torch.save(
                    {"actor": actor.state_dict(), "iter": iteration},
                    ckpt_dir / f"step_{iteration}.pt",
                )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
