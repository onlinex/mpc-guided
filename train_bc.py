"""Behavior-cloning training on our per-episode dataset format.

Same numerics as the ManiSkill state-based BC baseline — 256x256 ReLU MLP
(``src.actor.Actor``), single-action MSE, Adam lr 3e-4, batch 1024 — but
reads our per-episode npy files (state.npy + actions.npy) produced by
build_dataset.py instead of a monolithic h5.

For a strict standalone upstream reproduction, see train_bc_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.actor import Actor, ForwardModel
from src.bc import StateBCDataset


class IterationBasedBatchSampler(BatchSampler):
    """Resamples from an underlying BatchSampler until num_iterations is reached.

    Vendored from NVIDIA's DeepLearningExamples (same source as upstream
    ManiSkill bc.py). Lets DataLoader run for a fixed iteration count rather
    than a fixed number of epochs.
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


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    dataset_dir: str = "data/pickcube_rl"
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
    log_dir: str = "runs/bc"
    run_name: str | None = None
    cuda: bool = True


def parse_args() -> Args:
    d = Args()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default=d.env_id)
    p.add_argument("--dataset-dir", default=d.dataset_dir)
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


def evaluate(
    *,
    env: gym.Env,
    actor: Actor,
    device: torch.device,
    num_episodes: int,
    seed_base: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> dict[str, float]:
    success_once, success_at_end, returns, lengths = [], [], [], []
    actor.eval()
    with torch.no_grad():
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + i)
            ep_return, once, end, steps = 0.0, False, False, 0
            while True:
                steps += 1
                state = np.asarray(obs, dtype=np.float32).reshape(-1)
                state = (state - obs_mean) / obs_std
                action = actor(torch.from_numpy(state).to(device).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                sflag = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                once = once or sflag
                end = sflag
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                    np.asarray(truncated).reshape(-1)[0]
                ):
                    break
            success_once.append(float(once))
            success_at_end.append(float(end))
            returns.append(ep_return)
            lengths.append(steps)
    actor.train()
    return {
        "success_once": float(np.mean(success_once)),
        "success_at_end": float(np.mean(success_at_end)),
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

    # Validate dataset's control mode against ours.
    dataset_meta_path = Path(args.dataset_dir) / "metadata.json"
    if not dataset_meta_path.exists():
        raise SystemExit(f"dataset metadata not found at {dataset_meta_path}")
    dataset_meta = json.loads(dataset_meta_path.read_text())
    ds_cm = dataset_meta["config"]["control_mode"]
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

    ds = StateBCDataset(
        args.dataset_dir,
        device=device,
        num_demos=args.num_demos,
        normalize_states=args.normalize_states,
    )
    print(
        f"Dataset: {len(ds)} transitions from {ds.num_episodes} episodes "
        f"(state_dim={ds.state_dim} action_dim={ds.action_dim})"
    )

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)

    actor = Actor(ds.state_dim, ds.action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    forward_model = ForwardModel(ds.state_dim, ds.action_dim).to(device)
    forward_optimizer = optim.Adam(forward_model.parameters(), lr=args.lr)
    print(f"Device: {device} | run_dir: {run_dir}")
    print(f"Actor params: {sum(p.numel() for p in actor.parameters())}")
    print(f"ForwardModel params: {sum(p.numel() for p in forward_model.parameters())}")

    best_eval = defaultdict(float)
    try:
        for iteration, batch in enumerate(
            tqdm(dataloader, total=args.total_iters, desc="bc/train")
        ):
            obs, action, next_obs = batch

            # Actor update.
            pred = actor(obs)
            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Forward-model update (independent optimizer; gradients can't leak
            # into the actor since obs/action are leaf tensors here).
            pred_next = forward_model(obs, action)
            fwd_loss = F.mse_loss(pred_next, next_obs)
            forward_optimizer.zero_grad()
            fwd_loss.backward()
            forward_optimizer.step()

            if iteration % args.log_freq == 0:
                writer.add_scalar("losses/actor_loss", float(loss.detach().cpu()), iteration)
                writer.add_scalar("losses/dynamics_loss", float(fwd_loss.detach().cpu()), iteration)
                writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], iteration)

            if iteration % args.eval_freq == 0:
                metrics = evaluate(
                    env=env,
                    actor=actor,
                    device=device,
                    num_episodes=args.num_eval_episodes,
                    seed_base=args.seed * 1000 + iteration,
                    obs_mean=ds.stats.mean,
                    obs_std=ds.stats.std,
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
                            {
                                "actor": actor.state_dict(),
                                "forward_model": forward_model.state_dict(),
                                "iter": iteration,
                                "metrics": metrics,
                                "args": asdict(args),
                            },
                            ckpt_dir / f"best_eval_{k}.pt",
                        )

            if args.save_freq is not None and iteration % args.save_freq == 0:
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "forward_model": forward_model.state_dict(),
                        "iter": iteration,
                    },
                    ckpt_dir / f"step_{iteration}.pt",
                )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
