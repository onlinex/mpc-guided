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

from src.actor import Actor, ForwardModel, HeadLosses
from src.bc import StateBCDataset
from src.buffer import OnlineBuffer


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
    actor_loss_weight: float = 0.0
    total_loss_weight: float = 1.0
    actor_horizon: int = 1
    action_chunk: int = 1
    actor_surprise_coef: float = 0.0
    online_buffer_size: int = 300_000
    online_mix_ratio: float = 0.95
    explore_sigma: float = 0.0
    seed: int = 42
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 100
    log_freq: int = 1000
    rollout_freq: int = 1000
    num_rollout_episodes: int = 100
    num_eval_episodes: int = 50
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
    p.add_argument(
        "--actor-loss-weight",
        type=float,
        default=d.actor_loss_weight,
        help="Weight on the direct BC actor loss MSE(actor(obs), expert_action). "
        "Default 0.0 (pure model-based imitation). Set > 0 to mix in direct BC.",
    )
    p.add_argument(
        "--total-loss-weight",
        type=float,
        default=d.total_loss_weight,
        help="Weight on the model-based actor loss "
        "MSE(forward(obs, actor(obs)), next_obs). Default 1.0 (pure model-based). "
        "Set to 0 with --actor-loss-weight 1 for pure BC.",
    )
    p.add_argument(
        "--actor-surprise-coef",
        type=float,
        default=d.actor_surprise_coef,
        help="Subtract coef * std(normalize_actual(rollout_surprise)) from the "
        "actor's joint loss so the actor is rewarded for producing a DIVERSE "
        "(not just high) surprise distribution across the batch — avoids mode "
        "collapse where the actor picks one high-surprise trick everywhere. "
        "0 (default) = off. Only active when --total-loss-weight > 0 (needs the "
        "model-based gradient path). Start small (0.01-0.05) — competes with "
        "the imitation objective.",
    )
    p.add_argument(
        "--actor-horizon",
        type=int,
        default=d.actor_horizon,
        help="Number of forward-model rollouts the actor performs. Each rollout "
        "advances by --action-chunk env steps, so total env-step lookahead is "
        "H * N. H=1 (default) is single-rollout; H>1 stacks H queries to the "
        "actor and H invocations of the forward model.",
    )
    p.add_argument(
        "--action-chunk",
        type=int,
        default=d.action_chunk,
        help="Actions per actor query (N). The actor emits N actions per call; "
        "the forward model predicts the state N env steps after the chunk is "
        "executed open-loop; the env executes the chunk one action at a time "
        "before re-querying. N=1 (default) collapses to per-step prediction.",
    )
    p.add_argument(
        "--online-buffer-size",
        type=int,
        default=d.online_buffer_size,
        help="Capacity (transitions) of the FIFO buffer fed by eval rollouts.",
    )
    p.add_argument(
        "--online-mix-ratio",
        type=float,
        default=d.online_mix_ratio,
        help="Fraction of the dynamics-step batch sampled from the online buffer. "
        "0 = pure BC dynamics training (default). Clamped by buffer size early on.",
    )
    p.add_argument(
        "--explore-sigma",
        type=float,
        default=d.explore_sigma,
        help="Std-dev of Gaussian noise injected into the actor's bottleneck "
        "LATENT during rollouts only (0 = deterministic). Perturbs the plan, "
        "not individual actions — the decoder still expands it into a "
        "coherent chunk. Eval is always deterministic. Decoded chunk is "
        "clipped to [-1, 1] before execution and stored verbatim in the buffer.",
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--sim-backend", default=d.sim_backend)
    p.add_argument("--max-episode-steps", type=int, default=d.max_episode_steps)
    p.add_argument("--log-freq", type=int, default=d.log_freq)
    p.add_argument(
        "--rollout-freq",
        type=int,
        default=d.rollout_freq,
        help="Iterations between rollout/eval passes (both share cadence).",
    )
    p.add_argument(
        "--num-rollout-episodes",
        type=int,
        default=d.num_rollout_episodes,
        help="Episodes per rollout pass (feeds online buffer, no metrics).",
    )
    p.add_argument(
        "--num-eval-episodes",
        type=int,
        default=d.num_eval_episodes,
        help="Episodes per eval pass (deterministic metrics, best-ckpt signal).",
    )
    p.add_argument("--save-freq", type=int, default=d.save_freq)
    p.add_argument("--log-dir", default=d.log_dir)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--no-cuda", dest="cuda", action="store_false")
    p.set_defaults(cuda=d.cuda)
    parsed = p.parse_args()
    if parsed.actor_loss_weight < 0:
        p.error("--actor-loss-weight must be >= 0")
    if parsed.total_loss_weight < 0:
        p.error("--total-loss-weight must be >= 0")
    if parsed.actor_loss_weight == 0 and parsed.total_loss_weight == 0:
        p.error("at least one of --actor-loss-weight or --total-loss-weight must be > 0")
    if parsed.online_buffer_size < 1:
        p.error("--online-buffer-size must be >= 1")
    if not 0.0 <= parsed.online_mix_ratio <= 1.0:
        p.error("--online-mix-ratio must be in [0, 1]")
    if parsed.explore_sigma < 0:
        p.error("--explore-sigma must be >= 0")
    if parsed.actor_horizon < 1:
        p.error("--actor-horizon must be >= 1")
    if parsed.action_chunk < 1:
        p.error("--action-chunk must be >= 1")
    if parsed.actor_surprise_coef < 0:
        p.error("--actor-surprise-coef must be >= 0")
    if parsed.actor_surprise_coef > 0 and parsed.total_loss_weight == 0:
        p.error(
            "--actor-surprise-coef requires --total-loss-weight > 0 "
            "(surprise gradient flows through the model-based rollout)"
        )
    return Args(**vars(parsed))


def _act_chunk(
    actor: Actor,
    obs: np.ndarray,
    *,
    device: torch.device,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    latent_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy action *chunk* from normalized state. Shape ``(chunk_size, action_dim)``.

    With ``latent_sigma > 0``, Gaussian noise is injected into the actor's
    bottleneck latent (post-encoder, pre-decoder), so exploration perturbs
    the *plan* rather than each action independently — the decoder still
    expands it into a coherent chunk.
    """
    state = (np.asarray(obs, dtype=np.float32).reshape(-1) - obs_mean) / obs_std
    state_t = torch.from_numpy(state).to(device).unsqueeze(0)
    latent = actor.encode(state_t)
    if latent_sigma > 0:
        assert rng is not None, "rng required when latent_sigma > 0"
        noise = rng.normal(0.0, latent_sigma, size=latent.shape).astype(np.float32)
        latent = latent + torch.from_numpy(noise).to(device)
    chunk = actor.decode(latent)
    return chunk.squeeze(0).cpu().numpy().astype(np.float32)


def _unwrap(x) -> bool:
    """ManiSkill returns single-element batched tensors for done/truncated/success;
    pull out the scalar and cast to bool."""
    return bool(np.asarray(x).reshape(-1)[0])


def collect_rollouts(
    *,
    env: gym.Env,
    actor: Actor,
    device: torch.device,
    num_episodes: int,
    seed_base: int,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    buffer: OnlineBuffer,
    sigma: float,
    rng: np.random.Generator,
) -> None:
    """Greedy + Gaussian-noise rollouts purely to populate the buffer.

    No metrics. Action stored in the buffer is the one actually executed
    (post-noise, post-clip) so the dynamics model sees ground-truth (s, a, s').
    """
    actor.eval()
    N = actor.chunk_size
    with torch.no_grad():
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + i)
            done = False
            while not done:
                # Snapshot the chunk's start state, execute N actions
                # open-loop, snapshot the end state, push as one entry. If
                # the episode terminates mid-chunk, drop the partial — the
                # dynamics model only trains on completed N-step transitions.
                chunk_start = np.asarray(obs, dtype=np.float32).reshape(-1)
                chunk = _act_chunk(
                    actor, obs,
                    device=device, obs_mean=obs_mean, obs_std=obs_std,
                    latent_sigma=sigma, rng=rng,
                )
                if sigma > 0:
                    # Latent noise can push decoded actions outside [-1, 1].
                    chunk = np.clip(chunk, -1.0, 1.0)
                executed_chunk = chunk.copy()
                complete = True
                for t in range(N):
                    obs, _, terminated, truncated, _ = env.step(chunk[t])
                    if _unwrap(terminated) or _unwrap(truncated):
                        done = True
                        if t + 1 < N:
                            complete = False
                        break
                if complete:
                    chunk_end = np.asarray(obs, dtype=np.float32).reshape(-1)
                    buffer.add(chunk_start, executed_chunk, chunk_end)
    actor.train()


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
    N = actor.chunk_size
    with torch.no_grad():
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + i)
            ep_return, once, end, steps = 0.0, False, False, 0
            chunk, chunk_idx = None, N
            while True:
                steps += 1
                if chunk_idx >= N:
                    chunk = _act_chunk(actor, obs, device=device, obs_mean=obs_mean, obs_std=obs_std)
                    chunk_idx = 0
                action = chunk[chunk_idx]
                chunk_idx += 1
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                sflag = _unwrap(info.get("success", False))
                once = once or sflag
                end = sflag
                if _unwrap(terminated) or _unwrap(truncated):
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
        reward_mode="dense",
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
        horizon=args.actor_horizon,
        chunk_size=args.action_chunk,
    )
    print(
        f"Dataset: {len(ds)} samples from {ds.num_episodes} episodes "
        f"(state_dim={ds.state_dim} action_dim={ds.action_dim} "
        f"horizon={ds.horizon} chunk_size={ds.chunk_size})"
    )

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)

    actor = Actor(ds.state_dim, ds.action_dim, chunk_size=args.action_chunk).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    forward_model = ForwardModel(
        ds.state_dim, ds.action_dim, chunk_size=args.action_chunk
    ).to(device)
    forward_optimizer = optim.Adam(forward_model.parameters(), lr=args.lr)

    online_buffer = OnlineBuffer(
        capacity=args.online_buffer_size,
        state_dim=ds.state_dim,
        action_dim=ds.action_dim,
        chunk_size=args.action_chunk,
    )
    buffer_rng = np.random.default_rng(args.seed)
    explore_rng = np.random.default_rng(args.seed + 1)
    obs_mean_t = torch.from_numpy(ds.stats.mean.astype(np.float32)).to(device)
    obs_std_t = torch.from_numpy(ds.stats.std.astype(np.float32)).to(device)
    print(f"Device: {device} | run_dir: {run_dir}")
    print(f"Actor params: {sum(p.numel() for p in actor.parameters())}")
    print(f"ForwardModel params: {sum(p.numel() for p in forward_model.parameters())}")

    best_eval = defaultdict(float)
    try:
        w_a = args.actor_loss_weight
        w_t = args.total_loss_weight
        H = args.actor_horizon
        for iteration, batch in enumerate(
            tqdm(dataloader, total=args.total_iters, desc="bc/train")
        ):
            obs, action, next_obs, goal_obs = batch

            # 1. Forward-model update on a BC + online mix.
            k = min(int(args.batch_size * args.online_mix_ratio), len(online_buffer))
            n_bc = args.batch_size - k
            forward_optimizer.zero_grad()
            bc_losses = on_losses = None
            if n_bc > 0:
                bc_losses = forward_model.head_losses(
                    obs[:n_bc], action[:n_bc], next_obs[:n_bc]
                )
            if k > 0:
                on_s, on_a, on_ns = online_buffer.sample(k, buffer_rng)
                on_s_t = (torch.from_numpy(on_s).to(device) - obs_mean_t) / obs_std_t
                on_ns_t = (torch.from_numpy(on_ns).to(device) - obs_mean_t) / obs_std_t
                on_a_t = torch.from_numpy(on_a).to(device)
                on_losses = forward_model.head_losses(on_s_t, on_a_t, on_ns_t)
            losses = HeadLosses.combine(n_bc, bc_losses, k, on_losses)
            # RMSE not MSE on the surprise head: sqrt amplifies grads when
            # targets are tiny, which they are (per-sample dynamics errors).
            surprise_loss = losses.surprise.clamp_min(1e-12).sqrt()
            (losses.state + surprise_loss).backward()
            forward_optimizer.step()
            forward_model.update_surprise_stats(losses.per_sample_error)

            # 2. Actor update. total_loss rolls (actor + forward) H steps and
            #    compares to state_{t+H}; forward params are frozen for the
            #    rollout to avoid building grad buffers we wouldn't use anyway.
            optimizer.zero_grad()
            # Actor produces a length-N chunk per query; BC supervises the
            # full chunk against the expert's chunk.
            actor_loss = F.mse_loss(actor(obs), action)
            forward_model.requires_grad_(False)
            # detach_surprise=False keeps the surprise -> trunk -> action path
            # alive so the actor pull term's gradient can reach actor params.
            collect_grad_surprise = w_t > 0 and args.actor_surprise_coef > 0
            rollout_surprise: list[torch.Tensor] = []          # detached, for logging
            rollout_surprise_grad: list[torch.Tensor] = []     # grad-tracking, for loss
            actor_pull_term = None
            with torch.set_grad_enabled(w_t > 0):
                # H rollouts of (actor -> chunk -> forward); forward consumes
                # the full chunk and predicts the state N env steps ahead.
                s = obs
                for _ in range(H):
                    s, ps = forward_model(
                        s, actor(s), detach_surprise=not collect_grad_surprise
                    )
                    rollout_surprise.append(ps.detach())
                    if collect_grad_surprise:
                        rollout_surprise_grad.append(ps)
                total_loss = F.mse_loss(s, goal_obs)
            joint_loss = w_a * actor_loss + w_t * total_loss
            if rollout_surprise_grad:
                # Std (not mean) of normalized surprise: rewards diverse
                # rollouts, not just high-surprise ones — avoids the mode
                # collapse where the actor finds one high-surprise trick.
                stacked = forward_model.normalize_actual(
                    torch.stack(rollout_surprise_grad)
                )  # (H, B)
                actor_pull_term = args.actor_surprise_coef * stacked.std(dim=1).mean()
                joint_loss = joint_loss - actor_pull_term
            joint_loss.backward()
            optimizer.step()
            forward_model.requires_grad_(True)

            if iteration % args.log_freq == 0:
                writer.add_scalar("losses/actor_loss", float(actor_loss.detach().cpu()), iteration)
                writer.add_scalar("losses/dynamics_loss", float(losses.state.detach().cpu()), iteration)
                if bc_losses is not None:
                    writer.add_scalar("losses/dynamics_loss_bc", float(bc_losses.state.detach().cpu()), iteration)
                if on_losses is not None:
                    writer.add_scalar("losses/dynamics_loss_online", float(on_losses.state.detach().cpu()), iteration)
                writer.add_scalar("losses/total_loss", float(total_loss.detach().cpu()), iteration)
                writer.add_scalar("online/buffer_size", len(online_buffer), iteration)
                writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar("surprise/head_loss", float(surprise_loss.detach().cpu()), iteration)
                if actor_pull_term is not None:
                    writer.add_scalar(
                        "surprise/actor_pull",
                        float(actor_pull_term.detach().cpu()),
                        iteration,
                    )
                # (H, B) raw -> (H, B) normalized in [0, 1].
                norm = forward_model.normalize_actual(
                    torch.stack(rollout_surprise)
                ).cpu()
                for step_idx in range(norm.shape[0]):
                    step_vals = norm[step_idx]
                    writer.add_scalar(f"surprise/rollout_step/{step_idx + 1}", float(step_vals.mean()), iteration)
                    writer.add_histogram(f"surprise/rollout_dist/{step_idx + 1}", step_vals, iteration)
                writer.add_scalar("surprise/rollout_mean", float(norm.mean()), iteration)

            if iteration % args.rollout_freq == 0:
                collect_rollouts(
                    env=env,
                    actor=actor,
                    device=device,
                    num_episodes=args.num_rollout_episodes,
                    seed_base=args.seed * 1000 + iteration,
                    obs_mean=ds.stats.mean,
                    obs_std=ds.stats.std,
                    buffer=online_buffer,
                    sigma=args.explore_sigma,
                    rng=explore_rng,
                )
                metrics = evaluate(
                    env=env,
                    actor=actor,
                    device=device,
                    num_episodes=args.num_eval_episodes,
                    # Offset seeds so eval doesn't reuse rollout episodes.
                    seed_base=args.seed * 1000 + iteration + 500_000,
                    obs_mean=ds.stats.mean,
                    obs_std=ds.stats.std,
                )
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}", v, iteration)
                print(
                    f"iter={iteration} actor_loss={float(actor_loss):.4f} "
                    f"success_once={metrics['success_once']:.3f} "
                    f"success_at_end={metrics['success_at_end']:.3f} "
                    f"return={metrics['episode_return']:.2f}"
                )
                ckpt_payload = {
                    "actor": actor.state_dict(),
                    "forward_model": forward_model.state_dict(),
                    "iter": iteration,
                    "metrics": metrics,
                    "args": asdict(args),
                }
                torch.save(ckpt_payload, ckpt_dir / "latest.pt")
                for k in ("success_once", "success_at_end"):
                    if metrics[k] > best_eval[k]:
                        best_eval[k] = metrics[k]
                        torch.save(ckpt_payload, ckpt_dir / f"best_eval_{k}.pt")

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
