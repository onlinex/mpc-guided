"""Train a visual forward dynamics model from actor-sampled ManiSkill interaction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.actor import (
    Actor,
    ActorConfig,
    VideoActorTrainer,
    VideoActorTrainerConfig,
)
from src.backbone import (
    BACKBONE_PRECISIONS,
    R3M_FEAT_DIM,
    R3M_MODEL_IDS,
    build_backbone,
)
from src.datasets.video_pairs import VideoFramePairSampler
from src.observations import encode_observation
from src.dynamics import (
    DynamicsTrainer,
    DynamicsTrainerConfig,
    ForwardDynamicsConfig,
    ForwardDynamicsModel,
    TransitionReplayBuffer,
)
from src.utils import OUNoise, pick_device, to_scalar_bool


ENV_ID = "PickCube-v1"


@dataclass(frozen=True)
class TrainDynamicsConfig:
    env_id: str = ENV_ID
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"
    camera_uid: str | None = None
    camera_width: int = 224
    camera_height: int = 224
    r3m_model_id: str = "resnet18"
    backbone_precision: str = "fp32"
    seed: int = 0
    initial_episodes: int = 16
    collection_rounds: int = 40
    episodes_per_round: int = 32
    max_steps: int = 100
    buffer_capacity: int = 50_000
    train_steps_per_round: int = 50
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 3e-3
    grad_clip_norm: float | None = 10.0
    hidden_dims: tuple[int, ...] = (512, 512)
    log_dir: str = "runs/dynamics"
    run_name: str | None = None
    log_interval: int = 50
    actor_video_training: bool = True
    actor_video_dataset_dir: str = "data/pickcube_expert_videos"
    actor_train_steps_per_round: int = 30
    actor_batch_size: int = 32
    actor_pair_min_gap: int = 1
    actor_pair_max_gap: int = 5
    actor_video_cache_size: int = 8
    actor_pairs_per_video: int = 8
    actor_lr: float = 3e-4
    actor_weight_decay: float = 1e-3
    actor_grad_clip_norm: float | None = 10.0
    exploration_ou_theta: float = 0.15
    exploration_ou_std: float = 0.6
    device: str = "auto"


def parse_args() -> TrainDynamicsConfig:
    defaults = TrainDynamicsConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default=defaults.env_id)
    p.add_argument("--control-mode", default=defaults.control_mode)
    p.add_argument("--sim-backend", default=defaults.sim_backend)
    p.add_argument("--render-backend", default=defaults.render_backend)
    p.add_argument("--camera-uid", default=defaults.camera_uid)
    p.add_argument("--camera-width", type=int, default=defaults.camera_width)
    p.add_argument("--camera-height", type=int, default=defaults.camera_height)
    p.add_argument("--r3m-model-id", choices=R3M_MODEL_IDS, default=defaults.r3m_model_id)
    p.add_argument(
        "--backbone-precision",
        choices=BACKBONE_PRECISIONS,
        default=defaults.backbone_precision,
        help="R3M forward dtype. bf16/fp16 are ~2x faster on CUDA; backbone is frozen so no accuracy concern.",
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--initial-episodes", type=int, default=defaults.initial_episodes)
    p.add_argument("--collection-rounds", type=int, default=defaults.collection_rounds)
    p.add_argument("--episodes-per-round", type=int, default=defaults.episodes_per_round)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--buffer-capacity", type=int, default=defaults.buffer_capacity)
    p.add_argument("--train-steps-per-round", type=int, default=defaults.train_steps_per_round)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    p.add_argument("--hidden-dims", default=",".join(str(x) for x in defaults.hidden_dims))
    p.add_argument("--log-dir", default=defaults.log_dir)
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--log-interval", type=int, default=defaults.log_interval)
    p.add_argument(
        "--actor-video-training",
        action=argparse.BooleanOptionalAction,
        default=defaults.actor_video_training,
    )
    p.add_argument("--actor-video-dataset-dir", default=defaults.actor_video_dataset_dir)
    p.add_argument(
        "--actor-train-steps-per-round",
        type=int,
        default=defaults.actor_train_steps_per_round,
        help="Actor updates per round, run after the dynamics phase against the just-trained dynamics.",
    )
    p.add_argument("--actor-batch-size", type=int, default=defaults.actor_batch_size)
    p.add_argument("--actor-pair-min-gap", type=int, default=defaults.actor_pair_min_gap)
    p.add_argument("--actor-pair-max-gap", type=int, default=defaults.actor_pair_max_gap)
    p.add_argument("--actor-video-cache-size", type=int, default=defaults.actor_video_cache_size)
    p.add_argument(
        "--actor-pairs-per-video",
        type=int,
        default=defaults.actor_pairs_per_video,
        help="Number of (start, goal) pairs drawn per loaded video — reduces decode/cache pressure.",
    )
    p.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    p.add_argument("--actor-weight-decay", type=float, default=defaults.actor_weight_decay)
    p.add_argument("--actor-grad-clip-norm", type=float, default=defaults.actor_grad_clip_norm)
    p.add_argument(
        "--exploration-ou-theta",
        type=float,
        default=defaults.exploration_ou_theta,
        help="Decorrelation rate for AR(1)/OU exploration noise. Smaller = more committed.",
    )
    p.add_argument(
        "--exploration-ou-std",
        type=float,
        default=defaults.exploration_ou_std,
        help="Stationary std of the AR(1) exploration noise added during env collection. 0 disables.",
    )
    p.add_argument("--device", default=defaults.device)
    args = p.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x)
    if not hidden_dims:
        p.error("--hidden-dims must contain at least one layer size")
    for field in (
        "initial_episodes",
        "collection_rounds",
        "episodes_per_round",
        "max_steps",
        "buffer_capacity",
        "train_steps_per_round",
        "batch_size",
        "log_interval",
        "actor_batch_size",
        "actor_pair_min_gap",
        "actor_pair_max_gap",
        "actor_video_cache_size",
        "actor_pairs_per_video",
    ):
        if getattr(args, field) < 1:
            p.error(f"--{field.replace('_', '-')} must be >= 1")
    if args.actor_train_steps_per_round < 0:
        p.error("--actor-train-steps-per-round must be >= 0")
    if args.actor_pair_max_gap < args.actor_pair_min_gap:
        p.error("--actor-pair-max-gap must be >= --actor-pair-min-gap")
    return TrainDynamicsConfig(
        **{
            **vars(args),
            "obs_mode": "rgb",
            "hidden_dims": hidden_dims,
        }
    )


def make_env(cfg: TrainDynamicsConfig) -> gym.Env:
    camera_uid = cfg.camera_uid or "base_camera"
    sensor_configs = {
        camera_uid: {
            "width": cfg.camera_width,
            "height": cfg.camera_height,
        }
    }
    kwargs = {
        "obs_mode": cfg.obs_mode,
        "control_mode": cfg.control_mode,
        "sim_backend": cfg.sim_backend,
        "render_backend": cfg.render_backend,
        "sensor_configs": sensor_configs,
    }
    return gym.make(cfg.env_id, **kwargs)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return pick_device()
    return torch.device(name)


def create_run_dir(root: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dirname = timestamp
    if run_name:
        dirname = f"{timestamp}-{sanitize_run_name(run_name)}"
    run_dir = Path(root) / dirname
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    cleaned = cleaned.strip("-._")
    if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
        raise ValueError("--run-name must contain at least one alphanumeric character")
    return cleaned


def collect_actor_episodes(
    *,
    env: gym.Env,
    actor: Actor,
    buffer: TransitionReplayBuffer,
    backbone: torch.nn.Module,
    device: torch.device,
    cfg: TrainDynamicsConfig,
    writer: SummaryWriter,
    global_episode: int,
    global_step: int,
    phase: str,
    ou_noise: OUNoise | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[int, int]:
    progress = tqdm(
        range(cfg.episodes_per_round),
        desc=f"collect/{phase}",
        unit="episode",
        dynamic_ncols=True,
        leave=True,
    )
    for _ in progress:
        obs, _ = env.reset(seed=cfg.seed + global_episode)
        state = encode_observation(backbone, obs, device, cfg.camera_uid)
        if ou_noise is not None:
            ou_noise.reset()
        ep_return, succeeded = 0.0, False
        action_abs_sum, noise_abs_sum = 0.0, 0.0
        episode_actions: list[np.ndarray] = []
        for ep_step in range(cfg.max_steps):
            mean_action = deterministic_actor_action(actor, state, device)
            if ou_noise is not None:
                noise = ou_noise.sample()
                action = np.clip(mean_action + noise, action_low, action_high).astype(np.float32)
                noise_abs_sum += float(np.abs(noise).mean())
            else:
                action = mean_action
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_observation(backbone, next_obs, device, cfg.camera_uid)
            buffer.add(state, action, next_state)

            episode_actions.append(action)
            ep_return += float(np.asarray(reward).reshape(-1)[0])
            action_abs_sum += float(np.abs(action).mean())
            succeeded = succeeded or to_scalar_bool(info.get("success", False))
            state = next_state
            global_step += 1

            done = to_scalar_bool(terminated) or to_scalar_bool(truncated)
            if done:
                break

        action_within_episode_std = (
            float(np.stack(episode_actions, axis=0).std(axis=0).mean())
            if len(episode_actions) > 1
            else 0.0
        )
        writer.add_scalar("rollout/episode_return", ep_return, global_episode)
        writer.add_scalar("rollout/episode_success", float(succeeded), global_episode)
        writer.add_scalar("rollout/episode_steps", ep_step + 1, global_episode)
        writer.add_scalar(
            "rollout/policy_action_abs_mean",
            action_abs_sum / (ep_step + 1),
            global_episode,
        )
        writer.add_scalar(
            "rollout/exploration_noise_abs_mean",
            noise_abs_sum / (ep_step + 1) if ou_noise is not None else 0.0,
            global_episode,
        )
        writer.add_scalar(
            "rollout/action_within_episode_std",
            action_within_episode_std,
            global_episode,
        )
        writer.add_scalar("replay_buffer/size", len(buffer), global_step)
        progress.set_postfix(
            episode=global_episode,
            steps=ep_step + 1,
            ret=f"{ep_return:.2f}",
            success=int(succeeded),
            buffer=len(buffer),
            action=f"{action_abs_sum / (ep_step + 1):.2f}",
            noise=f"{(noise_abs_sum / (ep_step + 1)) if ou_noise is not None else 0.0:.2f}",
        )
        global_episode += 1
    return global_episode, global_step


def train_round(
    *,
    trainer: DynamicsTrainer,
    actor: Actor,
    actor_video_trainer: VideoActorTrainer | None,
    buffer: TransitionReplayBuffer,
    cfg: TrainDynamicsConfig,
    writer: SummaryWriter,
    device: torch.device,
    train_step: int,
    actor_train_step: int,
    round_idx: int,
    run_dir: Path,
) -> tuple[int, int]:
    train_step, last_dyn_metrics = _train_dynamics_phase(
        trainer=trainer,
        buffer=buffer,
        cfg=cfg,
        writer=writer,
        device=device,
        train_step=train_step,
        round_idx=round_idx,
    )
    actor_train_step = _train_actor_phase(
        actor_video_trainer=actor_video_trainer,
        cfg=cfg,
        writer=writer,
        actor_train_step=actor_train_step,
        round_idx=round_idx,
    )
    checkpoint_path = save_checkpoint(
        run_dir=run_dir,
        actor=actor,
        cfg=cfg,
        train_step=train_step,
        actor_train_step=actor_train_step,
        filename="actor_latest.pt",
    )
    tqdm.write(
        f"round_{round_idx} done: dyn_step={train_step} "
        f"loss={last_dyn_metrics['loss']:.6f} delta={last_dyn_metrics['delta_loss']:.6f} "
        f"actor_step={actor_train_step} | checkpoint: {checkpoint_path}"
    )
    return train_step, actor_train_step


def _train_dynamics_phase(
    *,
    trainer: DynamicsTrainer,
    buffer: TransitionReplayBuffer,
    cfg: TrainDynamicsConfig,
    writer: SummaryWriter,
    device: torch.device,
    train_step: int,
    round_idx: int,
) -> tuple[int, dict[str, float]]:
    progress = tqdm(
        range(cfg.train_steps_per_round),
        desc=f"dynamics/round_{round_idx}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )
    metrics: dict[str, float] = {"loss": float("nan"), "delta_loss": float("nan"), "grad_norm": float("nan")}
    for _ in progress:
        batch = buffer.sample(cfg.batch_size, device)
        metrics = trainer.train_step(batch)
        next_train_step = train_step + 1
        writer.add_scalar("dynamics/train_loss", metrics["loss"], train_step)
        writer.add_scalar("dynamics/grad_norm", metrics["grad_norm"], train_step)
        if train_step % 50 == 0:
            eval_metrics = trainer.eval_step(buffer.sample(cfg.batch_size, device))
            writer.add_scalar("dynamics/eval_loss", eval_metrics["loss"], train_step)
            writer.add_scalar("dynamics/eval_cosine", eval_metrics["cosine"], train_step)
            writer.add_scalar(
                "dynamics/identity_loss_ratio",
                eval_metrics["identity_loss_ratio"],
                train_step,
            )
        progress.set_postfix(
            step=next_train_step,
            loss=f"{metrics['loss']:.4f}",
            delta=f"{metrics['delta_loss']:.4f}",
            buffer=len(buffer),
        )
        if next_train_step % cfg.log_interval == 0:
            tqdm.write(
                f"dynamics step={next_train_step} round={round_idx} "
                f"loss={metrics['loss']:.6f} delta={metrics['delta_loss']:.6f} "
                f"grad={metrics['grad_norm']:.3f}"
            )
        train_step = next_train_step
    return train_step, metrics


def _train_actor_phase(
    *,
    actor_video_trainer: VideoActorTrainer | None,
    cfg: TrainDynamicsConfig,
    writer: SummaryWriter,
    actor_train_step: int,
    round_idx: int,
) -> int:
    if actor_video_trainer is None or cfg.actor_train_steps_per_round == 0:
        return actor_train_step
    progress = tqdm(
        range(cfg.actor_train_steps_per_round),
        desc=f"actor/round_{round_idx}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )
    for _ in progress:
        actor_metrics = actor_video_trainer.train_step()
        writer.add_scalar("actor/train_loss", actor_metrics["loss"], actor_train_step)
        writer.add_scalar("actor/goal_cosine", actor_metrics["cosine"], actor_train_step)
        writer.add_scalar("actor/frame_gap_mean", actor_metrics["frame_gap_mean"], actor_train_step)
        writer.add_scalar("actor/grad_norm", actor_metrics["grad_norm"], actor_train_step)
        actor_train_step += 1
        progress.set_postfix(
            step=actor_train_step,
            loss=f"{actor_metrics['loss']:.4f}",
            cos=f"{actor_metrics['cosine']:.3f}",
            gap=f"{actor_metrics['frame_gap_mean']:.1f}",
        )
    return actor_train_step


def save_checkpoint(
    *,
    run_dir: Path,
    actor: Actor,
    cfg: TrainDynamicsConfig,
    train_step: int,
    actor_train_step: int,
    filename: str,
) -> Path:
    checkpoint_path = run_dir / filename
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "actor_config": asdict(actor.config),
            "action_low": actor.action_low.detach().cpu(),
            "action_high": actor.action_high.detach().cpu(),
            "train_config": asdict(cfg),
            "train_step": train_step,
            "actor_train_step": actor_train_step,
        },
        checkpoint_path,
    )
    return checkpoint_path


@torch.no_grad()
def deterministic_actor_action(
    actor: Actor,
    state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    actor.eval()
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    return actor(state_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_actor_video_trainer(
    *,
    cfg: TrainDynamicsConfig,
    actor: Actor,
    dynamics: ForwardDynamicsModel,
    backbone: torch.nn.Module,
    device: torch.device,
) -> VideoActorTrainer | None:
    if not cfg.actor_video_training:
        return None

    sampler = VideoFramePairSampler(
        cfg.actor_video_dataset_dir,
        min_gap=cfg.actor_pair_min_gap,
        max_gap=cfg.actor_pair_max_gap,
        cache_size=cfg.actor_video_cache_size,
        pairs_per_video=cfg.actor_pairs_per_video,
        seed=cfg.seed,
    )
    return VideoActorTrainer(
        actor=actor,
        dynamics=dynamics,
        backbone=backbone,
        sampler=sampler,
        config=VideoActorTrainerConfig(
            lr=cfg.actor_lr,
            weight_decay=cfg.actor_weight_decay,
            grad_clip_norm=cfg.actor_grad_clip_norm,
            batch_size=cfg.actor_batch_size,
        ),
        device=device,
    )


def run(cfg: TrainDynamicsConfig) -> None:
    device = select_device(cfg.device)
    run_dir = create_run_dir(cfg.log_dir, cfg.run_name)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    writer = SummaryWriter(log_dir=str(run_dir))
    try:
        action_space = getattr(env, "single_action_space", env.action_space)
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
        action_dim = int(np.prod(action_space.shape))

        backbone = build_backbone(
            device,
            model_id=cfg.r3m_model_id,
            precision=cfg.backbone_precision,
        )
        model = ForwardDynamicsModel(
            ForwardDynamicsConfig(
                action_dim=action_dim,
                state_dim=R3M_FEAT_DIM,
                hidden_dims=cfg.hidden_dims,
            )
        ).to(device)
        trainer = DynamicsTrainer(
            model,
            DynamicsTrainerConfig(
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                grad_clip_norm=cfg.grad_clip_norm,
            ),
        )
        buffer = TransitionReplayBuffer(
            cfg.buffer_capacity,
            state_dim=R3M_FEAT_DIM,
            action_dim=action_dim,
            seed=cfg.seed,
        )
        actor = Actor(
            ActorConfig(action_dim=action_dim, state_dim=R3M_FEAT_DIM),
            action_low=torch.as_tensor(action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(action_space.high, dtype=torch.float32),
        ).to(device)
        ou_noise = (
            OUNoise(
                action_dim=action_dim,
                theta=cfg.exploration_ou_theta,
                stationary_std=cfg.exploration_ou_std,
                seed=cfg.seed,
            )
            if cfg.exploration_ou_std > 0.0
            else None
        )
        action_low_np = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        action_high_np = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        actor_video_trainer = build_actor_video_trainer(
            cfg=cfg,
            actor=actor,
            dynamics=model,
            backbone=backbone,
            device=device,
        )

        writer.add_text("config", repr(cfg), 0)
        writer.add_text("model", repr(model), 0)
        writer.add_text("actor", repr(actor), 0)
        tqdm.write(f"Logging TensorBoard events to {run_dir}")
        tqdm.write(f"Device: {device} | action_dim={action_dim} | state_dim={R3M_FEAT_DIM}")
        tqdm.write(
            f"Plan: collect {cfg.initial_episodes} initial episodes, then run "
            f"{cfg.collection_rounds} rounds x {cfg.train_steps_per_round} train steps "
            f"with {cfg.episodes_per_round} new episodes after each round."
        )
        if actor_video_trainer is not None:
            tqdm.write(
                f"Actor video training: {cfg.actor_train_steps_per_round} updates per round "
                f"(after dynamics phase), horizon=gap "
                f"~U[{cfg.actor_pair_min_gap},{cfg.actor_pair_max_gap}], "
                f"dataset={cfg.actor_video_dataset_dir}"
            )

        global_episode, global_step, train_step, actor_train_step = 0, 0, 0, 0
        initial_cfg = TrainDynamicsConfig(
            **{
                **cfg.__dict__,
                "episodes_per_round": cfg.initial_episodes,
            }
        )
        global_episode, global_step = collect_actor_episodes(
            env=env,
            actor=actor,
            buffer=buffer,
            backbone=backbone,
            device=device,
            cfg=initial_cfg,
            writer=writer,
            global_episode=global_episode,
            global_step=global_step,
            phase="initial_collect",
            ou_noise=ou_noise,
            action_low=action_low_np,
            action_high=action_high_np,
        )

        for round_idx in range(cfg.collection_rounds):
            tqdm.write(f"== dynamics training round {round_idx} ==")
            train_step, actor_train_step = train_round(
                trainer=trainer,
                actor=actor,
                actor_video_trainer=actor_video_trainer,
                buffer=buffer,
                cfg=cfg,
                writer=writer,
                device=device,
                train_step=train_step,
                actor_train_step=actor_train_step,
                round_idx=round_idx,
                run_dir=run_dir,
            )
            writer.add_scalar("progress/round", round_idx, train_step)
            tqdm.write(f"== sample generation round {round_idx} ==")
            global_episode, global_step = collect_actor_episodes(
                env=env,
                actor=actor,
                buffer=buffer,
                backbone=backbone,
                device=device,
                cfg=cfg,
                writer=writer,
                global_episode=global_episode,
                global_step=global_step,
                phase=f"collect_round_{round_idx}",
                ou_noise=ou_noise,
                action_low=action_low_np,
                action_high=action_high_np,
            )

        checkpoint_path = save_checkpoint(
            run_dir=run_dir,
            actor=actor,
            cfg=cfg,
            train_step=train_step,
            actor_train_step=actor_train_step,
            filename="actor.pt",
        )
        tqdm.write(
            f"Finished training: episodes={global_episode} transitions={len(buffer)} "
            f"env_steps={global_step} train_steps={train_step} "
            f"actor_train_steps={actor_train_step}"
        )
        tqdm.write(f"Saved actor checkpoint to {checkpoint_path}")
    finally:
        writer.close()
        env.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
