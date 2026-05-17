"""Build image-only video datasets from ManiSkill expert trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v3 as iio
import mani_skill.envs  # noqa: F401
import numpy as np
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import io_utils
from tqdm import tqdm

from src.observations import extract_proprio, extract_rgb


DEFAULT_PICKCUBE_TRAJECTORY = (
    "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5"
)


@dataclass(frozen=True)
class ExpertVideoDatasetConfig:
    trajectory_path: str = DEFAULT_PICKCUBE_TRAJECTORY
    output_dir: str = "data/pickcube_expert_videos"
    env_id: str | None = None
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"
    camera_uid: str = "base_camera"
    width: int = 224
    height: int = 224
    max_episodes: int | None = None
    frame_stride: int = 1
    fps: int = 20
    overwrite: bool = False
    save_actions: bool = True


@dataclass(frozen=True)
class EpisodeVideoRecord:
    episode_id: int
    episode_seed: int | None
    video_path: str
    actions_path: str | None
    proprio_path: str | None
    num_frames: int
    num_actions: int
    proprio_dim: int
    height: int
    width: int
    camera_uid: str
    control_mode: str
    source_trajectory: str


def build_expert_video_dataset(cfg: ExpertVideoDatasetConfig) -> list[EpisodeVideoRecord]:
    if cfg.frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {cfg.frame_stride}")
    if cfg.max_episodes is not None and cfg.max_episodes < 1:
        raise ValueError(f"max_episodes must be >= 1 when set, got {cfg.max_episodes}")

    trajectory_path = Path(cfg.trajectory_path).expanduser()
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Trajectory file does not exist: {trajectory_path}. "
            "Download it with: uv run python -m mani_skill.utils.download_demo PickCube-v1"
        )
    json_path = trajectory_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory metadata JSON does not exist: {json_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = output_dir / "actions"
    proprio_dir = output_dir / "proprio"
    if cfg.save_actions:
        actions_dir.mkdir(parents=True, exist_ok=True)
    proprio_dir.mkdir(parents=True, exist_ok=True)

    metadata = io_utils.load_json(str(json_path))
    env_id = cfg.env_id or metadata["env_info"]["env_id"]
    episodes = metadata["episodes"]
    if cfg.max_episodes is not None:
        episodes = episodes[: cfg.max_episodes]

    env = make_replay_env(metadata, cfg, env_id)
    action_space = getattr(env, "single_action_space", env.action_space)
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Expected Box action space, got {type(action_space).__name__}")
    action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    action_dim = int(action_low.shape[0])
    action_spec = _build_action_spec(env, action_dim)

    manifest_path = output_dir / "manifest.jsonl"
    records: list[EpisodeVideoRecord] = []
    try:
        with h5py.File(trajectory_path, "r") as h5_file, manifest_path.open(
            "w", encoding="utf-8"
        ) as manifest:
            for episode in tqdm(episodes, desc="building expert videos", unit="episode"):
                record = build_episode_record(
                    env=env,
                    h5_file=h5_file,
                    episode=episode,
                    cfg=cfg,
                    trajectory_path=trajectory_path,
                    videos_dir=videos_dir,
                    actions_dir=actions_dir,
                    proprio_dir=proprio_dir,
                    action_dim=action_dim,
                    action_low=action_low,
                    action_high=action_high,
                    action_spec=action_spec,
                )
                records.append(record)
                manifest.write(json.dumps(asdict(record)) + "\n")
    finally:
        env.close()

    write_dataset_metadata(output_dir, cfg, metadata, records)
    return records


def make_replay_env(
    metadata: dict[str, Any],
    cfg: ExpertVideoDatasetConfig,
    env_id: str,
) -> gym.Env:
    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(
        {
            "obs_mode": "rgb",
            "control_mode": cfg.control_mode,
            "sim_backend": cfg.sim_backend,
            "render_backend": cfg.render_backend,
            "sensor_configs": {
                cfg.camera_uid: {
                    "width": cfg.width,
                    "height": cfg.height,
                }
            },
            "num_envs": 1,
        }
    )
    return gym.make(env_id, **env_kwargs)


def build_episode_record(
    *,
    env: gym.Env,
    h5_file: h5py.File,
    episode: dict[str, Any],
    cfg: ExpertVideoDatasetConfig,
    trajectory_path: Path,
    videos_dir: Path,
    actions_dir: Path,
    proprio_dir: Path,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_spec: list["_ControllerSegment"],
) -> EpisodeVideoRecord:
    episode_id = int(episode["episode_id"])
    traj_key = f"traj_{episode_id}"
    if traj_key not in h5_file:
        raise KeyError(f"{traj_key} not found in {trajectory_path}")

    frame_name = f"episode_{episode_id:06d}"
    video_path = videos_dir / f"{frame_name}.mp4"
    actions_path = actions_dir / f"{frame_name}.npy"
    proprio_path = proprio_dir / f"{frame_name}.npy"
    if video_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{video_path} already exists; pass --overwrite to replace it")
    if cfg.save_actions and actions_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{actions_path} already exists; pass --overwrite to replace it")
    if proprio_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{proprio_path} already exists; pass --overwrite to replace it")

    env.reset(**episode["reset_kwargs"])
    env_states = trajectory_utils.dict_to_list_of_dicts(h5_file[traj_key]["env_states"])
    frames: list[np.ndarray] = []
    qpos_per_frame: list[np.ndarray] = []
    proprio_per_frame: list[np.ndarray] = []
    for state_index, env_state in enumerate(env_states):
        if state_index % cfg.frame_stride != 0:
            continue
        env.unwrapped.set_state_dict(env_state)
        obs = env.unwrapped.get_obs()
        frames.append(extract_rgb(obs, cfg.camera_uid))
        proprio_per_frame.append(extract_proprio(env))
        if cfg.save_actions:
            qpos_per_frame.append(_get_qpos(env))
    if not frames:
        raise ValueError(f"{traj_key} produced no frames")

    rgb = np.stack(frames, axis=0)
    iio.imwrite(video_path, rgb, fps=cfg.fps)
    proprio_arr = np.stack(proprio_per_frame, axis=0).astype(np.float32)
    np.save(proprio_path, proprio_arr)

    actions_arr: np.ndarray | None = None
    if cfg.save_actions:
        actions_arr = _derive_actions(
            qpos_per_frame, action_dim, action_low, action_high, action_spec
        )
        np.save(actions_path, actions_arr)

    return EpisodeVideoRecord(
        episode_id=episode_id,
        episode_seed=episode.get("episode_seed"),
        video_path=str(video_path),
        actions_path=str(actions_path) if cfg.save_actions else None,
        proprio_path=str(proprio_path),
        num_frames=int(rgb.shape[0]),
        num_actions=int(actions_arr.shape[0]) if actions_arr is not None else 0,
        proprio_dim=int(proprio_arr.shape[1]),
        height=int(rgb.shape[1]),
        width=int(rgb.shape[2]),
        camera_uid=cfg.camera_uid,
        control_mode=cfg.control_mode,
        source_trajectory=str(trajectory_path),
    )


def _get_qpos(env: gym.Env) -> np.ndarray:
    """Return the robot's current generalized joint positions as a 1-D float32 array."""
    qpos = env.unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, "detach"):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    return qpos


@dataclass(frozen=True)
class _ControllerSegment:
    """Where a controller writes into the action vector and how it maps qpos."""

    action_slice: slice
    qpos_indices: np.ndarray  # one per action dim (mimicked joints collapse to primary)
    use_delta: bool
    lower: float
    upper: float
    normalize_action: bool


def _build_action_spec(env: gym.Env, action_dim: int) -> list[_ControllerSegment]:
    """Inspect the active controller to map qpos -> action vector segments.

    Reads each sub-controller's joint names, bounds, and ``normalize_action`` so
    derived actions land in the same units the env expects from a live actor.
    """
    agent = env.unwrapped.agent
    robot = agent.robot
    active_joint_names = [j.name for j in robot.get_active_joints()]
    name_to_qpos_idx = {name: i for i, name in enumerate(active_joint_names)}

    controller = agent.controller
    if not hasattr(controller, "configs"):
        raise RuntimeError(
            "active controller has no .configs; cannot derive normalized actions"
        )

    segments: list[_ControllerSegment] = []
    cursor = 0
    for sub_name, sub_cfg in controller.configs.items():
        joint_names = list(sub_cfg.joint_names)
        mimic = getattr(sub_cfg, "mimic", None) or {}
        # primary (independent) joints: those NOT in mimic.keys()
        primary_joints = [name for name in joint_names if name not in mimic]
        if not primary_joints:
            raise RuntimeError(f"controller segment {sub_name!r} has no primary joints")
        seg_action_dim = len(primary_joints)
        qpos_indices = np.asarray(
            [name_to_qpos_idx[name] for name in primary_joints], dtype=np.int64
        )
        segments.append(
            _ControllerSegment(
                action_slice=slice(cursor, cursor + seg_action_dim),
                qpos_indices=qpos_indices,
                use_delta=bool(getattr(sub_cfg, "use_delta", False)),
                lower=float(sub_cfg.lower),
                upper=float(sub_cfg.upper),
                normalize_action=bool(getattr(sub_cfg, "normalize_action", True)),
            )
        )
        cursor += seg_action_dim
    if cursor != action_dim:
        raise RuntimeError(
            f"controller segments cover {cursor} action dims but env.action_space has "
            f"{action_dim}; controller layout is not what we expected"
        )
    return segments


def _derive_actions(
    qpos_per_frame: list[np.ndarray],
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_spec: list[_ControllerSegment],
) -> np.ndarray:
    """Derive actions matching the env's controller convention.

    For each sub-controller: take raw qpos (or qpos delta) at the relevant
    joints, then map to the controller's action range (with normalization if
    ``normalize_action=True``). Approximate (assumes one-step PD tracking),
    but unit-consistent with what the live actor emits.
    """
    n = len(qpos_per_frame) - 1
    actions = np.empty((n, action_dim), dtype=np.float32)
    for i in range(n):
        qpos_t = qpos_per_frame[i]
        qpos_tp1 = qpos_per_frame[i + 1]
        for seg in action_spec:
            raw = (
                qpos_tp1[seg.qpos_indices] - qpos_t[seg.qpos_indices]
                if seg.use_delta
                else qpos_tp1[seg.qpos_indices]
            )
            if seg.normalize_action:
                mid = 0.5 * (seg.lower + seg.upper)
                half = 0.5 * (seg.upper - seg.lower)
                norm = (raw - mid) / half
            else:
                norm = raw
            actions[i, seg.action_slice] = norm
    return np.clip(actions, action_low, action_high).astype(np.float32)


def write_dataset_metadata(
    output_dir: Path,
    cfg: ExpertVideoDatasetConfig,
    source_metadata: dict[str, Any],
    records: list[EpisodeVideoRecord],
) -> None:
    metadata = {
        "config": asdict(cfg),
        "source_env_info": source_metadata["env_info"],
        "num_episodes": len(records),
        "num_frames": sum(record.num_frames for record in records),
        "num_actions": sum(record.num_actions for record in records),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
