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
import torch
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import io_utils
from tqdm import tqdm


DEFAULT_PICKCUBE_TRAJECTORY = (
    "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5"
)


@dataclass(frozen=True)
class ExpertVideoDatasetConfig:
    trajectory_path: str = DEFAULT_PICKCUBE_TRAJECTORY
    output_dir: str = "data/pickcube_expert_videos"
    env_id: str | None = None
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"
    camera_uid: str = "base_camera"
    width: int = 224
    height: int = 224
    max_episodes: int | None = None
    frame_stride: int = 1
    fps: int = 20
    overwrite: bool = False


@dataclass(frozen=True)
class EpisodeVideoRecord:
    episode_id: int
    episode_seed: int | None
    video_path: str
    num_frames: int
    height: int
    width: int
    camera_uid: str
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

    metadata = io_utils.load_json(str(json_path))
    env_id = cfg.env_id or metadata["env_info"]["env_id"]
    episodes = metadata["episodes"]
    if cfg.max_episodes is not None:
        episodes = episodes[: cfg.max_episodes]

    env = make_replay_env(metadata, cfg, env_id)
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
) -> EpisodeVideoRecord:
    episode_id = int(episode["episode_id"])
    traj_key = f"traj_{episode_id}"
    if traj_key not in h5_file:
        raise KeyError(f"{traj_key} not found in {trajectory_path}")

    frame_name = f"episode_{episode_id:06d}"
    video_path = videos_dir / f"{frame_name}.mp4"
    if video_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{video_path} already exists; pass --overwrite to replace it")

    env.reset(**episode["reset_kwargs"])
    env_states = trajectory_utils.dict_to_list_of_dicts(h5_file[traj_key]["env_states"])
    frames = []
    for state_index, env_state in enumerate(env_states):
        if state_index % cfg.frame_stride != 0:
            continue
        env.unwrapped.set_state_dict(env_state)
        obs = env.unwrapped.get_obs()
        frames.append(extract_rgb(obs, cfg.camera_uid))
    if not frames:
        raise ValueError(f"{traj_key} produced no frames")

    rgb = np.stack(frames, axis=0)
    iio.imwrite(video_path, rgb, fps=cfg.fps)

    return EpisodeVideoRecord(
        episode_id=episode_id,
        episode_seed=episode.get("episode_seed"),
        video_path=str(video_path),
        num_frames=int(rgb.shape[0]),
        height=int(rgb.shape[1]),
        width=int(rgb.shape[2]),
        camera_uid=cfg.camera_uid,
        source_trajectory=str(trajectory_path),
    )


def extract_rgb(obs: Any, camera_uid: str) -> np.ndarray:
    try:
        rgb = obs["sensor_data"][camera_uid]["rgb"]
    except KeyError as exc:
        raise KeyError(f"camera_uid={camera_uid!r} not found in observation") from exc
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise ValueError(f"expected single-env RGB batch, got shape {rgb.shape}")
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError(f"expected RGB image rank 3, got shape {rgb.shape}")
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB last dimension 3, got shape {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb
    rgb_float = rgb.astype(np.float32)
    if rgb_float.size > 0 and rgb_float.max() <= 1.0:
        rgb_float *= 255.0
    return np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)


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
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
