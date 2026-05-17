"""Build video+action+state datasets from ManiSkill expert trajectories.

Replays each episode by ``env.step``-ing through the actions stored in the H5
file. Source trajectories must already be in the target control mode — use
``mani_skill.trajectory.replay_trajectory --target-control-mode <mode>``
upstream to convert if needed. Stored actions are then by construction the
actions that drive the env from one recorded state to the next, so dataset
``actions[i]`` is exactly what an actor should output at ``frame[i]`` to reach
``frame[i+1]``.
"""

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

from src.observations import extract_privileged_state, extract_proprio, extract_rgb


DEFAULT_PICKCUBE_TRAJECTORY = (
    "~/.maniskill/demos/PickCube-v1/motionplanning/"
    "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
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
    fps: int = 20
    overwrite: bool = False


@dataclass(frozen=True)
class EpisodeVideoRecord:
    episode_id: int
    episode_seed: int | None
    video_path: str
    actions_path: str
    proprio_path: str
    state_path: str
    num_frames: int
    num_actions: int
    proprio_dim: int
    state_dim: int
    height: int
    width: int
    camera_uid: str
    control_mode: str
    source_trajectory: str


def build_expert_video_dataset(cfg: ExpertVideoDatasetConfig) -> list[EpisodeVideoRecord]:
    if cfg.max_episodes is not None and cfg.max_episodes < 1:
        raise ValueError(f"max_episodes must be >= 1 when set, got {cfg.max_episodes}")

    trajectory_path = Path(cfg.trajectory_path).expanduser()
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Trajectory file does not exist: {trajectory_path}. "
            "Download with `uv run python -m mani_skill.utils.download_demo PickCube-v1` and "
            "convert to your target control mode with `mani_skill.trajectory.replay_trajectory "
            "--target-control-mode <mode> --save-traj --use-env-states`."
        )
    json_path = trajectory_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory metadata JSON does not exist: {json_path}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    actions_dir = output_dir / "actions"
    proprio_dir = output_dir / "proprio"
    state_dir = output_dir / "state"
    for d in (videos_dir, actions_dir, proprio_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    metadata = io_utils.load_json(str(json_path))
    source_control_mode = metadata["env_info"]["env_kwargs"].get("control_mode")
    if source_control_mode != cfg.control_mode:
        raise ValueError(
            f"Source trajectory control_mode is {source_control_mode!r} but "
            f"cfg.control_mode is {cfg.control_mode!r}. Stepping the env with the "
            f"H5's stored actions only reproduces the trajectory when both match. "
            f"Convert the trajectory with `mani_skill.trajectory.replay_trajectory "
            f"--target-control-mode {cfg.control_mode} --save-traj --use-env-states`."
        )
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
                    actions_dir=actions_dir,
                    proprio_dir=proprio_dir,
                    state_dir=state_dir,
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
    state_dir: Path,
) -> EpisodeVideoRecord:
    episode_id = int(episode["episode_id"])
    traj_key = f"traj_{episode_id}"
    if traj_key not in h5_file:
        raise KeyError(f"{traj_key} not found in {trajectory_path}")

    frame_name = f"episode_{episode_id:06d}"
    video_path = videos_dir / f"{frame_name}.mp4"
    actions_path = actions_dir / f"{frame_name}.npy"
    proprio_path = proprio_dir / f"{frame_name}.npy"
    state_path = state_dir / f"{frame_name}.npy"
    for p in (video_path, actions_path, proprio_path, state_path):
        if p.exists() and not cfg.overwrite:
            raise FileExistsError(f"{p} already exists; pass --overwrite to replace it")

    actions = np.asarray(h5_file[traj_key]["actions"][:], dtype=np.float32)
    env_states = trajectory_utils.dict_to_list_of_dicts(h5_file[traj_key]["env_states"])

    # Reset, then snap to the recorded initial state so the upcoming step()s
    # run from exactly the same configuration the actions were recorded under.
    env.reset(**episode["reset_kwargs"])
    env.unwrapped.set_state_dict(env_states[0])

    frames: list[np.ndarray] = []
    proprio_per_frame: list[np.ndarray] = []
    state_per_frame: list[np.ndarray] = []

    obs = env.unwrapped.get_obs()
    frames.append(extract_rgb(obs, cfg.camera_uid))
    proprio_per_frame.append(extract_proprio(env))
    state_per_frame.append(extract_privileged_state(obs))

    for action in actions:
        obs, _r, _term, _trunc, _info = env.step(action)
        frames.append(extract_rgb(obs, cfg.camera_uid))
        proprio_per_frame.append(extract_proprio(env))
        state_per_frame.append(extract_privileged_state(obs))

    rgb = np.stack(frames, axis=0)
    iio.imwrite(video_path, rgb, fps=cfg.fps)
    proprio_arr = np.stack(proprio_per_frame, axis=0).astype(np.float32)
    np.save(proprio_path, proprio_arr)
    state_arr = np.stack(state_per_frame, axis=0).astype(np.float32)
    np.save(state_path, state_arr)
    np.save(actions_path, actions)

    return EpisodeVideoRecord(
        episode_id=episode_id,
        episode_seed=episode.get("episode_seed"),
        video_path=str(video_path),
        actions_path=str(actions_path),
        proprio_path=str(proprio_path),
        state_path=str(state_path),
        num_frames=int(rgb.shape[0]),
        num_actions=int(actions.shape[0]),
        proprio_dim=int(proprio_arr.shape[1]),
        state_dim=int(state_arr.shape[1]),
        height=int(rgb.shape[1]),
        width=int(rgb.shape[2]),
        camera_uid=cfg.camera_uid,
        control_mode=cfg.control_mode,
        source_trajectory=str(trajectory_path),
    )


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
