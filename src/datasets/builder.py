"""Build per-episode datasets from any ManiSkill trajectory.h5.

For each episode the build replays the recorded actions through a fresh env
(in ``obs_mode="state"`` so the saved state vector matches what BC/RL training
will see at eval time) and writes:

    state/episode_NNNNNN.npy     (T+1, state_dim)   env obs_mode=state
    actions/episode_NNNNNN.npy   (T,   action_dim)  source h5 actions, verbatim
    proprio/episode_NNNNNN.npy   (T+1, proprio_dim) qpos+qvel via env.agent
    videos/episode_NNNNNN.mp4    (optional, written by a second pass)

Plus ``manifest.jsonl`` (one JSON object per episode) and ``metadata.json``
(env_id, control_mode, dims, source).

Replay is provably faithful by construction: the env we step is the same env
the policy will be evaluated in, and actions come from the source h5 unchanged.
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

from src.observations import extract_proprio, extract_rgb


@dataclass(frozen=True)
class BuildConfig:
    source_h5: str
    output_dir: str
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cpu"
    render_backend: str = "gpu"
    max_episodes: int | None = None
    include_video: bool = False
    camera_uid: str = "base_camera"
    width: int = 224
    height: int = 224
    fps: int = 20
    overwrite: bool = False


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    episode_seed: int | None
    state_path: str
    actions_path: str
    proprio_path: str
    video_path: str | None
    num_actions: int
    state_dim: int
    proprio_dim: int
    control_mode: str
    source_h5: str


def build_dataset(cfg: BuildConfig) -> list[EpisodeRecord]:
    source_h5 = Path(cfg.source_h5).expanduser()
    if not source_h5.exists():
        raise FileNotFoundError(f"source h5 not found: {source_h5}")
    json_path = source_h5.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"source metadata json not found: {json_path}")

    metadata = io_utils.load_json(str(json_path))
    env_id = metadata["env_info"]["env_id"]
    source_control_mode = metadata["env_info"]["env_kwargs"].get("control_mode")
    if source_control_mode != cfg.control_mode:
        raise ValueError(
            f"Source h5 control_mode={source_control_mode!r} but build "
            f"control_mode={cfg.control_mode!r}. Stepping the env with the "
            f"h5's actions only reproduces the trajectory when both match. "
            f"Convert upstream with "
            f"`mani_skill.trajectory.replay_trajectory --target-control-mode "
            f"{cfg.control_mode} --save-traj -o state`."
        )

    output_dir = Path(cfg.output_dir)
    state_dir = output_dir / "state"
    actions_dir = output_dir / "actions"
    proprio_dir = output_dir / "proprio"
    videos_dir = output_dir / "videos"
    for d in (state_dir, actions_dir, proprio_dir):
        d.mkdir(parents=True, exist_ok=True)
    if cfg.include_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    episodes = metadata["episodes"]
    if cfg.max_episodes is not None:
        episodes = episodes[: cfg.max_episodes]

    # Pass 1: state + proprio + actions. Always run.
    state_env = _make_env(metadata, cfg, obs_mode="state")
    try:
        with h5py.File(source_h5, "r") as h5_file:
            records = [
                _build_state_record(
                    env=state_env,
                    h5_file=h5_file,
                    episode=episode,
                    cfg=cfg,
                    state_dir=state_dir,
                    actions_dir=actions_dir,
                    proprio_dir=proprio_dir,
                    source_h5=source_h5,
                )
                for episode in tqdm(episodes, desc="build/state", unit="episode")
            ]
    finally:
        state_env.close()

    # Pass 2: rgb video. Only if requested.
    if cfg.include_video:
        rgb_env = _make_env(metadata, cfg, obs_mode="rgb")
        try:
            with h5py.File(source_h5, "r") as h5_file:
                records = [
                    _attach_video(
                        env=rgb_env,
                        h5_file=h5_file,
                        episode=episode,
                        record=record,
                        cfg=cfg,
                        videos_dir=videos_dir,
                    )
                    for record, episode in tqdm(
                        list(zip(records, episodes, strict=True)),
                        desc="build/video",
                        unit="episode",
                    )
                ]
        finally:
            rgb_env.close()

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")
    _write_dataset_metadata(output_dir, cfg, metadata, records)
    return records


def _make_env(
    metadata: dict[str, Any], cfg: BuildConfig, *, obs_mode: str
) -> gym.Env:
    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(
        {
            "obs_mode": obs_mode,
            "control_mode": cfg.control_mode,
            "sim_backend": cfg.sim_backend,
            "render_backend": cfg.render_backend,
            "num_envs": 1,
            # Source kwargs typically have reconfiguration_freq=1 (per-reset scene
            # rebuild, useful for GPU collection). We snap to env_states after
            # every reset, so reconfiguration is pure overhead — and on macOS
            # Vulkan it breaks the camera buffer for the rgb pass, producing
            # all-green frames from episode 1 onward. Disable it.
            "reconfiguration_freq": 0,
        }
    )
    if obs_mode == "rgb":
        env_kwargs["sensor_configs"] = {
            cfg.camera_uid: {"width": cfg.width, "height": cfg.height}
        }
    return gym.make(metadata["env_info"]["env_id"], **env_kwargs)


def _build_state_record(
    *,
    env: gym.Env,
    h5_file: h5py.File,
    episode: dict[str, Any],
    cfg: BuildConfig,
    state_dir: Path,
    actions_dir: Path,
    proprio_dir: Path,
    source_h5: Path,
) -> EpisodeRecord:
    episode_id = int(episode["episode_id"])
    traj_key = f"traj_{episode_id}"
    if traj_key not in h5_file:
        raise KeyError(f"{traj_key} not found in {source_h5}")

    stem = f"episode_{episode_id:06d}"
    state_path = state_dir / f"{stem}.npy"
    actions_path = actions_dir / f"{stem}.npy"
    proprio_path = proprio_dir / f"{stem}.npy"
    for p in (state_path, actions_path, proprio_path):
        if p.exists() and not cfg.overwrite:
            raise FileExistsError(f"{p} already exists; pass --overwrite to replace")

    actions = np.asarray(h5_file[traj_key]["actions"][:], dtype=np.float32)
    env_states = trajectory_utils.dict_to_list_of_dicts(h5_file[traj_key]["env_states"])

    env.reset(**episode.get("reset_kwargs", {}))
    env.unwrapped.set_state_dict(env_states[0])

    states: list[np.ndarray] = []
    proprios: list[np.ndarray] = []

    obs = env.unwrapped.get_obs()
    states.append(_obs_to_state(obs))
    proprios.append(extract_proprio(env))

    for action in actions:
        obs, _r, _t, _tr, _info = env.step(action)
        states.append(_obs_to_state(obs))
        proprios.append(extract_proprio(env))

    state_arr = np.stack(states, axis=0).astype(np.float32)
    proprio_arr = np.stack(proprios, axis=0).astype(np.float32)
    np.save(state_path, state_arr)
    np.save(proprio_path, proprio_arr)
    np.save(actions_path, actions)

    return EpisodeRecord(
        episode_id=episode_id,
        episode_seed=episode.get("episode_seed"),
        state_path=str(state_path),
        actions_path=str(actions_path),
        proprio_path=str(proprio_path),
        video_path=None,
        num_actions=int(actions.shape[0]),
        state_dim=int(state_arr.shape[1]),
        proprio_dim=int(proprio_arr.shape[1]),
        control_mode=cfg.control_mode,
        source_h5=str(source_h5),
    )


def _attach_video(
    *,
    env: gym.Env,
    h5_file: h5py.File,
    episode: dict[str, Any],
    record: EpisodeRecord,
    cfg: BuildConfig,
    videos_dir: Path,
) -> EpisodeRecord:
    episode_id = record.episode_id
    traj_key = f"traj_{episode_id}"
    stem = f"episode_{episode_id:06d}"
    video_path = videos_dir / f"{stem}.mp4"
    if video_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{video_path} already exists; pass --overwrite to replace")

    actions = np.asarray(h5_file[traj_key]["actions"][:], dtype=np.float32)
    env_states = trajectory_utils.dict_to_list_of_dicts(h5_file[traj_key]["env_states"])

    env.reset(**episode.get("reset_kwargs", {}))
    env.unwrapped.set_state_dict(env_states[0])

    frames: list[np.ndarray] = [extract_rgb(env.unwrapped.get_obs(), cfg.camera_uid)]
    for action in actions:
        obs, _r, _t, _tr, _info = env.step(action)
        frames.append(extract_rgb(obs, cfg.camera_uid))
    # Explicit codec/pix_fmt: imageio defaults can produce mp4s that QuickTime
    # / Finder preview decode with a green tint on macOS. yuv420p + libx264 is
    # the universally-decodable combination.
    iio.imwrite(
        video_path,
        np.stack(frames, axis=0),
        fps=cfg.fps,
        codec="libx264",
        pixelformat="yuv420p",
    )
    return EpisodeRecord(**{**asdict(record), "video_path": str(video_path)})


def _obs_to_state(obs: Any) -> np.ndarray:
    """In obs_mode=state the env returns a flat float vector (numpy or tensor)."""
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _write_dataset_metadata(
    output_dir: Path,
    cfg: BuildConfig,
    source_metadata: dict[str, Any],
    records: list[EpisodeRecord],
) -> None:
    if not records:
        raise ValueError("no episodes built")
    metadata = {
        "config": asdict(cfg),
        "source_env_info": source_metadata["env_info"],
        "num_episodes": len(records),
        "num_actions": sum(r.num_actions for r in records),
        "state_dim": records[0].state_dim,
        "proprio_dim": records[0].proprio_dim,
        "has_video": all(r.video_path is not None for r in records),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
