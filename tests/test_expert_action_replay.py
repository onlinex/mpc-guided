"""Diagnostic: replay derived expert actions in the env, verify they solve the task.

If this test fails, it means ``_derive_actions`` in src/datasets/expert_videos.py
is producing action labels that do NOT reproduce the expert trajectory — and any
downstream BC trained on those labels is doomed. This is the single most useful
red-flag indicator we have for the upstream data pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import io_utils

from src.datasets.expert_videos import ExpertVideoDatasetConfig, make_replay_env


pytestmark = [pytest.mark.env, pytest.mark.dataset]


def _load_manifest(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (dataset_dir / "manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_scalar(value) -> float:
    arr = np.asarray(value).reshape(-1)
    return float(arr[0])


def _resolve(path_str: str, dataset_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return dataset_dir / p


def _source_episode(entry: dict) -> tuple[Path, dict, dict]:
    """Return (h5_path, episode_metadata_entry, env_state_0) for the given dataset entry."""
    h5_path = Path(entry["source_trajectory"]).expanduser()
    meta = json.loads(h5_path.with_suffix(".json").read_text())
    ep_meta = next(e for e in meta["episodes"] if int(e["episode_id"]) == int(entry["episode_id"]))
    with h5py.File(h5_path, "r") as h5:
        env_states = trajectory_utils.dict_to_list_of_dicts(
            h5[f"traj_{entry['episode_id']}"]["env_states"]
        )
    return h5_path, ep_meta, env_states[0]


def _reset_like_build(env, ep_meta: dict, env_state_0: dict) -> None:
    env.reset(**ep_meta["reset_kwargs"])
    env.unwrapped.set_state_dict(env_state_0)


def _build_like_env(entry: dict):
    """Construct a replay env identical to the one the dataset was built with.

    Copies env_kwargs from the source H5 metadata so any divergence between the
    test and the dataset build is real, not env-config drift.
    """
    h5_path = Path(entry["source_trajectory"]).expanduser()
    source_metadata = io_utils.load_json(str(h5_path.with_suffix(".json")))
    cfg = ExpertVideoDatasetConfig(
        trajectory_path=str(h5_path),
        control_mode=entry.get("control_mode", "pd_joint_delta_pos"),
        camera_uid=entry.get("camera_uid", "base_camera"),
        width=int(entry.get("width", 224)),
        height=int(entry.get("height", 224)),
    )
    env_id = source_metadata["env_info"]["env_id"]
    return make_replay_env(source_metadata, cfg, env_id)


@pytest.mark.parametrize("episode_index", [0, 1, 2])
def test_derived_actions_solve_episode(dataset_dir, pickcube_env_factory, episode_index):
    """Replaying the dataset's derived actions in the env should complete the task.

    Asserts that ``info['success']`` becomes True before the action stream ends.
    A failure here means the BC label generation is wrong (likely cause: the
    one-step PD-tracking assumption in ``_derive_actions`` doesn't reproduce
    the expert behavior under the configured control_mode).
    """
    manifest = _load_manifest(dataset_dir)
    if episode_index >= len(manifest):
        pytest.skip(f"dataset only has {len(manifest)} episodes")

    entry = manifest[episode_index]
    if not entry.get("actions_path"):
        pytest.skip("manifest entry missing actions_path")
    actions = np.load(_resolve(entry["actions_path"], dataset_dir))
    control_mode = entry.get("control_mode", "pd_joint_delta_pos")
    _, ep_meta, env_state_0 = _source_episode(entry)

    env = pickcube_env_factory(control_mode=control_mode)
    try:
        _reset_like_build(env, ep_meta, env_state_0)
        succeeded = False
        total_return = 0.0
        for action in actions:
            _obs, reward, _terminated, _truncated, info = env.step(action.astype(np.float32))
            total_return += _to_scalar(reward)
            if _to_scalar(info.get("success", False)) > 0:
                succeeded = True
            # Don't break on terminated/truncated: ManiSkill flags `terminated`
            # the moment success is reached, but the demos in the H5 keep
            # stepping past that (the recorded trajectory is what it is). We
            # match: read success across the whole roll-out, don't truncate.
    finally:
        env.close()

    assert succeeded, (
        f"Episode {episode_index}: derived expert actions failed to solve the task "
        f"(total_return={total_return:.2f}). The action labels in "
        f"{entry['actions_path']} do not reproduce the expert trajectory — "
        f"BC trained on these labels cannot recover the task."
    )


@pytest.mark.parametrize("episode_index", [0])
def test_derived_actions_reproduce_qpos(dataset_dir, pickcube_env_factory, episode_index):
    """Replay derived actions and compare final qpos to the dataset's last proprio.

    Even if 'success' is unreachable through derived actions, the final joint
    state should match the dataset (which was recorded by setting the env to
    the expert's stored state). Large divergence here = action derivation is
    inconsistent with the live env's control dynamics.
    """
    manifest = _load_manifest(dataset_dir)
    entry = manifest[episode_index]
    actions = np.load(_resolve(entry["actions_path"], dataset_dir))
    proprio = np.load(_resolve(entry["proprio_path"], dataset_dir))
    half = proprio.shape[1] // 2
    expected_final_qpos = proprio[-1, :half]
    _, ep_meta, env_state_0 = _source_episode(entry)

    env = _build_like_env(entry)
    try:
        _reset_like_build(env, ep_meta, env_state_0)
        # Step ALL actions — the dataset build does the same and doesn't honor
        # terminated/truncated, so we must too if we want proprio[-1] to refer
        # to the same point in the trajectory.
        for action in actions:
            env.step(action.astype(np.float32))
        live_qpos = env.unwrapped.agent.robot.get_qpos()
        if hasattr(live_qpos, "detach"):
            live_qpos = live_qpos.detach().cpu().numpy()
        live_qpos = np.asarray(live_qpos, dtype=np.float32).reshape(-1)
    finally:
        env.close()

    diff = np.abs(live_qpos - expected_final_qpos).max()
    assert diff < 0.05, (
        f"Replayed qpos diverges from dataset proprio by {diff:.4f} rad at the "
        f"final step — derived actions are not faithfully reconstructing the "
        f"expert trajectory. Largest per-joint error position: "
        f"{int(np.argmax(np.abs(live_qpos - expected_final_qpos)))}"
    )
