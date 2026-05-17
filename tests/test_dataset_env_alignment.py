"""Verify the dataset's saved per-frame arrays line up with the live env.

If privileged-state dim or proprio dim disagrees between dataset and env, every
training run silently uses mismatched feature spaces. If the SAVED first-frame
state doesn't match the ENV's computation on the same set_state, the flatten
order or field selection drifted between writer and reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = [pytest.mark.env, pytest.mark.dataset]


def _load_first_manifest_entry(dataset_dir: Path) -> dict:
    with (dataset_dir / "manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError("empty manifest")


def _resolve(path_str: str, dataset_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    return dataset_dir / p


def test_env_proprio_dim_matches_dataset(dataset_dir, pickcube_env_factory):
    from src.observations import proprio_dim_of

    entry = _load_first_manifest_entry(dataset_dir)
    env = pickcube_env_factory(control_mode=entry.get("control_mode", "pd_joint_delta_pos"))
    try:
        dataset_dim = int(entry["proprio_dim"])
        env_dim = proprio_dim_of(env)
    finally:
        env.close()
    assert dataset_dim == env_dim, (
        f"proprio_dim mismatch: dataset={dataset_dim}, env={env_dim}"
    )


def test_env_privileged_state_dim_matches_dataset(dataset_dir, pickcube_env_factory):
    from src.observations import privileged_state_dim_of

    entry = _load_first_manifest_entry(dataset_dir)
    if not entry.get("state_path"):
        pytest.skip("dataset has no state_path — rebuild it to add privileged state")

    env = pickcube_env_factory(control_mode=entry.get("control_mode", "pd_joint_delta_pos"))
    try:
        dataset_dim = int(entry["state_dim"])
        env_dim = privileged_state_dim_of(env)
    finally:
        env.close()
    assert dataset_dim == env_dim, (
        f"state_dim mismatch: dataset={dataset_dim}, env={env_dim}; "
        "this means dataset and live env disagree on what privileged state IS"
    )


def test_dataset_state_matches_env_extraction(dataset_dir, pickcube_env_factory):
    """The saved state[0] for an episode must match what the env yields after
    set_state to that same env_state. Catches flatten-order regressions."""
    import h5py
    from mani_skill.trajectory import utils as trajectory_utils

    from src.observations import extract_privileged_state

    entry = _load_first_manifest_entry(dataset_dir)
    if not entry.get("state_path"):
        pytest.skip("dataset has no state_path")
    saved_state = np.load(_resolve(entry["state_path"], dataset_dir))[0]

    h5_path = Path(entry["source_trajectory"]).expanduser()
    if not h5_path.exists():
        pytest.skip(f"source trajectory not available: {h5_path}")

    env = pickcube_env_factory(control_mode=entry.get("control_mode", "pd_joint_delta_pos"))
    try:
        env.reset(seed=entry.get("episode_seed"))
        with h5py.File(h5_path, "r") as h5:
            traj_key = f"traj_{entry['episode_id']}"
            env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_key]["env_states"])
        env.unwrapped.set_state_dict(env_states[0])
        live_state = extract_privileged_state(env.unwrapped.get_obs())
    finally:
        env.close()

    assert live_state.shape == saved_state.shape, (
        f"state shape mismatch: saved={saved_state.shape}, env={live_state.shape}"
    )
    diff = np.abs(live_state - saved_state).max()
    assert diff < 1e-4, (
        f"saved state[0] differs from live extraction by {diff:.6f}; "
        "flatten order or field selection diverged between dataset build and "
        "current extract_privileged_state"
    )
