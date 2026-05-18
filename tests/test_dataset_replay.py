"""Verify the built dataset reproduces its source under a fresh env replay.

The builder steps the env, captures ``state`` from ``obs_mode="state"`` and
saves it. Faithfulness should follow by construction: stepping a fresh env
with the same actions from the same initial env_state must produce the same
state vector at every step. This test exists to catch any future change that
breaks that property (env config drift, action stream truncation, etc).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import io_utils

from src.datasets.builder import BuildConfig, _make_env, _obs_to_state


pytestmark = [pytest.mark.env, pytest.mark.dataset]


def _load_manifest(dataset_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (dataset_dir / "manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("episode_index", [0])
def test_replay_reproduces_saved_state(episode_index):
    dataset_dir = Path("data/pickcube_rl")
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        pytest.skip(f"{dataset_dir} not built; run build_dataset.py first")

    entry = _load_manifest(dataset_dir)[episode_index]
    actions = np.load(entry["actions_path"])
    saved_state = np.load(entry["state_path"])
    h5_path = Path(entry["source_h5"]).expanduser()
    source_metadata = io_utils.load_json(str(h5_path.with_suffix(".json")))
    ep_meta = next(
        e for e in source_metadata["episodes"]
        if int(e["episode_id"]) == int(entry["episode_id"])
    )
    with h5py.File(h5_path, "r") as h5:
        env_states = trajectory_utils.dict_to_list_of_dicts(
            h5[f"traj_{entry['episode_id']}"]["env_states"]
        )

    cfg = BuildConfig(
        source_h5=str(h5_path),
        output_dir="(unused)",
        control_mode=entry["control_mode"],
    )
    env = _make_env(source_metadata, cfg, obs_mode="state")
    try:
        env.reset(**ep_meta.get("reset_kwargs", {}))
        env.unwrapped.set_state_dict(env_states[0])
        replayed = [_obs_to_state(env.unwrapped.get_obs())]
        for action in actions:
            obs, _r, _t, _tr, _info = env.step(action.astype(np.float32))
            replayed.append(_obs_to_state(obs))
    finally:
        env.close()

    replayed_arr = np.stack(replayed, axis=0)
    assert replayed_arr.shape == saved_state.shape
    max_diff = float(np.abs(replayed_arr - saved_state).max())
    assert max_diff < 1e-4, (
        f"replay diverged from saved state by max {max_diff:.6f}; "
        f"builder is not deterministic"
    )
