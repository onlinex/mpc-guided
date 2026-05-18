"""Static file-level validation of a built dataset.

Doesn't step the env (that's ``test_dataset_replay.py``). Just opens every
manifest entry on disk and checks:

  * manifest + metadata schemas line up with what the builder writes,
  * per-episode shapes/dtypes match the dims declared in the manifest,
  * dims are consistent across all episodes,
  * videos (when present) have real content — not all-green, not stuck.

Designed to be fast (mmaps npy headers; only reads a sampled subset of videos)
and to catch the failure modes we've actually hit: silent corruption of a
subset of episodes, schema drift, all-one-color frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


pytestmark = [pytest.mark.dataset]

DATASET_DIR = Path("data/pickcube_rl")
VIDEO_SAMPLE_INDICES = [0, 1, 100, 500, 996]
MANIFEST_REQUIRED = {
    "episode_id", "state_path", "actions_path", "proprio_path",
    "num_actions", "state_dim", "proprio_dim",
    "control_mode", "source_h5",
}
METADATA_REQUIRED = {
    "config", "source_env_info", "num_episodes", "num_actions",
    "state_dim", "proprio_dim", "has_video",
}


def _skip_if_missing() -> None:
    if not (DATASET_DIR / "manifest.jsonl").exists():
        pytest.skip(f"{DATASET_DIR} not built; run build_dataset.py first")


def _load_manifest() -> list[dict]:
    with (DATASET_DIR / "manifest.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_metadata() -> dict:
    return json.loads((DATASET_DIR / "metadata.json").read_text())


# ---- schema ----


def test_manifest_schema():
    _skip_if_missing()
    manifest = _load_manifest()
    assert manifest, "manifest is empty"
    for i, row in enumerate(manifest):
        missing = MANIFEST_REQUIRED - row.keys()
        assert not missing, f"episode {i}: manifest missing keys {missing}"
        assert "video_path" in row, f"episode {i}: video_path key absent (may be null)"
        assert isinstance(row["episode_id"], int)
        assert isinstance(row["num_actions"], int) and row["num_actions"] > 0
        assert isinstance(row["state_dim"], int) and row["state_dim"] > 0
        assert isinstance(row["proprio_dim"], int) and row["proprio_dim"] > 0


def test_metadata_schema():
    _skip_if_missing()
    meta = _load_metadata()
    missing = METADATA_REQUIRED - meta.keys()
    assert not missing, f"metadata.json missing keys {missing}"
    assert meta["num_episodes"] == len(_load_manifest())
    assert meta["state_dim"] > 0 and meta["proprio_dim"] > 0
    assert isinstance(meta["has_video"], bool)


# ---- dim consistency + per-episode shape ----


def test_dims_consistent_across_episodes():
    _skip_if_missing()
    manifest = _load_manifest()
    state_dim = manifest[0]["state_dim"]
    proprio_dim = manifest[0]["proprio_dim"]
    for row in manifest:
        assert row["state_dim"] == state_dim, f"ep {row['episode_id']} has state_dim={row['state_dim']}"
        assert row["proprio_dim"] == proprio_dim, f"ep {row['episode_id']} has proprio_dim={row['proprio_dim']}"


def test_per_episode_shapes_and_dtypes():
    """Every episode's npy files match the manifest's declared dims and are finite."""
    _skip_if_missing()
    manifest = _load_manifest()
    action_dim_seen: int | None = None
    for row in manifest:
        T = row["num_actions"]
        state = np.load(row["state_path"], mmap_mode="r")
        actions = np.load(row["actions_path"], mmap_mode="r")
        proprio = np.load(row["proprio_path"], mmap_mode="r")
        ep_id = row["episode_id"]

        assert state.shape == (T + 1, row["state_dim"]), \
            f"ep {ep_id}: state shape {state.shape}, expected {(T + 1, row['state_dim'])}"
        assert proprio.shape == (T + 1, row["proprio_dim"]), \
            f"ep {ep_id}: proprio shape {proprio.shape}, expected {(T + 1, row['proprio_dim'])}"
        assert actions.shape[0] == T, \
            f"ep {ep_id}: actions length {actions.shape[0]}, expected {T}"

        if action_dim_seen is None:
            action_dim_seen = int(actions.shape[1])
        else:
            assert actions.shape[1] == action_dim_seen, \
                f"ep {ep_id}: action_dim={actions.shape[1]} differs from {action_dim_seen}"

        assert state.dtype == np.float32, f"ep {ep_id}: state dtype {state.dtype}"
        assert actions.dtype == np.float32, f"ep {ep_id}: actions dtype {actions.dtype}"
        assert proprio.dtype == np.float32, f"ep {ep_id}: proprio dtype {proprio.dtype}"

    # Spot-check finiteness on the first three episodes (cheap, mmap is fast).
    for row in manifest[:3]:
        for key in ("state_path", "actions_path", "proprio_path"):
            arr = np.asarray(np.load(row[key]))
            assert np.all(np.isfinite(arr)), f"ep {row['episode_id']}: non-finite in {key}"


# ---- video sanity ----


@pytest.mark.parametrize("episode_index", VIDEO_SAMPLE_INDICES)
def test_video_has_real_content(episode_index):
    """Catch all-green / stuck / wrong-shape videos across the dataset, not just ep 0.

    Three independent checks:
      1. Per-channel mean balance — catches uniform-color tints (the green bug).
      2. Cross-pixel std — catches solid-color frames.
      3. Cross-frame std — catches frames stuck on a single image.
    """
    _skip_if_missing()
    manifest = _load_manifest()
    if episode_index >= len(manifest):
        pytest.skip(f"only {len(manifest)} episodes")
    row = manifest[episode_index]
    if row.get("video_path") is None:
        pytest.skip("dataset built without --include-video")

    video = iio.imread(row["video_path"])
    assert video.ndim == 4 and video.shape[-1] == 3, \
        f"ep {row['episode_id']}: unexpected video shape {video.shape}"
    assert video.shape[0] == row["num_actions"] + 1, \
        f"ep {row['episode_id']}: {video.shape[0]} frames, expected {row['num_actions'] + 1}"

    means = video.mean(axis=(0, 1, 2))
    channel_imbalance = float(max(means)) / max(float(min(means)), 1.0)
    assert channel_imbalance < 5.0, (
        f"ep {row['episode_id']}: channel mean imbalance {channel_imbalance:.2f} "
        f"(per-channel means R/G/B={tuple(round(float(x)) for x in means)}); "
        "video probably has a uniform color tint"
    )

    overall_std = float(video.std())
    assert overall_std > 10.0, \
        f"ep {row['episode_id']}: video too uniform (std={overall_std:.2f})"

    cross_frame_std = float(video.std(axis=0).mean())
    assert cross_frame_std > 1.0, \
        f"ep {row['episode_id']}: frames don't vary (per-pixel cross-frame std={cross_frame_std:.2f})"
