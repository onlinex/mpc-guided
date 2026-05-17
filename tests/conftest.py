"""Shared pytest fixtures for the mpc-guided test suite.

Markers (declared in pyproject.toml):
    env      — test creates a ManiSkill env (slow, several seconds setup).
    dataset  — test reads files from data/pickcube_expert_videos; auto-skipped
               when the dataset hasn't been built.

Run subsets:
    uv run pytest -m "not env"                  # fast unit tests only
    uv run pytest tests/test_expert_action_replay.py  # the diagnostic
"""

from __future__ import annotations

from pathlib import Path

import pytest


DEFAULT_DATASET_DIR = Path("data/pickcube_expert_videos")


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    """Path to the built expert video dataset. Skips the test if absent."""
    manifest = DEFAULT_DATASET_DIR / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip(
            f"expert video dataset not found at {DEFAULT_DATASET_DIR}; "
            "build it with `uv run python build_pickcube_video_dataset.py`"
        )
    return DEFAULT_DATASET_DIR


@pytest.fixture(scope="session")
def pickcube_env_factory():
    """Returns a callable that builds a fresh PickCube env in rgb mode.

    Each call returns a new env — caller owns close(). Session-scoped only so
    the import-time cost (`mani_skill.envs`) is paid once.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    def _make(control_mode: str = "pd_joint_delta_pos", obs_mode: str = "rgb"):
        return gym.make(
            "PickCube-v1",
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend="physx_cpu",
            render_backend="gpu",
            sensor_configs={"base_camera": {"width": 64, "height": 64}},
        )

    return _make
