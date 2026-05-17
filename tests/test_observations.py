"""Unit tests for observation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.observations import extract_privileged_state, extract_rgb


def test_privileged_state_flattens_agent_and_extra():
    obs = {
        "agent": {"qpos": np.array([1.0, 2.0]), "qvel": np.array([3.0])},
        "extra": {"tcp_pose": np.array([0.1, 0.2, 0.3])},
        "sensor_data": {"camera": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}},
    }
    state = extract_privileged_state(obs)
    # sorted keys: agent.qpos, agent.qvel, extra.tcp_pose — sensor_data excluded.
    np.testing.assert_array_equal(state, np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32))


def test_privileged_state_is_deterministic_over_key_order():
    """Different dict insertion orders must produce the same flat vector."""
    obs_a = {
        "agent": {"qpos": np.array([1.0]), "qvel": np.array([2.0])},
        "extra": {"a": np.array([10.0]), "b": np.array([20.0])},
    }
    obs_b = {
        "extra": {"b": np.array([20.0]), "a": np.array([10.0])},
        "agent": {"qvel": np.array([2.0]), "qpos": np.array([1.0])},
    }
    np.testing.assert_array_equal(extract_privileged_state(obs_a), extract_privileged_state(obs_b))


def test_privileged_state_accepts_torch_tensors():
    obs = {
        "agent": {"qpos": torch.tensor([1.0, 2.0])},
        "extra": {"x": torch.tensor([3.0])},
    }
    state = extract_privileged_state(obs)
    np.testing.assert_array_equal(state, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_privileged_state_rejects_non_dict():
    with pytest.raises(TypeError):
        extract_privileged_state(np.array([1.0]))


def test_extract_rgb_strips_alpha():
    obs = {"sensor_data": {"cam": {"rgb": np.zeros((8, 8, 4), dtype=np.uint8)}}}
    rgb = extract_rgb(obs, camera_uid="cam")
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
