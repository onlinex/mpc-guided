"""Helpers for extracting and encoding ManiSkill RGB observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.backbone import encode_images


def extract_rgb(obs: Any, camera_uid: str | None = None) -> np.ndarray:
    """Return a single ``(H, W, 3)`` uint8 RGB frame from a ManiSkill observation.

    When ``camera_uid`` is given, reads ``obs["sensor_data"][camera_uid]["rgb"]``.
    Otherwise descends the obs dict recursively looking for the first ``rgb``
    key. Strips alpha channel if present and converts torch tensors / float
    images to uint8.
    """
    if camera_uid is not None:
        try:
            rgb = obs["sensor_data"][camera_uid]["rgb"]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"camera_uid={camera_uid!r} not found in observation") from exc
    else:
        rgb = _find_rgb(obs)
        if rgb is None:
            raise KeyError("could not find an 'rgb' image in the ManiSkill observation")
    return _to_uint8_frame(rgb)


def encode_observation(
    backbone: nn.Module,
    obs: Any,
    device: torch.device,
    camera_uid: str | None = None,
) -> np.ndarray:
    """Encode a single observation's RGB frame into ``(state_dim,)`` R3M features."""
    rgb = extract_rgb(obs, camera_uid)  # (H, W, 3) uint8
    rgb_tensor = torch.as_tensor(rgb).unsqueeze(0)  # (1, H, W, 3)
    features = encode_images(backbone, rgb_tensor, device)
    return features[0].detach().cpu().numpy().astype(np.float32)


def _find_rgb(value: Any) -> Any | None:
    if isinstance(value, dict):
        if "rgb" in value:
            return value["rgb"]
        for child in value.values():
            found = _find_rgb(child)
            if found is not None:
                return found
    return None


def _to_uint8_frame(rgb: Any) -> np.ndarray:
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        if rgb.shape[0] != 1:
            raise ValueError(f"expected single-env RGB batch (B=1), got shape {rgb.shape}")
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError(f"expected RGB rank 3 or 4, got shape {rgb.shape}")
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB last dim to be 3, got shape {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb
    rgb_float = rgb.astype(np.float32)
    if rgb_float.size > 0 and rgb_float.max() <= 1.0:
        rgb_float = rgb_float * 255.0
    return np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
