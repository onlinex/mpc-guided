"""R3M backbone and image preprocessing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from r3m import load_r3m


R3M_INPUT_SIZE = 224
R3M_MODEL_ID = "resnet18"
R3M_FEAT_DIM = 512
R3M_MODEL_IDS = ("resnet18", "resnet34")


def r3m_backbone_name(model_id: str) -> str:
    return f"r3m_{model_id}"


R3M_BACKBONE_NAME = r3m_backbone_name(R3M_MODEL_ID)


def preprocess_rgb(rgb_uint8: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, H, W, 3)`` uint8 RGB to R3M's float ``(B, 3, 224, 224)`` input."""
    x = rgb_uint8.permute(0, 3, 1, 2).to(torch.float32)
    return F.interpolate(x, size=R3M_INPUT_SIZE, mode="bilinear", align_corners=False)


def build_backbone(
    device: torch.device,
    model_id: str = R3M_MODEL_ID,
) -> nn.Module:
    if model_id not in R3M_MODEL_IDS:
        raise ValueError(f"Unsupported R3M model_id={model_id!r}; expected one of {R3M_MODEL_IDS}.")
    backbone = load_r3m(model_id).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def encode_images(backbone: nn.Module, rgb_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Preprocess images and return frozen ``(B, 512)`` R3M features."""
    images = preprocess_rgb(rgb_uint8.to(device))
    with torch.no_grad():
        return backbone(images)


def encode_images_grad(backbone: nn.Module, rgb_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Like ``encode_images`` but keeps the forward pass in autograd."""
    images = preprocess_rgb(rgb_uint8.to(device))
    return backbone(images)
