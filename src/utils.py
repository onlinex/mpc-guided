"""Small generic helpers used across the repo."""

from __future__ import annotations

import numpy as np
import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def to_scalar_bool(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.any().item())
    return bool(np.asarray(x).any())
