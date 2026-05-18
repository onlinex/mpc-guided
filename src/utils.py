"""Small generic helpers used across the repo."""

from __future__ import annotations

import numpy as np
import torch


class OUNoise:
    """Discretized Ornstein-Uhlenbeck noise (AR(1) on the action vector).

    ``X_{t+1} = (1 - theta) * X_t + theta * mu + sigma * eps``, with ``sigma``
    derived from the requested stationary standard deviation so users can
    parameterize exploration intensity directly.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        theta: float,
        stationary_std: float,
        mu: float = 0.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 < theta <= 1.0:
            raise ValueError(f"theta must be in (0, 1], got {theta}")
        if stationary_std < 0.0:
            raise ValueError(f"stationary_std must be >= 0, got {stationary_std}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")

        self.action_dim = action_dim
        self.theta = float(theta)
        self.mu = float(mu)
        self.stationary_std = float(stationary_std)
        self.sigma = float(stationary_std * np.sqrt(theta * (2.0 - theta)))
        self.rng = np.random.default_rng(seed)
        self.x = np.full(action_dim, self.mu, dtype=np.float32)

    def reset(self) -> None:
        self.x[:] = self.mu

    def sample(self) -> np.ndarray:
        eps = self.rng.standard_normal(self.action_dim).astype(np.float32)
        self.x = (1.0 - self.theta) * self.x + self.theta * self.mu + self.sigma * eps
        return self.x.copy()


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
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
