"""State-based BC dataset reading the per-episode format from src/datasets/builder.py.

Mirrors the data model of the upstream ManiSkill BC baseline (a flat
``state, action`` table over all transitions) but reads our per-episode npy
files via ``manifest.jsonl`` instead of a monolithic h5. State here means the
env's canonical ``obs_mode="state"`` vector that the builder saved verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetStats:
    """Per-dim normalization stats computed from training observations."""

    mean: np.ndarray  # (state_dim,)
    std: np.ndarray   # (state_dim,)


class StateBCDataset(Dataset):
    """Flat ``(state, action)`` pairs concatenated across episodes.

    Drops the last state of each episode (no matching action), matching
    upstream's ``trajectory["obs"][:-1]`` convention.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        device: torch.device,
        num_demos: int | None = None,
        normalize_states: bool = False,
        horizon: int = 1,
        chunk_size: int = 1,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        # Total env-step lookahead the row needs to support: H chunks of N
        # actions each. Goal state is state[t + N*H]; dynamics target is
        # state[t + N] (one chunk forward).
        span = horizon * chunk_size
        dataset_dir = Path(dataset_dir)
        manifest_path = dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest not found at {manifest_path}; build with build_dataset.py"
            )

        records: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if num_demos is not None:
            records = records[:num_demos]
        if not records:
            raise ValueError(f"no episodes in {manifest_path}")

        # For each episode of length T (transitions), valid start positions are
        # 0..T-span. ``actions`` becomes a (n, chunk_size, action_dim) chunk
        # per row; ``next_obs`` is state[t+chunk_size] (one-chunk dynamics
        # target); ``goal_obs`` is state[t+span] (H-chunk actor target).
        obs_list, act_list, next_obs_list, goal_obs_list = [], [], [], []
        for rec in tqdm(records, desc="bc/load", unit="ep"):
            state = np.load(rec["state_path"]).astype(np.float32)
            actions = np.load(rec["actions_path"]).astype(np.float32)
            T = min(state.shape[0] - 1, actions.shape[0])
            if T < span:
                continue
            n = T - span + 1
            # (n, chunk_size, action_dim) via sliding window over actions.
            idx = np.arange(n)[:, None] + np.arange(chunk_size)[None, :]
            obs_list.append(state[:n])
            act_list.append(actions[idx])
            next_obs_list.append(state[chunk_size : n + chunk_size])
            goal_obs_list.append(state[span : n + span])

        if not obs_list:
            raise ValueError(
                f"no episodes long enough for span={span} (horizon={horizon} * "
                f"chunk_size={chunk_size}) in {manifest_path}"
            )
        self.observations = np.vstack(obs_list).astype(np.float32)
        self.actions = np.vstack(act_list).astype(np.float32)
        self.next_observations = np.vstack(next_obs_list).astype(np.float32)
        self.goal_observations = np.vstack(goal_obs_list).astype(np.float32)
        self.horizon = int(horizon)
        self.chunk_size = int(chunk_size)
        assert (
            self.observations.shape[0]
            == self.actions.shape[0]
            == self.next_observations.shape[0]
            == self.goal_observations.shape[0]
        )

        self.state_dim = int(self.observations.shape[1])
        self.action_dim = int(self.actions.shape[-1])
        self.num_episodes = len(records)
        self.device = device

        if normalize_states:
            mean = self.observations.mean(axis=0)
            std = self.observations.std(axis=0)
            std[std < 1e-6] = 1.0  # guard against constant dims
            self.stats = DatasetStats(mean=mean, std=std)
            self.observations = (self.observations - mean) / std
            # Same normalization applied to next_obs (dynamics target) and
            # goal_obs (actor H-step target) so they share the input space.
            self.next_observations = (self.next_observations - mean) / std
            self.goal_observations = (self.goal_observations - mean) / std
        else:
            self.stats = DatasetStats(
                mean=np.zeros(self.state_dim, dtype=np.float32),
                std=np.ones(self.state_dim, dtype=np.float32),
            )

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(self.observations[idx]).float().to(self.device)
        act = torch.from_numpy(self.actions[idx]).float().to(self.device)
        next_obs = torch.from_numpy(self.next_observations[idx]).float().to(self.device)
        goal_obs = torch.from_numpy(self.goal_observations[idx]).float().to(self.device)
        return obs, act, next_obs, goal_obs
