"""Sampling frame pairs from video datasets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


@dataclass(frozen=True)
class VideoPairBatch:
    """A batch of start and goal RGB frames."""

    start_rgb: torch.Tensor
    goal_rgb: torch.Tensor
    frame_gaps: torch.Tensor


@dataclass(frozen=True)
class VideoRecord:
    video_path: Path
    num_frames: int


class VideoFramePairSampler:
    """Samples ordered frame pairs from MP4 videos listed in a manifest."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        min_gap: int,
        max_gap: int,
        cache_size: int = 8,
        pairs_per_video: int = 1,
        seed: int = 0,
    ) -> None:
        if min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {min_gap}")
        if max_gap < min_gap:
            raise ValueError(f"max_gap must be >= min_gap, got {max_gap} < {min_gap}")
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        if pairs_per_video < 1:
            raise ValueError(f"pairs_per_video must be >= 1, got {pairs_per_video}")

        self.dataset_dir = Path(dataset_dir)
        self.records = load_video_records(self.dataset_dir)
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.cache_size = cache_size
        self.pairs_per_video = pairs_per_video
        self.rng = np.random.default_rng(seed)
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def sample(self, batch_size: int, device: torch.device) -> VideoPairBatch:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        starts, goals, gaps = [], [], []
        remaining = batch_size
        while remaining > 0:
            record = self.records[int(self.rng.integers(0, len(self.records)))]
            video = self._load_video(record.video_path)
            num_pairs = min(self.pairs_per_video, remaining)
            for _ in range(num_pairs):
                start_idx, goal_idx = self._sample_indices(len(video))
                starts.append(video[start_idx])
                goals.append(video[goal_idx])
                gaps.append(goal_idx - start_idx)
            remaining -= num_pairs

        return VideoPairBatch(
            start_rgb=torch.as_tensor(np.stack(starts), dtype=torch.uint8, device=device),
            goal_rgb=torch.as_tensor(np.stack(goals), dtype=torch.uint8, device=device),
            frame_gaps=torch.as_tensor(gaps, dtype=torch.float32, device=device),
        )

    def _sample_indices(self, num_frames: int) -> tuple[int, int]:
        if num_frames <= self.min_gap:
            raise ValueError(
                f"video has {num_frames} frames, but min_gap={self.min_gap} requires more"
            )
        max_gap = min(self.max_gap, num_frames - 1)
        gap = int(self.rng.integers(self.min_gap, max_gap + 1))
        start_idx = int(self.rng.integers(0, num_frames - gap))
        return start_idx, start_idx + gap

    def _load_video(self, path: Path) -> np.ndarray:
        if path in self._cache:
            video = self._cache.pop(path)
            self._cache[path] = video
            return video

        video = np.asarray(iio.imread(path))
        if video.ndim != 4 or video.shape[-1] < 3:
            raise ValueError(f"expected video shape (T, H, W, 3+), got {video.shape} for {path}")
        video = video[..., :3]
        if video.dtype != np.uint8:
            video = to_uint8(video)

        self._cache[path] = video
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return video


def load_video_records(dataset_dir: Path) -> list[VideoRecord]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Video dataset manifest does not exist: {manifest_path}. "
            "Build it with build_pickcube_video_dataset.py first."
        )

    records: list[VideoRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            video_path = Path(item["video_path"])
            if not video_path.is_absolute():
                video_path = video_path if video_path.exists() else dataset_dir / video_path
            if not video_path.exists():
                raise FileNotFoundError(f"video listed in manifest does not exist: {video_path}")
            records.append(VideoRecord(video_path=video_path, num_frames=int(item["num_frames"])))
    if not records:
        raise ValueError(f"no videos found in {manifest_path}")
    return records


def to_uint8(video: np.ndarray) -> np.ndarray:
    video_float = video.astype(np.float32)
    if video_float.size > 0 and video_float.max() <= 1.0:
        video_float *= 255.0
    return np.clip(video_float, 0.0, 255.0).astype(np.uint8)
