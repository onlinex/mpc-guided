"""Load (rgb_t, action_t, rgb_{t+1}) transitions from the expert video dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from src.backbone import encode_images
from src.datasets.video_pairs import to_uint8
from src.dynamics.buffer import TransitionReplayBuffer


def _backbone_cache_key(backbone: torch.nn.Module) -> str:
    """Stable cache key derived from R3M model_id and forward dtype."""
    dtype = next(backbone.parameters()).dtype
    dtype_str = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        dtype, str(dtype).replace("torch.", "")
    )
    model_id = getattr(backbone, "name", None) or type(backbone).__name__
    # r3m's loader returns a wrapped module; try a few common attrs for stability.
    for attr in ("model_id", "module_name", "name"):
        value = getattr(backbone, attr, None)
        if isinstance(value, str) and value:
            model_id = value
            break
    return f"{model_id}_{dtype_str}"


@dataclass(frozen=True)
class ExpertTransitionRecord:
    video_path: Path
    actions_path: Path
    proprio_path: Path
    num_actions: int


def count_expert_transitions(dataset_dir: str | Path) -> int:
    """Sum ``num_actions`` across the manifest without loading any media."""
    records = load_transition_records(Path(dataset_dir))
    return sum(int(record.num_actions) for record in records)


def seed_buffer_with_expert(
    *,
    buffer: TransitionReplayBuffer,
    dataset_dir: str | Path,
    backbone: torch.nn.Module,
    device: torch.device,
    encode_batch_size: int = 256,
    use_cache: bool = True,
) -> int:
    """Encode every expert transition once and push it into the replay buffer.

    Per-episode encoded R3M features are cached under
    ``dataset_dir/encoded/{model}_{precision}/episode_NNNNNN.npy`` so subsequent
    runs skip mp4 decode and the R3M forward entirely.

    Returns the number of transitions added. The caller is responsible for
    calling ``buffer.pin_current_contents()`` after this if expert data should
    survive subsequent eviction.
    """
    dataset_dir = Path(dataset_dir)
    records = load_transition_records(dataset_dir)
    total = sum(int(record.num_actions) for record in records)
    if total > buffer.capacity:
        raise ValueError(
            f"expert dataset has {total} transitions but buffer capacity is "
            f"{buffer.capacity}; allocate a larger buffer before seeding"
        )
    cache_dir = dataset_dir / "encoded" / _backbone_cache_key(backbone) if use_cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    cache_hits = 0
    progress = tqdm(records, desc="seed/expert", unit="episode", dynamic_ncols=True, leave=True)
    for record in progress:
        actions = np.asarray(np.load(record.actions_path), dtype=np.float32)
        proprio = np.asarray(np.load(record.proprio_path), dtype=np.float32)
        if proprio.ndim != 2:
            raise ValueError(
                f"expected proprio shape (T, D), got {proprio.shape} for {record.proprio_path}"
            )
        cache_path = (
            cache_dir / f"{record.video_path.stem}.npy" if cache_dir is not None else None
        )
        features = _load_or_encode_features(
            record=record,
            cache_path=cache_path,
            backbone=backbone,
            device=device,
            encode_batch_size=encode_batch_size,
            num_actions=actions.shape[0],
        )
        if cache_path is not None and cache_path.exists():
            cache_hits += 1
        num_pairs = min(features.shape[0] - 1, actions.shape[0], proprio.shape[0] - 1)
        if num_pairs < 1:
            continue
        for i in range(num_pairs):
            buffer.add(
                features[i],
                proprio[i],
                actions[i],
                features[i + 1],
                proprio[i + 1],
            )
            added += 1
        progress.set_postfix(transitions=added, cached=cache_hits)
    return added


def _load_or_encode_features(
    *,
    record: ExpertTransitionRecord,
    cache_path: Path | None,
    backbone: torch.nn.Module,
    device: torch.device,
    encode_batch_size: int,
    num_actions: int,
) -> np.ndarray:
    expected_frames = num_actions + 1
    if cache_path is not None and cache_path.exists():
        features = np.load(cache_path)
        if features.ndim == 2 and features.shape[0] >= expected_frames:
            return features[:expected_frames].astype(np.float32, copy=False)
        # stale cache (e.g., dataset rebuilt) — overwrite below
    video = np.asarray(iio.imread(record.video_path))
    if video.ndim != 4 or video.shape[-1] < 3:
        raise ValueError(f"expected video shape (T, H, W, 3+), got {video.shape}")
    video = video[..., :3]
    if video.dtype != np.uint8:
        video = to_uint8(video)
    take = min(expected_frames, video.shape[0])
    frames_tensor = torch.as_tensor(video[:take], dtype=torch.uint8)
    with torch.no_grad():
        chunks = []
        for start in range(0, frames_tensor.shape[0], encode_batch_size):
            chunk = frames_tensor[start : start + encode_batch_size]
            chunks.append(encode_images(backbone, chunk, device).cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32)
    if cache_path is not None:
        np.save(cache_path, features)
    return features


def load_transition_records(dataset_dir: Path) -> list[ExpertTransitionRecord]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Video dataset manifest does not exist: {manifest_path}. "
            "Build it with build_pickcube_video_dataset.py first."
        )

    records: list[ExpertTransitionRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("actions_path"):
                raise ValueError(
                    f"manifest entry for episode {item.get('episode_id')} has no actions_path; "
                    "rebuild the dataset with save_actions=True"
                )
            if not item.get("proprio_path"):
                raise ValueError(
                    f"manifest entry for episode {item.get('episode_id')} has no proprio_path; "
                    "rebuild the dataset so proprio (qpos+qvel) is saved per frame"
                )
            video_path = _resolve(item["video_path"], dataset_dir)
            actions_path = _resolve(item["actions_path"], dataset_dir)
            proprio_path = _resolve(item["proprio_path"], dataset_dir)
            if not actions_path.exists():
                raise FileNotFoundError(f"actions file missing: {actions_path}")
            if not proprio_path.exists():
                raise FileNotFoundError(f"proprio file missing: {proprio_path}")
            records.append(
                ExpertTransitionRecord(
                    video_path=video_path,
                    actions_path=actions_path,
                    proprio_path=proprio_path,
                    num_actions=int(item["num_actions"]),
                )
            )
    if not records:
        raise ValueError(f"no transitions found in {manifest_path}")
    return records


def _resolve(path_str: str, dataset_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return dataset_dir / path
