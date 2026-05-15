"""Create an image-only video dataset from PickCube expert trajectories."""

from __future__ import annotations

import argparse

from src.datasets.expert_videos import (
    DEFAULT_PICKCUBE_TRAJECTORY,
    ExpertVideoDatasetConfig,
    build_expert_video_dataset,
)


def parse_args() -> ExpertVideoDatasetConfig:
    defaults = ExpertVideoDatasetConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-path", default=DEFAULT_PICKCUBE_TRAJECTORY)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--env-id", default=defaults.env_id)
    p.add_argument("--sim-backend", default=defaults.sim_backend)
    p.add_argument("--render-backend", default=defaults.render_backend)
    p.add_argument("--camera-uid", default=defaults.camera_uid)
    p.add_argument("--width", type=int, default=defaults.width)
    p.add_argument("--height", type=int, default=defaults.height)
    p.add_argument("--max-episodes", type=int, default=defaults.max_episodes)
    p.add_argument("--frame-stride", type=int, default=defaults.frame_stride)
    p.add_argument("--fps", type=int, default=defaults.fps)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.width < 1:
        p.error("--width must be >= 1")
    if args.height < 1:
        p.error("--height must be >= 1")
    if args.frame_stride < 1:
        p.error("--frame-stride must be >= 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        p.error("--max-episodes must be >= 1 when set")
    if args.fps < 1:
        p.error("--fps must be >= 1")
    return ExpertVideoDatasetConfig(**vars(args))


def main() -> None:
    records = build_expert_video_dataset(parse_args())
    frames = sum(record.num_frames for record in records)
    print(f"Built expert video dataset: episodes={len(records)} frames={frames}", flush=True)


if __name__ == "__main__":
    main()
