"""Build a per-episode dataset from any ManiSkill trajectory.h5.

Source h5 must already be in the target ``--control-mode``. RL demos ship in
``pd_joint_delta_pos`` and work as-is. Motionplanning demos ship in
``pd_joint_pos`` and need a prior ``replay_trajectory ... -c pd_joint_delta_pos
--save-traj`` conversion before use.

Examples (both use the RL demos so the control mode is already correct):

  # State-only (fast, ~2.5 min for 1k episodes).
  uv run python build_dataset.py \
    --source-h5 ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
    --output-dir data/pickcube_rl

  # State + per-episode mp4s (second pass through env in obs_mode=rgb; ~5 min).
  uv run python build_dataset.py \
    --source-h5 ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
    --output-dir data/pickcube_rl \
    --include-video --overwrite
"""

from __future__ import annotations

import argparse

from src.datasets.builder import BuildConfig, build_dataset


def parse_args() -> BuildConfig:
    d = BuildConfig(source_h5="", output_dir="")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-h5", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--control-mode", default=d.control_mode)
    p.add_argument("--sim-backend", default=d.sim_backend)
    p.add_argument("--render-backend", default=d.render_backend)
    p.add_argument("--max-episodes", type=int, default=d.max_episodes)
    p.add_argument(
        "--include-video",
        action=argparse.BooleanOptionalAction,
        default=d.include_video,
        help="Off by default. On = additional rgb pass through env to save mp4s.",
    )
    p.add_argument("--camera-uid", default=d.camera_uid)
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--height", type=int, default=d.height)
    p.add_argument("--fps", type=int, default=d.fps)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        p.error("--max-episodes must be >= 1 when set")
    return BuildConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    records = build_dataset(cfg)
    total = sum(r.num_actions for r in records)
    print(
        f"Built {cfg.output_dir}: episodes={len(records)} "
        f"actions={total} state_dim={records[0].state_dim} "
        f"video={'yes' if cfg.include_video else 'no'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
