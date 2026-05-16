# mpc-control

Goal-state behavior cloning with learned visual dynamics on ManiSkill.

This repository is set up with the shared ManiSkill infrastructure from
`ra-control`, without the SAC / FPO / METRA / RPO algorithm packages.

## Local Smoke Rollout

```bash
uv sync
uv run python demo.py --episodes 1 --max-steps 50
```

Headless state rollout:

```bash
uv run python demo.py --render-mode none --render-backend none --episodes 2 --max-steps 50
```

RGB observation rollout:

```bash
uv run python demo.py --obs-mode rgb --render-mode rgb_array --episodes 1 --max-steps 50
```

Play back a trained actor checkpoint in sim:

```bash
uv run python demo.py --checkpoint runs/dynamics/<RUN>/actor.pt --episodes 3 --max-steps 100
```

`--checkpoint` forces `--obs-mode rgb`, builds the R3M backbone, and uses the
actor's deterministic mean action by default. Pass `--stochastic` to add
AR(1)/OU exploration noise on top of the mean (matches how training collection
drives the env). Tune via `--exploration-ou-theta` and `--exploration-ou-std`.

## Visual Dynamics Model and Goal-State BC Actor

The reusable forward dynamics components live in `src/dynamics/`.
`ForwardDynamicsModel` consumes an R3M visual feature state `(B, 512)` and a
robot action `(B, action_dim)`, and returns the predicted next visual feature
state `(B, 512)`. Rollout actions are sampled by the squashed Gaussian
`StochasticActor` in `src/actor/`.

`train_dynamics.py` jointly trains both, alternating per round between:

1. **Collect** episodes in ManiSkill with the current actor's deterministic
   mean action plus AR(1)/OU exploration noise (committed correlated noise,
   not per-step jitter).
2. **Dynamics phase** — `--train-steps-per-round` updates against the buffer.
3. **Actor phase** — `--actor-train-steps-per-round` BC-on-goal updates against
   the just-trained dynamics. Each step samples (start, goal) frame pairs from
   the expert video dataset (gap K sampled per pair from
   `[--actor-pair-min-gap, --actor-pair-max-gap]`), encodes both with R3M, rolls
   the actor's deterministic mean action through frozen dynamics for K steps,
   and minimizes MSE between the predicted state at step K and the goal latent.
4. **Checkpoint** — saves `actor_latest.pt` at the end of every round; `actor.pt`
   is written once at the end of the run.

```bash
uv run python train_dynamics.py --run-name pickcube-actor-v1
```

By default this runs 40 rounds with 32 collected episodes, 50 dynamics steps,
and 30 actor steps each. The trainer uses R3M `resnet18` by default. Each
invocation writes to a timestamped directory under `runs/dynamics`, e.g.
`runs/dynamics/20260515-142233/actor.pt`.

Useful actor-video flags:

```bash
uv run python train_dynamics.py \
  --actor-video-dataset-dir data/pickcube_expert_videos \
  --actor-train-steps-per-round 30 \
  --actor-pair-min-gap 1 \
  --actor-pair-max-gap 5 \
  --actor-pairs-per-video 8
```

The actor's rollout horizon is set per-sample to the gap K of each (start,
goal) pair — there is no separate horizon flag. `--actor-pairs-per-video`
controls how many (start, goal) pairs are drawn from each loaded video before
moving to the next — higher values reduce MP4 decode and cache pressure during
actor training.

Exploration noise (env collection only — actor training rollouts are
deterministic):

```bash
uv run python train_dynamics.py \
  --exploration-ou-theta 0.15 \
  --exploration-ou-std 0.5
```

`--exploration-ou-std 0` disables exploration entirely and collects with the
deterministic actor mean.

Disable actor-video training:

```bash
uv run python train_dynamics.py --no-actor-video-training
```

R3M precision (frozen backbone, so lossless to drop precision; ~2× faster on
CUDA):

```bash
uv run python train_dynamics.py --backbone-precision bf16
```

For a quick smoke run:

```bash
uv run python train_dynamics.py \
  --run-name smoke \
  --initial-episodes 2 \
  --collection-rounds 1 \
  --episodes-per-round 1 \
  --max-steps 25 \
  --train-steps-per-round 25 \
  --batch-size 32 \
  --actor-train-steps-per-round 5
```

View logs:

```bash
uv run tensorboard --logdir runs
```

TensorBoard groups:

- `dynamics/*`: forward dynamics train/eval losses, gradient norm, and
  `identity_loss_ratio` — `model_loss / identity_baseline_loss` on the eval
  batch. ≈1 means the model collapsed to predicting "no change"; <1 means it
  beats identity by that fraction.
- `actor/*`: video-goal actor loss, goal cosine, mean sampled frame gap, and
  gradient norm.
- `rollout/*`: environment rollout returns, success, episode length, mean
  action magnitude, exploration-noise magnitude, and per-episode action std
  across timesteps (`action_within_episode_std` — low values indicate a
  near-open-loop policy).
- `replay_buffer/*`: replay buffer size.
- `progress/*`: coarse run progress markers.

## Expert Video Dataset

Build an image-only dataset from official PickCube expert trajectories:

```bash
uv run python build_pickcube_video_dataset.py \
  --max-episodes 50 \
  --output-dir data/pickcube_expert_videos
```

This replays expert environment states, renders `base_camera` RGB frames, and
writes MP4 videos plus a manifest:

```text
data/pickcube_expert_videos/
  videos/episode_000000.mp4
  manifest.jsonl
  metadata.json
```

If the PickCube demos are missing:

```bash
uv run python -m mani_skill.utils.download_demo PickCube-v1
```

## dstack

One-time W&B secret setup, if future training code uses W&B:

```bash
uv run dstack secret set wandb_api_key '<your-wandb-key>'
```

Submit the remote dynamics smoke job:

```bash
uv run dstack apply -f dstack.yml
```
