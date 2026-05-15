# mpc-control

TD-MPC2 experiments on ManiSkill.

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

## Visual Dynamics Model

The reusable forward dynamics components live in `src/dynamics/`.
`ForwardDynamicsModel` consumes an R3M visual feature state `(B, 512)` and a
robot action `(B, action_dim)`, and returns the predicted next visual feature
state `(B, 512)`. Rollout actions are sampled by the squashed Gaussian
`StochasticActor` in `src/actor/`.

Train it from ManiSkill interaction sampled by the stochastic actor:

```bash
uv run python train_dynamics.py --run-name pickcube-actor-v1
```

By default this collects 16 initial actor-sampled episodes, then runs 10 rounds
of 1000 gradient steps with 8 more actor-sampled episodes collected after each
round. The trainer uses R3M `resnet18` by default. Each invocation writes to a
timestamped directory under `runs/dynamics`, for example
`runs/dynamics/20260515-142233`, and saves `dynamics_model.pt` inside that run
directory. Intermediate checkpoints overwrite `dynamics_model_latest.pt` every
1000 training steps by default.

During dynamics training, the actor is also trained from the expert video
dataset every 250 dynamics steps by default. That loop samples a start and goal
video frame one timestep apart, encodes both with R3M, rolls the actor through
the current learned dynamics for 1 step, and optimizes actor parameters against
the goal latent.

Useful actor-video flags:

```bash
uv run python train_dynamics.py \
  --actor-video-dataset-dir data/pickcube_expert_videos \
  --actor-train-interval 250 \
  --actor-train-steps 10 \
  --actor-rollout-horizon 1 \
  --actor-pair-min-gap 1 \
  --actor-pair-max-gap 1
```

Disable actor-video training:

```bash
uv run python train_dynamics.py --no-actor-video-training
```

Change checkpoint frequency:

```bash
uv run python train_dynamics.py --checkpoint-interval 500
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
  --checkpoint-interval 25 \
  --actor-train-interval 25 \
  --actor-train-steps 1
```

View logs:

```bash
uv run tensorboard --logdir runs
```

TensorBoard groups:

- `dynamics/*`: forward dynamics train/eval losses and gradient norm.
- `actor/*`: video-goal actor loss, goal cosine, policy std, and gradient norm.
- `rollout/*`: environment rollout returns, success, episode length, and sampled policy action scale.
- `replay_buffer/*`: replay buffer size.
- `progress/*`: coarse run progress markers.

## Expert Video Dataset

Build an image-only dataset from official PickCube expert trajectories:

```bash
uv run python build_pickcube_video_dataset.py \
  --max-episodes 10 \
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
