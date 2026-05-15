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

Play back a trained actor checkpoint in sim:

```bash
uv run python demo.py --checkpoint runs/dynamics/<RUN>/actor.pt --episodes 3 --max-steps 100
```

`--checkpoint` forces `--obs-mode rgb`, builds the R3M backbone, and uses the
actor's deterministic mean action by default. Pass `--no-deterministic` to
sample stochastic actions instead.

## Visual Dynamics Model and Goal-State BC Actor

The reusable forward dynamics components live in `src/dynamics/`.
`ForwardDynamicsModel` consumes an R3M visual feature state `(B, 512)` and a
robot action `(B, action_dim)`, and returns the predicted next visual feature
state `(B, 512)`. Rollout actions are sampled by the squashed Gaussian
`StochasticActor` in `src/actor/`.

`train_dynamics.py` jointly trains both, alternating per round between:

1. **Collect** episodes in ManiSkill with the current actor.
2. **Dynamics phase** — `--train-steps-per-round` updates against the buffer.
3. **Actor phase** — `--actor-train-steps-per-round` BC-on-goal updates against
   the just-trained dynamics. Each step samples (start, goal) frame pairs from
   the expert video dataset, encodes both with R3M, rolls the actor through
   frozen dynamics for `--actor-rollout-horizon` steps, and minimizes MSE
   between the predicted final latent and the goal latent.
4. **Checkpoint** — saves `actor_latest.pt` at the end of every round; `actor.pt`
   is written once at the end of the run.

```bash
uv run python train_dynamics.py --run-name pickcube-actor-v1
```

By default this runs 40 rounds with 32 collected episodes, 100 dynamics steps,
and 30 actor steps each. The trainer uses R3M `resnet18` by default. Each
invocation writes to a timestamped directory under `runs/dynamics`, e.g.
`runs/dynamics/20260515-142233/actor.pt`.

Useful actor-video flags:

```bash
uv run python train_dynamics.py \
  --actor-video-dataset-dir data/pickcube_expert_videos \
  --actor-train-steps-per-round 30 \
  --actor-rollout-horizon 1 \
  --actor-pair-min-gap 1 \
  --actor-pair-max-gap 1 \
  --actor-pairs-per-video 8
```

`--actor-pairs-per-video` controls how many (start, goal) pairs are drawn from
each loaded video before moving to the next — higher values reduce MP4 decode
and cache pressure during actor training.

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

- `dynamics/*`: forward dynamics train/eval losses and gradient norm.
- `actor/*`: video-goal actor loss, goal cosine, policy std, frame gap, and gradient norm.
- `rollout/*`: environment rollout returns, success, episode length, and sampled policy action scale.
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
