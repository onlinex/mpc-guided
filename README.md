# mpc-control

Goal-state behavior cloning with learned forward dynamics on ManiSkill PickCube.

The actor is trained by rolling forward through a learned `ForwardDynamicsModel`
for K steps and matching the predicted state at step K to a goal frame sampled
from expert demonstrations.

## Local Smoke Rollout

```bash
uv sync
uv run python demo.py --episodes 1 --max-steps 50
```

Play back a trained checkpoint:

```bash
uv run python demo.py --checkpoint runs/dynamics/<RUN>/actor.pt --episodes 3
```

`--checkpoint` forces `--obs-mode rgb`, builds the R3M backbone, and runs the
actor's deterministic mean. Pass `--stochastic` to add AR(1)/OU exploration
noise on top.

Note: `demo.py` currently assumes an R3M-trained checkpoint. Loading a
`--use-privileged-state` checkpoint via `demo.py` is not yet wired.

## Expert Video Dataset

Source demos must already be in the target control mode. PickCube
motionplanning demos ship in `pd_joint_pos` — convert to `pd_joint_delta_pos`
first:

```bash
uv run python -m mani_skill.utils.download_demo PickCube-v1
uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --target-control-mode pd_joint_delta_pos --save-traj --use-env-states
```

Then build our derived dataset:

```bash
uv run python build_pickcube_video_dataset.py
```

This replays each episode by stepping the env with the H5's stored actions
(matching control mode is validated at build time) and writes per-episode:

```text
data/pickcube_expert_videos/
  videos/episode_NNNNNN.mp4        # base_camera RGB
  actions/episode_NNNNNN.npy       # (T, action_dim)  pd_joint_delta_pos
  proprio/episode_NNNNNN.npy       # (T+1, 18) qpos + qvel
  state/episode_NNNNNN.npy         # (T+1, 29) privileged state (agent + extra)
  manifest.jsonl
  metadata.json
```

Because we step actions in the same env an actor will be evaluated in, the
dataset is provably faithful — verified by `tests/test_expert_action_replay.py`.

## Training

```bash
uv run python train_dynamics.py --run-name my-run
```

Per round, the trainer:

1. **Collect** episodes in the env using the current actor (+ optional OU noise).
2. **Dynamics phase** — `--train-steps-per-round` gradient steps against the buffer.
3. **Actor phase** — `--actor-train-steps-per-round` updates against the
   just-trained dynamics. Each step samples (start, goal) expert frame pairs,
   rolls actor through dynamics for gap K, and minimizes MSE between predicted
   visual at step K and the goal visual.
4. **Checkpoint** `actor_latest.pt`.

### Privileged-state mode

The `--use-privileged-state` flag bypasses the R3M backbone entirely and feeds
the simulator's flat `agent + extra` state vector as the "visual" stream. Used
for diagnostic experiments to isolate dynamics-learning issues from visual-
encoding issues. Requires `state.npy` in the dataset (already saved by the
build script above).

```bash
uv run python train_dynamics.py --use-privileged-state
```

### BC mode for the actor

`--actor-bc-mode` switches the actor training step from goal-state-via-dynamics
to direct behavior cloning: MSE between `actor(state, proprio)` and the H5
action label, sampled from the replay buffer. Dynamics is irrelevant in this
mode and can be skipped with `--pretrain-dynamics-steps 0 --train-steps-per-round 0`.

Optional `--actor-bc-input-noise <std>` adds Gaussian noise to inputs during
BC training (DART-style augmentation against closed-loop covariate shift).

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--hidden-dims` | `512,512` | Dynamics MLP hidden sizes |
| `--actor-hidden-dims` | `256,256` | Actor MLP hidden sizes |
| `--buffer-capacity` | `200000` | On-policy slots; expert transitions are added on top and pinned |
| `--pretrain-dynamics-steps` | `10000` | One-shot dynamics pretrain on seeded expert before collection rounds start |
| `--train-steps-per-round` | `50` | In-loop dynamics steps per round; `0` freezes dynamics after pretrain |
| `--actor-train-steps-per-round` | `30` | Actor steps per round |
| `--actor-pair-min-gap` / `--actor-pair-max-gap` | `1` / `5` | Rollout horizon range (goal-state actor) |
| `--exploration-ou-std` | `0.0` | AR(1)/OU noise std added during env collection; `0` = deterministic |
| `--use-privileged-state` | off | Skip R3M; use simulator state as visual stream |
| `--actor-bc-mode` | off | Direct BC instead of dynamics rollout for the actor |
| `--seed-buffer-with-expert` | on | Encode every expert transition into the buffer at startup |
| `--pin-expert-in-buffer` | on | Pinned entries are never evicted |

### TensorBoard

```bash
uv run tensorboard --logdir runs
```

Key groups:

- `dynamics/*` — train/eval losses, `visual_identity_loss_ratio` (model loss /
  identity-baseline loss; <1 means it beats predicting "no change"), plus
  per-source breakdown: `visual_identity_loss_ratio_expert` and
  `..._on_policy` (lets you tell whether dynamics is forgetting the expert
  region vs failing to fit the on-policy region).
- `actor/*` — actor loss (goal-state MSE in rollout mode, action MSE in BC
  mode), grad norm, action magnitude.
- `rollout/*` — env return, success, action magnitude, exploration noise mean,
  `action_within_episode_std` (low = near-open-loop).
- `replay_buffer/*` — size, pinned count, expert seeded count.

## Tests

```bash
uv run pytest -m "not env"                              # fast unit tests, ~2s
uv run pytest                                           # full suite incl. env-dependent
uv run pytest tests/test_expert_action_replay.py        # data pipeline integrity
```

Markers:
- `env` — requires a live ManiSkill env (slow).
- `dataset` — requires the built expert video dataset on disk (auto-skipped otherwise).

The `expert_action_replay` test re-steps each episode's actions through a
freshly built env and asserts the recorded success and qpos trajectory are
reproduced exactly. If this test fails, the dataset is no longer trustworthy
and downstream training will be broken in ways that are hard to debug.

## dstack

```bash
uv run dstack secret set wandb_api_key '<your-wandb-key>'   # one-time
uv run dstack apply -f dstack.yml                           # submit run
```
