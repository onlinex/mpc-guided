# mpc-control

State-based behavior cloning on ManiSkill PickCube, with an optional jointly-trained
forward dynamics model that can be mixed into the actor's loss for model-grounded
regularization (or used as the sole training signal).

Two parallel BC entrypoints:

- **[train_bc_baseline.py](train_bc_baseline.py)** — byte-equivalent port of the
  upstream ManiSkill state-based BC baseline (`examples/baselines/bc/bc.py`). Reads
  the monolithic h5 produced by `replay_trajectory`. Use this when you want a
  strict reference against the published number.
- **[train_bc.py](train_bc.py)** — same numerics (256x256 ReLU MLP, single-action
  MSE, Adam lr 3e-4, batch 1024) but reads our per-episode dataset format produced
  by `build_dataset.py`. Adds an independently-trained `ForwardModel` and exposes
  `--actor-loss-weight` / `--total-loss-weight` to ablate direct vs model-based
  actor losses.

## Quick start

```bash
uv sync

# 1. Download RL demos (one-time).
uv run python -m mani_skill.utils.download_demo PickCube-v1

# 2. Convert to state obs + pd_joint_delta_pos control (drops a new h5 next to source).
uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o state \
  --save-traj -b physx_cpu -n 10

# 3a. Run the strict upstream baseline (h5 in, eval rollouts out).
uv run python train_bc_baseline.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --total-iters 50000 --run-name pickcube-bc-baseline

# 3b. OR: build our per-episode dataset, train on that.
uv run python build_dataset.py \
  --source-h5 ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
  --output-dir data/pickcube_rl
uv run python train_bc.py \
  --dataset-dir data/pickcube_rl \
  --total-iters 50000 --run-name pickcube-bc-perepisode
```

State-based BC on RL demos reaches `eval/success_at_end` ≈ 0.8 within 50k iters on
PickCube. The two scripts produce comparable curves.

## Dataset format

`build_dataset.py` accepts any ManiSkill `trajectory.h5` and writes per-episode
files:

```text
data/<tag>/
  state/episode_NNNNNN.npy     # (T+1, state_dim)   env's obs_mode=state vector
  actions/episode_NNNNNN.npy   # (T,   action_dim)  source h5 actions, verbatim
  proprio/episode_NNNNNN.npy   # (T+1, proprio_dim) qpos + qvel
  videos/episode_NNNNNN.mp4    # optional, --include-video
  manifest.jsonl               # one json object per episode (paths + dims)
  metadata.json                # env_id, control_mode, dims, source, has_video
```

The build is two-pass: state/actions/proprio always; rgb video only with
`--include-video` (~2x slower). Source h5's `control_mode` must match `--control-mode`
(default `pd_joint_delta_pos`).

Faithfulness check: [tests/test_dataset_replay.py](tests/test_dataset_replay.py)
replays episode 0 through a fresh env from the saved `env_state` and asserts the
state trajectory matches the saved `state.npy` within 1e-4.

Static validation: [tests/test_dataset_validity.py](tests/test_dataset_validity.py)
checks manifest/metadata schemas, per-episode shapes/dtypes/finiteness across
all episodes, and video content sanity (channel balance, cross-pixel and
cross-frame variance) — catches the all-green or solid-color rendering class
of bugs.

## Actor + ForwardModel

[src/actor/](src/actor/) is the active code path. Two minimal modules:

- `Actor`: `state_dim → 256 → ReLU → 256 → ReLU → action_dim`. Raw linear output,
  no squashing — matches the upstream baseline.
- `ForwardModel`: `(state, action) → 256 → ReLU → 256 → ReLU → state_dim`.

Both are pure MLPs. The forward model is trained alongside the actor on the same
per-episode data with its own optimizer, so by default it has zero influence on
the actor (gradients can't leak across optimizers).

### Joint actor loss

`train_bc.py` computes three losses per iteration:

- `losses/actor_loss`  = `MSE(actor(obs), expert_action)`
- `losses/dynamics_loss` = `MSE(forward(obs, expert_action), next_obs)` (trains forward)
- `losses/total_loss` = `MSE(forward(obs, actor(obs)), next_obs)` (couples actor→forward→state)

The actor is trained on a weighted combination:

```
actor_step_loss = actor_loss_weight * actor_loss + total_loss_weight * total_loss
```

| `--actor-loss-weight` | `--total-loss-weight` | Behavior |
|---|---|---|
| `0.0` (default) | `1.0` (default) | Pure model-based imitation. Actor only sees expert states; learns to produce actions that the (jointly-trained) forward model maps onto expert next states. On PickCube reaches comparable final reward to direct BC, slightly worse actor_loss along the way. |
| `1.0` | `> 0` | Joint: direct BC + model-grounded regularization. Robust on PickCube; on this dataset `1.0`/`5.0` train as well as pure BC. |
| `1.0` | `0.0` | Pure BC. ForwardModel still trains but doesn't influence actor. `total_loss` logged as diagnostic. |

`total_loss - dynamics_loss` is a useful compounding-error diagnostic: with the
actor in-distribution it's small; if the actor drifts off-policy w.r.t. forward
the gap grows.

### Other flags

```text
--dataset-dir          data/pickcube_rl
--total-iters          50000
--batch-size           1024
--lr                   3e-4
--rollout-freq         1000       # train iters between rollout/eval rounds
--num-rollout-episodes 100        # episodes per rollout (feeds online buffer)
--num-eval-episodes    50         # episodes per eval (deterministic metrics)
--max-episode-steps    100
--normalize-states     off        # per-dim zero-mean unit-var input normalization
--seed                 42
```

## Eval / playback

```bash
uv run python play_bc_baseline.py \
  --checkpoint runs/bc/<RUN>/checkpoints/best_eval_success_at_end.pt \
  --episodes 5
```

Opens the SAPIEN viewer (macOS uses MoltenVK). Pass `--no-gui` for headless or
`--video-dir <path>` to save mp4s via `RecordEpisode`. Loads checkpoints from both
`train_bc_baseline.py` and `train_bc.py` (they share the actor format).

## Tests

```bash
uv run pytest                               # active suite (legacy auto-skipped)
uv run pytest -m "not env"                  # fast unit tests, ~5s
uv run pytest -m "dataset"                  # static dataset validation (needs build)
uv run pytest -m "env and dataset"          # slow replay-through-env tests
uv run pytest -m "legacy"                   # parked tests (src/legacy/, legacy/)
```

Markers (declared in [pyproject.toml](pyproject.toml)):

- `env` — constructs a ManiSkill env (seconds of import + reset time).
- `dataset` — reads from `data/pickcube_rl/`; auto-skipped if not built.
- `legacy` — anything under `tests/legacy/`; deselected by default.

## Repo layout

Active:

```text
src/
  actor/         Actor + ForwardModel (the current BC stack)
  bc/            StateBCDataset (loads per-episode dataset)
  buffer/        OnlineBuffer (FIFO ring fed by rollouts)
  datasets/      builder.py for dataset construction
  backbone.py    encode_images (R3M wrapper, used by dataset builder)
  observations.py
train_bc.py             Per-episode-format BC trainer with joint actor+dynamics loss
train_bc_baseline.py    H5-format upstream-equivalent BC trainer
build_dataset.py        Per-episode dataset builder
play_bc.py              Checkpoint replay for train_bc.py
play_bc_baseline.py     Checkpoint replay for train_bc_baseline.py
```

Parked (kept for later, not on the BC path):

```text
src/legacy/             Old code: actor/ (chunked+squashed), dynamics/ (episode store +
                        multi-step trainer), datasets/ (expert_transitions, video_pairs),
                        networks.py, rollout.py, utils.py
legacy/                 Old entry points: train_dynamics.py, demo.py
tests/legacy/           Tests for the above; auto-skipped from default pytest runs
```

The parked code is functional and tested (`uv run pytest -m legacy`) but isn't
imported by `train_bc*.py`. The dynamics path is worth revisiting if/when the BC
baseline is locked down and we want to layer planning or multi-step world-model
training on top.

## dstack

```bash
uv run dstack secret set wandb_api_key '<your-wandb-key>'   # one-time
uv run dstack apply -f dstack.yml                           # submit run
```
