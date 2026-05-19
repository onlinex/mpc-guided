# mpc-control

Model-based imitation on ManiSkill PickCube. The actor is trained **only** through
a jointly-learned forward dynamics model: given expert `(state, next_state)` pairs,
it has to find actions that the forward model maps from one to the other. The
forward model itself is trained mostly on the actor's own on-policy rollouts via
an online buffer, with optional action noise for exploration.

The main entry point is **[train.py](train.py)** (training) + **[play.py](play.py)**
(rollout). A separate **[train_bc_baseline.py](train_bc_baseline.py)** is kept as
a byte-equivalent port of the upstream ManiSkill state-based BC baseline — used
only as a reference point, not the path we're developing.

## Quick start

```bash
uv sync

# 1. Download RL demos (one-time).
uv run python -m mani_skill.utils.download_demo PickCube-v1

# 2. Build the per-episode dataset.
uv run python build_dataset.py \
  --source-h5 ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
  --output-dir data/pickcube_rl

# 3. Train (defaults: pure model-based, 0.95 online mix, no exploration noise).
uv run python train.py \
  --dataset-dir data/pickcube_rl \
  --total-iters 50000 --run-name pickcube-mb
```

Reaches `eval/success_at_end` ≈ 0.8 on PickCube within 50k iters. The
[upstream BC baseline](#baseline-reference) gets a comparable number with
ordinary BC; the interesting bit here is that the actor is solving the task
without ever seeing the expert's actions as a training signal.

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

[src/actor/](src/actor/) defines:

- `Actor`: `state_dim → 256 → ReLU → 256 → ReLU → action_dim`. Raw linear output.
- `ForwardModel`: shared `trunk` (`Linear(state+action, 256) → ReLU`) feeds two
  heads — `state_head` (2-layer MLP → `state_dim`) for next-state prediction,
  and `surprise_head` (2-layer MLP → softplus scalar) that learns to predict
  the state head's own per-sample MSE. The trunk is detached before the
  surprise head, so surprise training never tugs the predictor.

Each has its own optimizer — gradients can't leak across them. The forward model
is the only path through which the actor's loss reaches reality.

### Losses

Per iteration, `train.py` computes:

- `losses/actor_loss` = `MSE(actor(obs), expert_action)` — diagnostic only at the default weights.
- `losses/dynamics_loss` = `MSE(state_head(trunk(obs, action)), next_obs)` — trains trunk + state_head.
- `surprise/head_loss` = `RMSE(surprise_head, detached_per_sample_dynamics_error)` — trains surprise_head.
- `losses/total_loss` = `MSE(forward^H(obs, actor), state_{t+H})` — H-step rollout, the actor's actual training signal at default weights.

The actor is trained on a weighted mix, optionally minus an exploration term:

```
actor_step_loss = actor_loss_weight * actor_loss
                + total_loss_weight * total_loss
                - actor_surprise_coef * std(normalized_surprise across batch)
```

| `--actor-loss-weight` | `--total-loss-weight` | Behavior |
|---|---|---|
| `0.0` (default) | `1.0` (default) | **Pure model-based imitation.** Actor never sees the expert's action as a target; it only sees expert `(s, s_{t+H})` pairs and finds actions whose rolled-out forward predictions land on the goal state. Surprisingly competitive on PickCube. |
| `1.0` | `> 0` | Joint: direct BC + model-grounded regularization. |
| `1.0` | `0.0` | Pure BC. ForwardModel still trains but doesn't influence the actor. |

`total_loss - dynamics_loss` is a useful compounding-error diagnostic: small
when the actor stays in distribution, grows when it drifts off-policy w.r.t.
the forward model.

### Surprise

The surprise head is a self-modeling diagnostic + an optional exploration knob.
At training time it learns the forward model's own per-sample prediction error;
calling `forward_model.normalize_actual(raw)` sigmoid-standardizes against EMA
buffers to give a `[0, 1]` reading where 0.5 = "as surprised as the recent
training-time average." Buffers (`surprise_mean`, `surprise_sq_mean`) persist
in `state_dict` so checkpoints are self-contained.

Set `--actor-surprise-coef > 0` to reward the actor for producing a *diverse*
distribution of normalized surprise across the batch (std, not mean — avoids
the mode collapse where the actor picks one high-surprise trick everywhere).
Competes with the imitation objective; start in the 0.01–0.05 range. Requires
`--total-loss-weight > 0`.

### Online buffer + rollouts

The dynamics model trains on a mix of expert transitions (from the BC dataset)
and on-policy transitions collected from live env rollouts. Each `rollout_freq`
iterations the trainer runs two passes:

1. **Rollouts** — `num_rollout_episodes` (default 100) episodes in the env, optionally
   with Gaussian action noise (`--explore-sigma`, default 0). Each `(s, a, s')` is
   pushed into a fixed-size FIFO ring ([`OnlineBuffer`](src/buffer/online.py)). No
   metrics, no influence on the eval signal.
2. **Eval** — `num_eval_episodes` (default 50) deterministic episodes for metrics
   (`success_at_end`, `success_once`, `episode_return`). Drives the best-checkpoint
   signal.

Per dynamics-step, `k = floor(batch_size * online_mix_ratio)` samples are drawn
from the buffer (clamped by buffer size early in training); the rest comes from
the BC batch. The two slices are normalized with the same pinned dataset stats so
the forward model sees a single space. The actor's `total_loss` always uses the
BC slice (`obs`, `goal_obs`) only — no online tensors enter the actor's graph.

The default (`online_mix_ratio=0.95`) means the forward model is shaped almost
entirely by what the actor *actually does* in the env, while the actor learns
inverse-dynamics targets against that on-policy-trained model.

Logged tensors:

- `losses/dynamics_loss_bc` / `losses/dynamics_loss_online` — disjoint sub-batches
- `online/buffer_size` — useful for tracking buffer warmup

### Other flags

```text
--dataset-dir          data/pickcube_rl
--total-iters          50000
--batch-size           1024
--lr                   3e-4
--actor-horizon        1          # H-step actor rollout vs state_{t+H} (H=1 = single-step)
--actor-surprise-coef  0.0        # reward batch-std of normalized rollout surprise (0 = off)
--online-buffer-size   300000     # FIFO capacity (transitions)
--online-mix-ratio     0.95       # fraction of dynamics batch from online buffer
--explore-sigma        0.1        # rollout-only Gaussian noise on actions
--rollout-freq         1000       # train iters between rollout/eval rounds
--num-rollout-episodes 100        # episodes per rollout (feeds buffer, no metrics)
--num-eval-episodes    50         # deterministic eval episodes (drives best-ckpt)
--max-episode-steps    100
--normalize-states     off        # per-dim zero-mean unit-var input normalization
--seed                 42
```

## Eval / playback

```bash
uv run python play.py \
  --checkpoint runs/bc/<RUN>/checkpoints/best_eval_success_at_end.pt \
  --episodes 5
```

Opens the SAPIEN viewer (macOS uses MoltenVK). `--no-gui` for headless;
`--video-dir <path>` to save mp4s via `RecordEpisode`. Recovers normalization
stats from the checkpoint's saved args if the run used `--normalize-states`.

`play_bc_baseline.py` is the equivalent loader for `train_bc_baseline.py`
checkpoints.

## Baseline reference

`train_bc_baseline.py` is a byte-equivalent port of the upstream ManiSkill
state-based BC baseline (`examples/baselines/bc/bc.py`). It reads the monolithic
h5 from `replay_trajectory` rather than the per-episode dataset, and trains the
actor with plain `MSE(actor(obs), expert_action)`. Use it when you want a strict
comparison against the published number — not as a starting point for changes.

```bash
uv run python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o state \
  --save-traj -b physx_cpu -n 10
uv run python train_bc_baseline.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --total-iters 50000 --run-name pickcube-bc-baseline
```

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
  actor/                Actor + ForwardModel
  bc/                   StateBCDataset (loads per-episode dataset)
  buffer/               OnlineBuffer (FIFO ring fed by rollouts)
  datasets/             builder.py for per-episode dataset construction
  backbone.py           encode_images (R3M wrapper, used by builder)
  observations.py

train.py                Main trainer (model-based imitation + online buffer)
play.py                 Checkpoint replay for train.py
build_dataset.py        Per-episode dataset builder

train_bc_baseline.py    Reference: upstream-equivalent state-based BC
play_bc_baseline.py     Reference: checkpoint replay for the baseline
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
imported by `train.py` or `train_bc_baseline.py`. Worth revisiting only if we
want to layer planning or multi-step world-model training on top.

## dstack

```bash
uv run dstack secret set wandb_api_key '<your-wandb-key>'   # one-time
uv run dstack apply -f dstack.yml                           # submit run
```
