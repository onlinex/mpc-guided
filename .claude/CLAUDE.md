# CLAUDE.md

Project-specific context. Keep concise — this file is loaded into Claude's context every session.

## What this repo is

Model-based imitation on ManiSkill PickCube. The actor is trained **only** through a jointly-learned forward dynamics model: given expert `(state, state_{t+H})` pairs, it has to find actions whose H-step rollout through the forward model lands on the goal state. The forward model trains mostly on the actor's own on-policy rollouts.

See [README.md](README.md) for the user-facing version.

## Active code path

When working on the project, default to these — don't touch anything else unless asked:

- [train.py](train.py) — main trainer
- [play.py](play.py) — checkpoint replay
- [build_dataset.py](build_dataset.py) — per-episode dataset builder
- [src/actor/](src/actor/) — `Actor` + `ForwardModel`
- [src/bc/](src/bc/) — `StateBCDataset`
- [src/buffer/](src/buffer/) — `OnlineBuffer`
- [src/datasets/builder.py](src/datasets/builder.py) — dataset builder internals

## Frozen — do not modify

These are baseline-only references from upstream, kept for comparison. They are not the path we're developing. **Do not change them** unless explicitly asked:

- [train_bc_baseline.py](train_bc_baseline.py)
- [play_bc_baseline.py](play_bc_baseline.py)

## Parked — off the active path

These are kept tested but aren't imported by the active flow. Don't reach for them when implementing new features:

- [src/legacy/](src/legacy/) — old actor/, dynamics/, datasets/, networks.py, rollout.py, utils.py
- [legacy/](legacy/) — old entry points (train_dynamics.py, demo.py)
- [tests/legacy/](tests/legacy/) — auto-skipped from default pytest runs

## Design decisions worth knowing

- **Two optimizers, no crossover**: `actor_optimizer` updates actor params only; `forward_optimizer` updates forward model params only. Even when a backward writes grads to the "wrong" set (e.g., actor's `total_loss` writes grads to forward params via the chain rule), those grads are discarded by the next `zero_grad()`. The separation is enforced by optimizer ownership, not by `.train()`/`.eval()` mode.
- **Online buffer feeds dynamics only**: the actor's `total_loss` always uses the BC slice (`obs`, `goal_obs`) — no online tensors enter the actor's graph.
- **Surprise head with EMA calibration**: [src/actor/forward.py](src/actor/forward.py) keeps `surprise_mean` / `surprise_sq_mean` as `register_buffer`s — they persist via `state_dict`, so checkpoints are self-contained. `normalize_actual()` is the canonical way to get a [0, 1] reading.
- **`ForwardModel.head_losses(s, a, target)` returns a `HeadLosses` NamedTuple** with `(state, surprise, per_sample_error)`. Use it and `HeadLosses.combine(n1, l1, n2, l2)` when computing per-slice dynamics losses — the "surprise target = per-sample MSE of state head" relationship lives on the model, not in the trainer.
- **`detach_surprise` kwarg in `ForwardModel.forward`**: defaults to True so the head's training loss doesn't tug the trunk — we tried joint training and both losses got worse (capacity competition), so the trunk is kept driven only by `state_head`'s loss. Pass False from the actor rollout when you want the actor to optimize *through* the surprise signal (used by `--actor-surprise-coef > 0`).
- **Variable-horizon actor rollout**: `--actor-horizon H` rolls actor+forward H steps and compares the final predicted state to `state_{t+H}`. H=1 is byte-equivalent to the original single-step behavior.
- **`reward_mode="dense"`** for env construction in both [train.py](train.py) and [play.py](play.py). Don't switch back to sparse without good reason — the `episode_return` metric becomes meaningless under sparse.

## Testing

pytest markers (declared in [pyproject.toml](pyproject.toml)):

- `env` — constructs a ManiSkill env (slow, seconds per test).
- `dataset` — reads from `data/pickcube_rl/`; auto-skipped if not built.
- `legacy` — tests under `tests/legacy/`; deselected by default.

Default `uv run pytest` runs everything except `legacy`. Use explicit marker filters to narrow down.

## Dataset assumption

Most workflows assume the per-episode dataset exists at `data/pickcube_rl/`. Build with `build_dataset.py`. The `dataset` pytest marker auto-skips if missing.
