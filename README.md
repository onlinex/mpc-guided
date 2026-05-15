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

## dstack

One-time W&B secret setup, if future training code uses W&B:

```bash
uv run dstack secret set wandb_api_key '<your-wandb-key>'
```

Submit the current remote smoke rollout:

```bash
uv run dstack apply -f dstack.yml
```

Override demo args at submit time:

```bash
uv run dstack apply -f dstack.yml -- --episodes 4 --max-steps 100
```

When the TD-MPC2 trainer is added, replace the final command in `dstack.yml`
with that training entrypoint and keep the `maniskill/base` image plus
`physx_cuda` backend.
