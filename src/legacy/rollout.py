"""Generic environment rollout loop.

Drives a single non-vectorized Gymnasium env with an ``actor(obs) -> action``
callable for ``num_episodes``, optionally rendering each step. If ``actor`` has
a ``reset()`` method it is called between episodes.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from src.legacy.utils import to_scalar_bool


def rollout(
    env: gym.Env,
    actor: Callable[[dict], np.ndarray],
    num_episodes: int,
    *,
    max_steps: int = 200,
    action_chunk_size: int = 1,
    seed: int = 42,
    render: bool = False,
    log_episode: Callable[[int, int, float, bool], None] | None = None,
) -> dict:
    """Roll out ``actor`` and return aggregate success-rate / return metrics."""
    if action_chunk_size < 1:
        raise ValueError(f"action_chunk_size must be >= 1, got {action_chunk_size}")

    successes = 0
    returns: list[float] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(actor, "reset"):
            actor.reset()
        ep_return, succeeded, env_steps, done = 0.0, False, 0, False
        while env_steps < max_steps and not done:
            action_chunk = np.asarray(actor(obs))
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)
            if action_chunk.shape[0] != action_chunk_size:
                raise ValueError(
                    f"actor returned {action_chunk.shape[0]} actions, expected {action_chunk_size}"
                )
            for action in action_chunk:
                obs, reward, terminated, truncated, info = env.step(action)
                env_steps += 1
                if render:
                    env.render()
                ep_return += float(np.asarray(reward).reshape(-1)[0])
                succeeded = succeeded or to_scalar_bool(info.get("success", False))
                done = to_scalar_bool(terminated) or to_scalar_bool(truncated)
                if done or env_steps >= max_steps:
                    break
        successes += int(succeeded)
        returns.append(ep_return)
        if log_episode is not None:
            log_episode(ep, env_steps, ep_return, succeeded)
    return {
        "success_rate": successes / num_episodes,
        "mean_return": float(np.mean(returns)),
    }
