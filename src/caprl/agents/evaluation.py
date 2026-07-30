"""Locked chronological policy evaluation."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

from caprl.agents.ppo import PPOAgent


def evaluate_agent(
    agent: PPOAgent,
    env_factory: Callable[[], gym.Env[np.ndarray, np.ndarray]],
    *,
    deterministic: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate one complete chronological interval without model updates."""
    env = env_factory()
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.predict(observation, deterministic=deterministic)
        observation, _, terminated, truncated, _ = env.step(action)
    trajectory = env.trajectory.copy()
    env.close()
    return trajectory
