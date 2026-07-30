"""Compact reproducible PPO trainer for bounded cap utilization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from caprl.agents.policies import BetaActorCritic


@dataclass(frozen=True)
class PPOConfig:
    hidden_layers: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    batch_size: int = 256
    epochs_per_update: int = 10
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    rollout_steps: int = 2048
    training_steps: int = 1_000_000
    device: str = "cpu"


@dataclass(frozen=True)
class PPOTrainingMetric:
    steps: int
    policy_loss: float
    value_loss: float
    entropy: float
    mean_reward: float


@dataclass
class PPOAgent:
    model: BetaActorCritic
    config: PPOConfig
    observation_size: int

    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.config.device)
        with torch.no_grad():
            distribution, _ = self.model.distribution_and_value(tensor)
            action = distribution.mean if deterministic else distribution.sample()
        return np.asarray([float(action.cpu())], dtype=np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "config": asdict(self.config),
            "observation_size": self.observation_size,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "PPOAgent":
        config_values = dict(payload["config"])
        config_values["hidden_layers"] = tuple(config_values["hidden_layers"])
        config = PPOConfig(**config_values)
        observation_size = int(payload["observation_size"])
        model = BetaActorCritic(observation_size, hidden_layers=config.hidden_layers)
        model.load_state_dict(payload["model"])
        model.to(config.device)
        model.eval()
        return cls(model, config, observation_size)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_ppo(
    env_factory: Callable[[], gym.Env[np.ndarray, np.ndarray]],
    *,
    config: PPOConfig | None = None,
    seed: int = 42,
    progress_callback: Callable[[PPOTrainingMetric], None] | None = None,
) -> tuple[PPOAgent, list[PPOTrainingMetric]]:
    config = config or PPOConfig()
    _validate_config(config)
    seed_everything(seed)
    env = env_factory()
    observation, _ = env.reset(seed=seed)
    observation_size = int(np.prod(env.observation_space.shape))
    model = BetaActorCritic(observation_size, hidden_layers=config.hidden_layers).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    agent = PPOAgent(model, config, observation_size)
    metrics: list[PPOTrainingMetric] = []
    total_steps = 0
    rng = np.random.default_rng(seed)
    while total_steps < config.training_steps:
        rollout_length = min(config.rollout_steps, config.training_steps - total_steps)
        rollout, observation = _collect_rollout(
            env, model, observation, rollout_length, config, seed + total_steps + 1
        )
        advantages, returns = _generalized_advantages(rollout, config)
        metric = _ppo_update(model, optimizer, rollout, advantages, returns, config, rng)
        total_steps += rollout_length
        training_metric = PPOTrainingMetric(
            steps=total_steps,
            policy_loss=metric[0],
            value_loss=metric[1],
            entropy=metric[2],
            mean_reward=float(np.mean(rollout["rewards"])),
        )
        metrics.append(training_metric)
        if progress_callback is not None:
            progress_callback(training_metric)
    env.close()
    model.eval()
    return agent, metrics


def _collect_rollout(
    env: gym.Env[np.ndarray, np.ndarray],
    model: BetaActorCritic,
    observation: np.ndarray,
    length: int,
    config: PPOConfig,
    reset_seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    storage: dict[str, list[object]] = {
        "observations": [],
        "actions": [],
        "log_probabilities": [],
        "values": [],
        "next_values": [],
        "rewards": [],
        "terminated": [],
        "done": [],
    }
    for step in range(length):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=config.device
        )
        with torch.no_grad():
            distribution, value = model.distribution_and_value(observation_tensor)
            action = distribution.sample().clamp(1e-6, 1.0 - 1e-6)
            log_probability = distribution.log_prob(action)
        next_observation, reward, terminated, truncated, _ = env.step(
            np.asarray([float(action.cpu())], dtype=np.float32)
        )
        with torch.no_grad():
            _, next_value = model.distribution_and_value(
                torch.as_tensor(next_observation, dtype=torch.float32, device=config.device)
            )
        storage["observations"].append(observation)
        storage["actions"].append(float(action.cpu()))
        storage["log_probabilities"].append(float(log_probability.cpu()))
        storage["values"].append(float(value.cpu()))
        storage["next_values"].append(float(next_value.cpu()))
        storage["rewards"].append(float(reward))
        storage["terminated"].append(float(terminated))
        storage["done"].append(float(terminated or truncated))
        if terminated or truncated:
            observation, _ = env.reset(seed=reset_seed + step)
        else:
            observation = next_observation
    return {key: np.asarray(value, dtype=np.float32) for key, value in storage.items()}, observation


def _generalized_advantages(
    rollout: dict[str, np.ndarray], config: PPOConfig
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rollout["rewards"])
    running_advantage = 0.0
    for index in range(len(advantages) - 1, -1, -1):
        bootstrap = 1.0 - rollout["terminated"][index]
        continuation = 1.0 - rollout["done"][index]
        delta = (
            rollout["rewards"][index]
            + config.gamma * rollout["next_values"][index] * bootstrap
            - rollout["values"][index]
        )
        running_advantage = (
            delta + config.gamma * config.gae_lambda * continuation * running_advantage
        )
        advantages[index] = running_advantage
    returns = advantages + rollout["values"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def _ppo_update(
    model: BetaActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    config: PPOConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=config.device)
        for key, value in rollout.items()
    }
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=config.device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=config.device)
    losses: list[tuple[float, float, float]] = []
    for _ in range(config.epochs_per_update):
        for indices in _minibatches(len(advantages), config.batch_size, rng):
            distribution, values = model.distribution_and_value(tensors["observations"][indices])
            new_log_probabilities = distribution.log_prob(tensors["actions"][indices])
            ratio = torch.exp(new_log_probabilities - tensors["log_probabilities"][indices])
            unclipped = ratio * advantage_tensor[indices]
            clipped = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip) * advantage_tensor[
                indices
            ]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, return_tensor[indices])
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()
            losses.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
    return tuple(float(np.mean(values)) for values in zip(*losses, strict=True))


def _minibatches(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    permutation = rng.permutation(size)
    return [permutation[start : start + batch_size] for start in range(0, size, batch_size)]


def _validate_config(config: PPOConfig) -> None:
    positive = (
        config.learning_rate,
        config.gae_lambda,
        config.batch_size,
        config.epochs_per_update,
        config.max_gradient_norm,
        config.rollout_steps,
        config.training_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("PPO optimization settings must be positive")
    if not 0 <= config.ppo_clip < 1 or not 0 <= config.gamma <= 1:
        raise ValueError("PPO clip and gamma are outside their valid ranges")
