"""Gymnasium environment for cap-utilization trading actions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-asset environment where action at t earns return t to t+1."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.Series,
        policy_features: pd.DataFrame,
        caps: pd.Series,
        *,
        regime_data: pd.DataFrame | None = None,
        risk_data: pd.DataFrame | None = None,
        episode_length: int | None = 252,
        transaction_cost_rate: float = 0.001,
        random_start: bool = False,
        include_cap_in_observation: bool = True,
    ) -> None:
        super().__init__()
        if not prices.index.equals(policy_features.index) or not prices.index.equals(caps.index):
            raise ValueError("prices, policy features, and caps must have identical indices")
        for optional in (regime_data, risk_data):
            if optional is not None and not prices.index.equals(optional.index):
                raise ValueError("all optional data must share the price index")
        if len(prices) < 2:
            raise ValueError("at least two timestamps are required")
        if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
            raise ValueError("timestamps must be unique and increasing")
        if prices.isna().any() or (prices <= 0).any():
            raise ValueError("prices must be non-missing and positive")
        if policy_features.empty or not np.isfinite(policy_features.to_numpy(dtype=float)).all():
            raise ValueError("policy features must be non-empty and finite")
        cap_values = caps.to_numpy(dtype=float)
        if not np.isfinite(cap_values).all() or not ((0 <= cap_values) & (cap_values <= 1)).all():
            raise ValueError("caps must be finite and in [0, 1]")
        if episode_length is not None and episode_length <= 0:
            raise ValueError("episode_length must be positive or None")
        if transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate must be non-negative")

        self.prices = prices.astype(float)
        self.policy_features = policy_features.astype(np.float32)
        self.caps = caps.astype(float)
        self.regime_data = regime_data
        self.risk_data = risk_data
        self.episode_length = episode_length
        self.transaction_cost_rate = transaction_cost_rate
        self.random_start = random_start
        self.include_cap_in_observation = include_cap_in_observation
        self._next_returns = self.prices.shift(-1) / self.prices - 1.0
        observation_size = self.policy_features.shape[1] + 2 + int(include_cap_in_observation)
        float_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-float_limit, high=float_limit, shape=(observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._trajectory: list[dict[str, Any]] = []
        self._position = 0
        self._end_position = 0
        self._wealth = 1.0
        self._running_max = 1.0
        self._previous_exposure = 0.0

    @property
    def trajectory(self) -> pd.DataFrame:
        if not self._trajectory:
            return pd.DataFrame()
        return pd.DataFrame.from_records(self._trajectory).set_index("timestamp")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        maximum_start = len(self.prices) - 2
        requested_length = self.episode_length or len(self.prices) - 1
        if self.random_start:
            maximum_start = max(0, len(self.prices) - 1 - requested_length)
            start = int(self.np_random.integers(0, maximum_start + 1))
        else:
            start = 0
        start = int(options.get("start_index", start))
        if not 0 <= start <= len(self.prices) - 2:
            raise ValueError("start_index must leave at least one subsequent return")
        length = min(requested_length, len(self.prices) - 1 - start)
        self._position = start
        self._end_position = start + length
        self._wealth = 1.0
        self._running_max = 1.0
        self._previous_exposure = 0.0
        self._trajectory = []
        return self._observation(), {"start_index": start, "end_index": self._end_position}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = float(np.asarray(action, dtype=float).reshape(-1)[0])
        if not np.isfinite(action_value) or not 0.0 <= action_value <= 1.0:
            raise ValueError("action must be finite and in [0, 1]")
        timestamp = self.prices.index[self._position]
        next_timestamp = self.prices.index[self._position + 1]
        cap = float(self.caps.iloc[self._position])
        exposure = action_value * cap
        turnover = abs(exposure - self._previous_exposure)
        asset_return = float(self._next_returns.iloc[self._position])
        gross_return = exposure * asset_return
        transaction_cost = self.transaction_cost_rate * turnover
        net_return = gross_return - transaction_cost
        net_factor = 1.0 + net_return
        invalid = net_factor <= 0.0
        if invalid:
            reward = -1e6
            self._wealth = 0.0
        else:
            reward = float(np.log(net_factor))
            self._wealth *= net_factor
        self._running_max = max(self._running_max, self._wealth)
        drawdown = self._wealth / self._running_max - 1.0
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "next_timestamp": next_timestamp,
            "raw_action": action_value,
            "cap": cap,
            "exposure": exposure,
            "previous_exposure": self._previous_exposure,
            "turnover": turnover,
            "asset_return": asset_return,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "reward": reward,
            "wealth": self._wealth,
            "drawdown": drawdown,
            "cap_binds": bool(cap < 1.0 and action_value > 0.0),
            "invalid": invalid,
        }
        if self.regime_data is not None:
            record.update(self.regime_data.iloc[self._position].to_dict())
        if self.risk_data is not None:
            record.update(self.risk_data.iloc[self._position].to_dict())
            if "conditional_cvar" in self.risk_data:
                record["step_risk_charge"] = exposure * float(
                    self.risk_data["conditional_cvar"].iloc[self._position]
                )
        self._trajectory.append(record)
        self._previous_exposure = exposure
        self._position += 1
        terminated = invalid
        truncated = self._position >= self._end_position and not terminated
        info = record.copy()
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        static = self.policy_features.iloc[self._position].to_numpy(dtype=np.float32)
        drawdown = self._wealth / self._running_max - 1.0
        dynamic = [self._previous_exposure, drawdown]
        if self.include_cap_in_observation:
            dynamic.append(float(self.caps.iloc[self._position]))
        return np.concatenate((static, np.asarray(dynamic, dtype=np.float32)))
