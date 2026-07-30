"""Timing-safe deterministic backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from caprl.evaluation.accounting import portfolio_step


TRAJECTORY_COLUMNS = (
    "next_timestamp",
    "asset_return",
    "exposure",
    "previous_exposure",
    "turnover",
    "gross_return",
    "transaction_cost",
    "net_return",
    "reward",
    "wealth",
    "drawdown",
)


def run_backtest(
    prices: pd.Series,
    target_exposures: pd.Series,
    *,
    transaction_cost_rate: float = 0.0,
    initial_wealth: float = 1.0,
    initial_exposure: float = 0.0,
) -> pd.DataFrame:
    """Apply exposure at t to the simple asset return from t to t+1."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if not prices.index.equals(target_exposures.index):
        raise ValueError("prices and target_exposures must have identical indices")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("timestamps must be unique and increasing")
    if len(prices) < 2:
        raise ValueError("at least two prices are required")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("prices must be non-missing and positive")
    exposures = target_exposures.astype(float)
    if not np.isfinite(exposures.to_numpy()).all() or not exposures.between(0, 1).all():
        raise ValueError("target exposures must be finite and in [0, 1]")

    next_returns = prices.shift(-1) / prices - 1.0
    wealth = initial_wealth
    running_max = initial_wealth
    previous_exposure = initial_exposure
    records: list[dict[str, object]] = []
    for position, timestamp in enumerate(prices.index[:-1]):
        step = portfolio_step(
            wealth=wealth,
            previous_exposure=previous_exposure,
            target_exposure=float(exposures.iloc[position]),
            next_return=float(next_returns.iloc[position]),
            transaction_cost_rate=transaction_cost_rate,
        )
        wealth = step.wealth
        running_max = max(running_max, wealth)
        records.append(
            {
                "timestamp": timestamp,
                "next_timestamp": prices.index[position + 1],
                "asset_return": float(next_returns.iloc[position]),
                "exposure": step.exposure,
                "previous_exposure": step.previous_exposure,
                "turnover": step.turnover,
                "gross_return": step.gross_return,
                "transaction_cost": step.transaction_cost,
                "net_return": step.net_return,
                "reward": step.reward,
                "wealth": wealth,
                "drawdown": wealth / running_max - 1.0,
            }
        )
        previous_exposure = step.exposure
    trajectory = pd.DataFrame.from_records(records).set_index("timestamp")
    return trajectory.loc[:, TRAJECTORY_COLUMNS]

