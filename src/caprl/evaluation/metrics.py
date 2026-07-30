"""Performance and trading-behavior metrics for step trajectories."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def calculate_metrics(trajectory: pd.DataFrame, *, periods_per_year: int = 252) -> dict[str, float]:
    required = {"net_return", "reward", "wealth", "drawdown", "turnover", "transaction_cost", "exposure"}
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    if trajectory.empty:
        raise ValueError("trajectory must not be empty")
    returns = trajectory["net_return"].astype(float)
    observations = len(returns)
    years = observations / periods_per_year
    ending_wealth = float(trajectory["wealth"].iloc[-1])
    cumulative_return = ending_wealth - 1.0
    annualized_log_return = float(trajectory["reward"].sum() / years)
    cagr = math.exp(annualized_log_return) - 1.0
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = _safe_ratio(float(returns.mean() * periods_per_year), annualized_volatility)
    downside = returns[returns < 0]
    downside_volatility = float(downside.std(ddof=1) * math.sqrt(periods_per_year)) if len(downside) > 1 else math.nan
    sortino = _safe_ratio(float(returns.mean() * periods_per_year), downside_volatility)
    maximum_drawdown = float(trajectory["drawdown"].min())
    calmar = _safe_ratio(cagr, abs(maximum_drawdown))
    monthly = (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0
    rolling_year = (1.0 + returns).rolling(periods_per_year).apply(np.prod, raw=True) - 1.0
    material_changes = int((trajectory["turnover"] > 0.01).sum())
    return {
        "cumulative_net_return": cumulative_return,
        "cagr": cagr,
        "annualized_log_return": annualized_log_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "worst_month": float(monthly.min()),
        "worst_rolling_year": float(rolling_year.min()) if rolling_year.notna().any() else math.nan,
        "annual_turnover": float(trajectory["turnover"].sum() / years),
        "transaction_cost_drag": float(trajectory["transaction_cost"].sum()),
        "average_exposure": float(trajectory["exposure"].mean()),
        "exposure_volatility": float(trajectory["exposure"].std(ddof=1)),
        "material_position_changes": float(material_changes),
        "cap_binding_frequency": float(trajectory["cap_binds"].mean()) if "cap_binds" in trajectory else 0.0,
        "mean_cap_utilization": _mean_cap_utilization(trajectory),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def _mean_cap_utilization(trajectory: pd.DataFrame) -> float:
    if "cap" not in trajectory:
        return math.nan
    positive_cap = trajectory["cap"] > 0
    if not positive_cap.any():
        return 0.0
    return float((trajectory.loc[positive_cap, "exposure"] / trajectory.loc[positive_cap, "cap"]).mean())

