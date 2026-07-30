"""Risk-mandate metrics distinct from drawdown diagnostics."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def calculate_risk_metrics(
    trajectory: pd.DataFrame,
    *,
    target_budget: float | None = None,
    gamma: float = 1.0,
) -> dict[str, float]:
    if "conditional_cvar" not in trajectory or "exposure" not in trajectory:
        return {}
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    charges = trajectory["exposure"].to_numpy() * trajectory["conditional_cvar"].to_numpy()
    discounted = float(np.sum(np.power(gamma, np.arange(len(charges))) * charges))
    result = {
        "discounted_realized_risk_charge": discounted,
        "normalized_average_risk_charge": float(np.mean(charges)),
        "realized_return_cvar_95": _historical_cvar(-trajectory["net_return"].to_numpy(), 0.95),
    }
    if target_budget is not None:
        result["realized_risk_over_beta"] = discounted / target_budget if target_budget > 0 else math.nan
    return result


def _historical_cvar(losses: np.ndarray, confidence: float) -> float:
    value_at_risk = np.quantile(losses, confidence)
    return float(np.mean(losses[losses >= value_at_risk]))

