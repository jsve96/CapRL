"""Exposure and trading behavior conditional on filtered regimes."""

from __future__ import annotations

import pandas as pd


def calculate_regime_metrics(trajectory: pd.DataFrame) -> pd.DataFrame:
    required = {"bull_probability", "exposure", "turnover", "net_return", "drawdown"}
    missing = required.difference(trajectory.columns)
    if missing:
        raise ValueError(f"trajectory is missing regime fields: {sorted(missing)}")
    bins = pd.cut(
        trajectory["bull_probability"],
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        include_lowest=True,
    )
    grouped = trajectory.groupby(bins, observed=False)
    result = grouped[["exposure", "turnover", "net_return", "drawdown"]].mean()
    result["observations"] = grouped.size()
    if "cap" in trajectory:
        result["average_cap"] = grouped["cap"].mean()
    if "raw_action" in trajectory:
        result["average_utilization"] = grouped["raw_action"].mean()
    result.index = result.index.astype(str)
    result.index.name = "bull_probability_bin"
    return result
