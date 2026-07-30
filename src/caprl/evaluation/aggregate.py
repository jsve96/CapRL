"""Aggregation over folds and RL seeds."""

from __future__ import annotations

import pandas as pd


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if "strategy" not in metrics:
        raise ValueError("metrics must contain a strategy column")
    numeric = [
        column
        for column in metrics.select_dtypes(include="number").columns
        if column not in {"seed", "fold"}
    ]
    records = []
    for strategy, group in metrics.groupby("strategy", sort=True):
        for metric in numeric:
            values = group[metric].dropna()
            if values.empty:
                continue
            records.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "mean": float(values.mean()),
                    "observations": len(values),
                }
            )
    return pd.DataFrame(records)

