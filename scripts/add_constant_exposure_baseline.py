#!/usr/bin/env python3
"""Backfill a constant-exposure baseline into a completed experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caprl.baselines import buy_hold_exposure
from caprl.evaluation.backtest import run_backtest
from caprl.evaluation.config import load_frozen_experiment_config
from caprl.evaluation.metrics import calculate_metrics
from caprl.evaluation.regime_metrics import calculate_regime_metrics
from caprl.evaluation.risk_metrics import calculate_risk_metrics
from caprl.evaluation.runner import aggregate_completed_folds, _bootstrap_record
from caprl.utils.serialization import write_frame


def _prices_from_trajectory(trajectory: pd.DataFrame) -> pd.Series:
    index = trajectory.index.append(
        pd.DatetimeIndex([pd.Timestamp(trajectory["next_timestamp"].iloc[-1])])
    )
    values = np.r_[1.0, np.cumprod(1.0 + trajectory["asset_return"].to_numpy(dtype=float))]
    return pd.Series(values, index=index, name="adjusted_close")


def _enrich_from_reference(
    trajectory: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    diagnostic_columns = [
        "bear_probability",
        "bull_probability",
        "most_likely_state",
        "posterior_entropy",
        "conditional_mean",
        "conditional_second_moment",
        "conditional_cvar",
    ]
    result = trajectory.join(reference.loc[:, diagnostic_columns])
    result["cap"] = reference["cap"]
    result["cap_binds"] = result["exposure"] >= result["cap"] - 1e-12
    result["raw_action"] = np.divide(
        result["exposure"],
        result["cap"],
        out=np.zeros(len(result)),
        where=result["cap"].to_numpy() > 0,
    )
    result["step_risk_charge"] = result["exposure"] * result["conditional_cvar"]
    return result


def backfill(experiment: Path, *, exposure: float, strategy: str) -> int:
    if not 0.0 <= exposure <= 1.0:
        raise ValueError("exposure must be in [0, 1]")
    config = load_frozen_experiment_config(experiment / "config.yaml")
    fold_directories = sorted(path for path in experiment.glob("fold_*") if path.is_dir())
    if not fold_directories:
        raise ValueError(f"no completed folds found in {experiment}")

    for fold_directory in fold_directories:
        fold = int(fold_directory.name.split("_")[-1])
        reference = pd.read_parquet(fold_directory / "backtests" / "buy_hold.parquet")
        prices = _prices_from_trajectory(reference)
        trajectory = run_backtest(
            prices,
            buy_hold_exposure(prices, exposure=exposure),
            transaction_cost_rate=config.transaction_cost_bps / 10_000.0,
        )
        trajectory = _enrich_from_reference(trajectory, reference)
        write_frame(fold_directory / "backtests" / f"{strategy}.parquet", trajectory)
        write_frame(
            fold_directory / "regime_metrics" / f"{strategy}.parquet",
            calculate_regime_metrics(trajectory),
        )

        cap_parameters = json.loads((fold_directory / "cap_parameters.json").read_text())
        metric = {
            "fold": fold,
            "strategy": strategy,
            "seed": None,
            **calculate_metrics(trajectory),
            **calculate_risk_metrics(
                trajectory,
                target_budget=float(cap_parameters["target_budget"]),
                gamma=config.risk_gamma,
            ),
        }
        fold_metrics_path = fold_directory / "summary_metrics.parquet"
        fold_metrics = pd.read_parquet(fold_metrics_path)
        fold_metrics = pd.concat(
            [fold_metrics[fold_metrics["strategy"] != strategy], pd.DataFrame([metric])],
            ignore_index=True,
        )
        write_frame(fold_metrics_path, fold_metrics)

        full_cap = pd.read_parquet(fold_directory / "backtests" / "full_cap.parquet")
        bootstrap = _bootstrap_record(
            strategy,
            None,
            trajectory["net_return"],
            full_cap["net_return"],
            config,
        )
        bootstrap_path = fold_directory / "bootstrap_results.parquet"
        bootstrap_frame = pd.read_parquet(bootstrap_path)
        bootstrap_frame = pd.concat(
            [
                bootstrap_frame[bootstrap_frame["strategy"] != strategy],
                pd.DataFrame([bootstrap]),
            ],
            ignore_index=True,
        )
        write_frame(bootstrap_path, bootstrap_frame)

    aggregate_completed_folds(experiment)
    return len(fold_directories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--exposure", type=float, default=0.52)
    parser.add_argument("--strategy", default="buy_hold_52")
    arguments = parser.parse_args()
    folds = backfill(
        arguments.experiment,
        exposure=arguments.exposure,
        strategy=arguments.strategy,
    )
    print(
        f"backfilled {arguments.strategy} at exposure {arguments.exposure:.2f} "
        f"into {folds} folds"
    )


if __name__ == "__main__":
    main()
