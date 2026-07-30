#!/usr/bin/env python3
"""Run a compact PPO hyperparameter ablation and rank trials on validation data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from caprl.data.loader import load_price_csv
from caprl.evaluation.config import load_experiment_config
from caprl.evaluation.runner import ExperimentConfig, run_experiment


DEFAULT_OUTPUT = Path("outputs/hyperparameter")
DEFAULT_CONFIG = Path("configs/experiments/primary.yaml")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_VARIANTS = ("full",)
DEFAULT_FOLDS = (0, 4, 9, 14, 18)
LEARNING_RATES = (1e-4, 3e-4)
EPOCHS_PER_UPDATE = (3, 5, 10)
GAMMAS = (0.99, 1.0)


def trial_definitions() -> list[dict[str, float | int | str]]:
    """Return the pre-registered 2 x 3 x 2 compact tuning grid."""
    definitions = []
    for number, (learning_rate, epochs, gamma) in enumerate(
        product(LEARNING_RATES, EPOCHS_PER_UPDATE, GAMMAS),
        start=1,
    ):
        definitions.append(
            {
                "trial": f"trial_{number:02d}",
                "learning_rate": learning_rate,
                "epochs_per_update": epochs,
                "gamma": gamma,
            }
        )
    return definitions


def tuning_config(
    base: ExperimentConfig,
    trial: dict[str, float | int | str],
    *,
    seeds: tuple[int, ...],
    variants: tuple[str, ...],
    training_steps: int,
) -> ExperimentConfig:
    ppo = replace(
        base.ppo,
        learning_rate=float(trial["learning_rate"]),
        epochs_per_update=int(trial["epochs_per_update"]),
        gamma=float(trial["gamma"]),
        training_steps=training_steps,
    )
    return replace(base, ppo=ppo, rl_seeds=seeds, variants=variants)


def summarize_trial(
    trial: dict[str, float | int | str],
    trial_directory: Path,
) -> list[dict[str, Any]]:
    """Summarize validation metrics after averaging seeds within each fold."""
    metrics = pd.read_parquet(trial_directory / "validation_seed_metrics.parquet")
    metric_columns = [
        "cumulative_net_return",
        "sharpe_ratio",
        "maximum_drawdown",
        "annual_turnover",
    ]
    records = []
    for strategy, strategy_metrics in metrics.groupby("strategy", observed=True):
        fold_means = strategy_metrics.groupby("fold", observed=True)[metric_columns].mean()
        seed_returns = strategy_metrics.groupby("seed", observed=True)[
            "cumulative_net_return"
        ].mean()
        records.append(
            {
                **trial,
                "strategy": strategy,
                "folds": strategy_metrics["fold"].nunique(),
                "seeds": strategy_metrics["seed"].nunique(),
                "median_validation_return": fold_means["cumulative_net_return"].median(),
                "median_validation_sharpe": fold_means["sharpe_ratio"].median(),
                "median_validation_drawdown": fold_means["maximum_drawdown"].median(),
                "median_validation_turnover": fold_means["annual_turnover"].median(),
                "validation_return_fold_iqr": (
                    fold_means["cumulative_net_return"].quantile(0.75)
                    - fold_means["cumulative_net_return"].quantile(0.25)
                ),
                "validation_return_seed_std": seed_returns.std(ddof=1),
            }
        )
    return records


def write_manifest(
    output: Path,
    trials: list[dict[str, float | int | str]],
    *,
    arguments: argparse.Namespace,
) -> None:
    manifest = {
        "purpose": "PPO hyperparameter selection using validation periods only",
        "ranking": [
            "median_validation_return (descending)",
            "median_validation_sharpe (descending)",
            "median_validation_turnover (ascending)",
        ],
        "data": str(arguments.data),
        "base_config": str(arguments.config),
        "training_steps": arguments.training_steps,
        "seeds": list(arguments.seeds),
        "variants": list(arguments.variants),
        "folds": list(arguments.folds),
        "trials": trials,
    }
    (output / "tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def rank_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "median_validation_return",
            "median_validation_sharpe",
            "median_validation_turnover",
        ],
        ascending=[False, False, True],
        ignore_index=True,
    )


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("seeds must be distinct non-negative integers")
    return parsed


def parse_string_tuple(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("variants must be distinct non-empty names")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 12-trial PPO learning-rate/epoch/gamma ablation. "
            "Trials are ranked exclusively on validation-period metrics."
        )
    )
    parser.add_argument("data", type=Path, help="CSV containing date and adjusted_close")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-steps", type=int, default=300_000)
    parser.add_argument("--seeds", type=parse_integer_tuple, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--folds",
        type=parse_integer_tuple,
        default=DEFAULT_FOLDS,
        help="comma-separated screening folds (default: 0,4,9,14,18)",
    )
    parser.add_argument("--variants", type=parse_string_tuple, default=DEFAULT_VARIANTS)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="adjusted_close")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip completed trials and retain their validation summaries",
    )
    arguments = parser.parse_args()
    if not arguments.data.is_file():
        parser.error(f"price file does not exist: {arguments.data}")
    if not arguments.config.is_file():
        parser.error(f"experiment config does not exist: {arguments.config}")
    if arguments.training_steps <= 0:
        parser.error("--training-steps must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    trials = trial_definitions()
    write_manifest(arguments.output, trials, arguments=arguments)
    prices = load_price_csv(
        arguments.data,
        date_column=arguments.date_column,
        adjusted_close_column=arguments.price_column,
    )["adjusted_close"]
    base_config = load_experiment_config(arguments.config)
    records: list[dict[str, Any]] = []

    for trial in trials:
        trial_directory = arguments.output / str(trial["trial"])
        validation_metrics = trial_directory / "validation_seed_metrics.parquet"
        if validation_metrics.is_file() and arguments.resume:
            logging.info("retaining completed %s", trial["trial"])
        else:
            if trial_directory.exists() and any(trial_directory.iterdir()):
                parser.error(
                    f"{trial_directory} is not empty; use --resume only if the trial "
                    "has a completed validation_seed_metrics.parquet"
                )
            config = tuning_config(
                base_config,
                trial,
                seeds=arguments.seeds,
                variants=arguments.variants,
                training_steps=arguments.training_steps,
            )
            logging.info("starting %s: %s", trial["trial"], asdict(config.ppo))
            run_experiment(
                prices,
                trial_directory,
                config=config,
                run_rl=True,
                fold_indices=arguments.folds,
            )
        records.extend(summarize_trial(trial, trial_directory))
        rank_summary(records).to_csv(arguments.output / "tuning_summary.csv", index=False)

    logging.info(
        "completed %d trials; ranking written to %s",
        len(trials),
        arguments.output / "tuning_summary.csv",
    )


if __name__ == "__main__":
    main()
