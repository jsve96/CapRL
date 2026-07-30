#!/usr/bin/env python3
"""Run the CVaR-level by risk-fraction sensitivity analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from caprl.data.loader import load_price_csv
from caprl.evaluation.config import load_experiment_config
from caprl.evaluation.runner import ExperimentConfig, run_experiment


DEFAULT_OUTPUT = Path("outputs/cvar_risk_ablation")
DEFAULT_CONFIG = Path("configs/experiments/primary.yaml")
DEFAULT_VARIANTS = ("full",)
CVAR_LEVELS = (0.875, 0.90, 0.925, 0.95)
RISK_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SUMMARY_METRICS = (
    "cumulative_net_return",
    "sharpe_ratio",
    "maximum_drawdown",
    "annual_turnover",
    "transaction_cost_drag",
    "average_exposure",
    "cap_binding_frequency",
    "mean_cap_utilization",
    "realized_risk_over_beta",
)


def ablation_definitions() -> list[dict[str, float | str]]:
    """Return the complete 4 x 4 CVaR-level by risk-fraction design."""
    definitions: list[dict[str, float | str]] = []
    for number, (cvar, risk_fraction) in enumerate(
        product(CVAR_LEVELS, RISK_FRACTIONS),
        start=1,
    ):
        definitions.append(
            {
                "experiment": f"cvar_risk_{number:02d}",
                "design": "cvar_x_risk_fraction",
                "cvar_confidence": cvar,
                "risk_fraction": risk_fraction,
            }
        )
    return definitions


def make_config(
    base: ExperimentConfig,
    definition: dict[str, float | str],
    *,
    seeds: tuple[int, ...],
    variants: tuple[str, ...],
    training_steps: int,
) -> ExperimentConfig:
    return replace(
        base,
        cvar_confidence=float(definition["cvar_confidence"]),
        risk_fraction=float(definition["risk_fraction"]),
        rl_seeds=seeds,
        variants=variants,
        ppo=replace(base.ppo, training_steps=training_steps),
    )


def summarize_experiment(
    definition: dict[str, float | str],
    experiment_directory: Path,
) -> list[dict[str, Any]]:
    """Average PPO seeds within fold, then describe dispersion across folds."""
    metrics = pd.read_parquet(experiment_directory / "seed_metrics.parquet")
    records: list[dict[str, Any]] = []
    for strategy, strategy_metrics in metrics.groupby("strategy", observed=True):
        fold_metrics = strategy_metrics.groupby("fold", observed=True)[
            list(SUMMARY_METRICS)
        ].mean()
        record: dict[str, Any] = {
            **definition,
            "strategy": strategy,
            "folds": strategy_metrics["fold"].nunique(),
            "seeds": strategy_metrics["seed"].nunique(),
        }
        for metric in SUMMARY_METRICS:
            values = fold_metrics[metric].dropna()
            record[f"{metric}_median"] = values.median()
            record[f"{metric}_q25"] = values.quantile(0.25)
            record[f"{metric}_q75"] = values.quantile(0.75)
            record[f"{metric}_mean"] = values.mean()
        if strategy_metrics["seed"].notna().any():
            seed_means = strategy_metrics.groupby("seed", observed=True)[
                "cumulative_net_return"
            ].mean()
            record["return_seed_std"] = seed_means.std(ddof=1)
        else:
            record["return_seed_std"] = float("nan")
        records.append(record)
    return records


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be distinct non-negative integers")
    return parsed


def parse_string_tuple(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("variants must be distinct non-empty names")
    return parsed


def write_manifest(
    output: Path,
    definitions: list[dict[str, float | str]],
    arguments: argparse.Namespace,
    base_config: ExperimentConfig,
) -> None:
    manifest = {
        "purpose": "CVaR-level by normalized risk-fraction sensitivity analysis",
        "design": {
            "cvar_confidence": list(CVAR_LEVELS),
            "risk_fraction": list(RISK_FRACTIONS),
            "fixed_transaction_cost_bps": base_config.transaction_cost_bps,
            "unique_experiments": len(definitions),
        },
        "aggregation": (
            "PPO seeds are averaged within fold before across-fold summaries; "
            "seed dispersion is reported separately."
        ),
        "data": str(arguments.data),
        "base_config": str(arguments.config),
        "training_steps": arguments.training_steps,
        "seeds": list(arguments.seeds),
        "variants": list(arguments.variants),
        "folds": "all" if arguments.folds is None else list(arguments.folds),
        "experiments": definitions,
    }
    (output / "cvar_risk_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 16-configuration CVaR-level by risk-fraction sensitivity analysis."
        )
    )
    parser.add_argument("data", type=Path, help="CSV containing date and adjusted_close")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=parse_integer_tuple,
        default=None,
        help="comma-separated RL seeds; defaults to all seeds in the experiment config",
    )
    parser.add_argument(
        "--folds",
        type=parse_integer_tuple,
        default=None,
        help="optional comma-separated fold subset; defaults to all folds",
    )
    parser.add_argument(
        "--variants",
        type=parse_string_tuple,
        default=DEFAULT_VARIANTS,
        help="PPO variants to retrain (default: full)",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="adjusted_close")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip experiments with a completed seed_metrics.parquet",
    )
    arguments = parser.parse_args()
    if not arguments.data.is_file():
        parser.error(f"price file does not exist: {arguments.data}")
    if not arguments.config.is_file():
        parser.error(f"experiment config does not exist: {arguments.config}")

    base_config = load_experiment_config(arguments.config)
    arguments.seeds = arguments.seeds or base_config.rl_seeds
    if arguments.training_steps is None:
        arguments.training_steps = base_config.ppo.training_steps
    if arguments.training_steps <= 0:
        parser.error("--training-steps must be positive")
    if arguments.output.exists() and any(arguments.output.iterdir()) and not arguments.resume:
        parser.error(
            f"{arguments.output} is not empty; choose a new --output or use --resume"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    definitions = ablation_definitions()
    write_manifest(arguments.output, definitions, arguments, base_config)
    prices = load_price_csv(
        arguments.data,
        date_column=arguments.date_column,
        adjusted_close_column=arguments.price_column,
    )["adjusted_close"]
    records: list[dict[str, Any]] = []

    for definition in definitions:
        experiment_directory = arguments.output / str(definition["experiment"])
        completed_metrics = experiment_directory / "seed_metrics.parquet"
        if completed_metrics.is_file() and arguments.resume:
            logging.info("retaining completed %s", definition["experiment"])
        else:
            if experiment_directory.exists() and any(experiment_directory.iterdir()):
                parser.error(
                    f"{experiment_directory} is not empty; use --resume only when "
                    "seed_metrics.parquet exists"
                )
            config = make_config(
                base_config,
                definition,
                seeds=arguments.seeds,
                variants=arguments.variants,
                training_steps=arguments.training_steps,
            )
            logging.info(
                "starting %s: CVaR=%.3f risk_fraction=%.2f cost=%.1f bps",
                definition["experiment"],
                definition["cvar_confidence"],
                definition["risk_fraction"],
                config.transaction_cost_bps,
            )
            run_experiment(
                prices,
                experiment_directory,
                config=config,
                run_rl=True,
                fold_indices=arguments.folds,
            )
        records.extend(summarize_experiment(definition, experiment_directory))
        pd.DataFrame(records).to_csv(
            arguments.output / "cvar_risk_ablation_summary.csv",
            index=False,
        )

    logging.info(
        "completed %d experiments; summary written to %s",
        len(definitions),
        arguments.output / "cvar_risk_ablation_summary.csv",
    )


if __name__ == "__main__":
    main()
