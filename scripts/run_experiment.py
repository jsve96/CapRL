#!/usr/bin/env python3
"""Run the configured walk-forward experiment from a local price CSV."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path

from caprl.data.loader import load_price_csv
from caprl.evaluation.config import load_experiment_config, load_frozen_experiment_config
from caprl.evaluation.runner import aggregate_completed_folds, run_experiment


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="CSV containing date and adjusted_close")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/primary.yaml")
    )
    parser.add_argument("--no-rl", action="store_true", help="run deterministic baselines only")
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        default=None,
        help="run only this zero-based fold; repeat to run multiple folds",
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="freeze the resolved config and metadata without running a fold",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild root metric files from completed fold directories and exit",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="adjusted_close")
    parser.add_argument(
        "--training-steps",
        type=int,
        default=None,
        help="override PPO steps per variant, seed, and fold without changing YAML",
    )
    arguments = parser.parse_args()
    if not arguments.data.is_file():
        parser.error(
            f"price file does not exist: {arguments.data}. "
            "Supply a CSV with date and adjusted_close columns."
        )
    frozen_config = arguments.output / "config.yaml"
    if arguments.aggregate_only:
        if not frozen_config.is_file():
            parser.error(f"experiment is not initialized: {frozen_config}")
        completed = aggregate_completed_folds(arguments.output)
        print(f"aggregated {completed} completed folds in {arguments.output}")
        return

    data = load_price_csv(
        arguments.data,
        date_column=arguments.date_column,
        adjusted_close_column=arguments.price_column,
    )
    if frozen_config.is_file():
        if arguments.training_steps is not None:
            parser.error(
                "--training-steps cannot override an existing experiment's frozen config"
            )
        config = load_frozen_experiment_config(frozen_config)
        print(f"loaded frozen experiment config from {frozen_config}")
    else:
        config = load_experiment_config(arguments.config)
    if arguments.training_steps is not None:
        if arguments.training_steps <= 0:
            parser.error("--training-steps must be positive")
        config = replace(
            config,
            ppo=replace(config.ppo, training_steps=arguments.training_steps),
        )
        print(f"overriding PPO training steps: {arguments.training_steps:,}")
    fold_indices = () if arguments.initialize_only else (
        tuple(arguments.fold) if arguments.fold is not None else None
    )
    results = run_experiment(
        data["adjusted_close"],
        arguments.output,
        config=config,
        run_rl=not arguments.no_rl,
        fold_indices=fold_indices,
    )
    if arguments.initialize_only:
        print(f"initialized experiment in {arguments.output}")
        return
    print(f"completed {len(results)} folds in {arguments.output}")


if __name__ == "__main__":
    main()
