#!/usr/bin/env python3
"""Build SAC revision controls from a frozen primary experiment without retraining PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caprl.data.features import build_market_features
from caprl.data.loader import load_price_csv
from caprl.evaluation.backtest import run_backtest
from caprl.evaluation.metrics import calculate_metrics
from caprl.evaluation.revision_controls import (
    calibrate_static_cap,
    match_constant_utilization,
    matched_fixed_exposure,
    volatility_target_utilization,
)
from caprl.evaluation.risk_metrics import calculate_risk_metrics
from caprl.regimes.gaussian_hmm import GaussianHMM
from caprl.risk.conditional import build_conditional_risk_frame
from caprl.risk.mixture_cvar import TwoStateCVarGrid
from caprl.utils.serialization import write_frame, write_json


METRICS = (
    "cumulative_net_return",
    "sharpe_ratio",
    "maximum_drawdown",
    "annual_turnover",
    "average_exposure",
    "realized_risk_over_beta",
)
PRIMARY_COMPARISONS = (
    ("full", "regime_cap_fixed_utilization"),
    ("full_cap", "static_budget_cap"),
    ("full", "volatility_target_regime_cap"),
    ("full", "validation_matched_fixed_exposure"),
    ("full", "unconstrained"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    run(args.data, args.primary, args.output, args.bootstrap_samples, args.seed)


def run(
    data_path: Path,
    primary: Path,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = _read_mapping(primary / "config.yaml")
    prices = load_price_csv(data_path)["adjusted_close"]
    features = build_market_features(prices)
    prices = prices.loc[features.index]
    old_metrics = pd.read_parquet(primary / "seed_metrics.parquet")
    validation_metrics = pd.read_parquet(primary / "validation_seed_metrics.parquet")
    new_records: list[dict[str, object]] = []
    matching_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    fold_directories = sorted(primary.glob("fold_*"))
    if not fold_directories:
        raise ValueError(f"no completed folds found in {primary}")
    for fold_directory in fold_directories:
        fold = int(fold_directory.name.split("_")[-1])
        split = _read_mapping(fold_directory / "split_definition.json")
        indices = {
            name: features.index[
                (features.index >= pd.Timestamp(values["start"]))
                & (features.index < pd.Timestamp(values["end_exclusive"]))
            ]
            for name, values in split.items()
        }
        model = GaussianHMM.load(fold_directory / "hmm_parameters.json")
        filtered = pd.read_parquet(fold_directory / "filtered_probabilities.parquet")
        cap_parameters = _read_mapping(fold_directory / "cap_parameters.json")
        grid = TwoStateCVarGrid.build(
            model.parameters.means,
            model.parameters.variances,
            confidence=float(config["cvar_confidence"]),
            grid_size=int(config["cvar_grid_size"]),
        )
        risk = build_conditional_risk_frame(
            filtered,
            model.parameters.means,
            model.parameters.variances,
            eta=float(cap_parameters["eta"]),
            confidence=float(config["cvar_confidence"]),
            cvar_grid=grid,
        )
        gamma = float(config["risk_gamma"])
        target_budget = float(cap_parameters["target_budget"])
        cost_rate = float(config["transaction_cost_bps"]) / 10_000.0
        action_indices = {name: index[:-1] for name, index in indices.items()}

        static = calibrate_static_cap(
            risk.loc[indices["calibration"], "conditional_cvar"].to_numpy(),
            target_budget=target_budget,
            gamma=gamma,
        )
        static_signal = pd.Series(static.cap, index=indices["test"])
        static_trajectory = _backtest(
            prices,
            indices["test"],
            static_signal,
            filtered,
            risk,
            cost_rate,
            effective_cap=static_signal,
        )
        _store_trajectory(output, fold, "static_budget_cap", None, static_trajectory)
        new_records.append(
            _record(fold, "static_budget_cap", None, static_trajectory, target_budget, gamma)
        )

        selected_target, validation_score = _select_volatility_target(
            prices,
            features,
            risk,
            filtered,
            indices["validation"],
            cost_rate,
        )
        adaptive_utilization = volatility_target_utilization(
            features.loc[indices["test"], "realized_vol_20d"],
            target_volatility=selected_target,
        )
        adaptive_signal = adaptive_utilization * risk.loc[indices["test"], "exposure_cap"]
        adaptive_trajectory = _backtest(
            prices, indices["test"], adaptive_signal, filtered, risk, cost_rate
        )
        _store_trajectory(
            output, fold, "volatility_target_regime_cap", None, adaptive_trajectory
        )
        new_records.append(
            _record(
                fold,
                "volatility_target_regime_cap",
                None,
                adaptive_trajectory,
                target_budget,
                gamma,
            )
        )

        fold_validation = validation_metrics[
            (validation_metrics["fold"] == fold) & (validation_metrics["strategy"] == "full")
        ]
        if fold_validation.empty:
            raise ValueError(f"missing Full PPO validation metrics for fold {fold}")
        validation_caps = risk.loc[action_indices["validation"], "exposure_cap"]
        for row in fold_validation.itertuples(index=False):
            rl_seed = int(row.seed)
            target_exposure = float(row.average_exposure)
            utilization = match_constant_utilization(target_exposure, validation_caps)
            regime_signal = utilization * risk.loc[indices["test"], "exposure_cap"]
            regime_trajectory = _backtest(
                prices, indices["test"], regime_signal, filtered, risk, cost_rate
            )
            _store_trajectory(
                output,
                fold,
                "regime_cap_fixed_utilization",
                rl_seed,
                regime_trajectory,
            )
            new_records.append(
                _record(
                    fold,
                    "regime_cap_fixed_utilization",
                    rl_seed,
                    regime_trajectory,
                    target_budget,
                    gamma,
                )
            )

            fixed_signal = matched_fixed_exposure(target_exposure, indices["test"])
            fixed_trajectory = _backtest(
                prices,
                indices["test"],
                fixed_signal,
                filtered,
                risk,
                cost_rate,
                effective_cap=pd.Series(1.0, index=indices["test"]),
            )
            _store_trajectory(
                output,
                fold,
                "validation_matched_fixed_exposure",
                rl_seed,
                fixed_trajectory,
            )
            new_records.append(
                _record(
                    fold,
                    "validation_matched_fixed_exposure",
                    rl_seed,
                    fixed_trajectory,
                    target_budget,
                    gamma,
                )
            )
            matching_records.append(
                {
                    "fold": fold,
                    "seed": rl_seed,
                    "ppo_validation_mean_exposure": target_exposure,
                    "validation_mean_regime_cap": float(validation_caps.mean()),
                    "constant_utilization": utilization,
                    "fixed_util_validation_exposure": float(
                        (utilization * validation_caps).mean()
                    ),
                    "constant_fixed_exposure": target_exposure,
                    "static_budget_cap": static.cap,
                    "static_calibration_risk": static.achieved_risk,
                    "calibration_target_budget": target_budget,
                    "adaptive_target_volatility": selected_target,
                    "adaptive_validation_sharpe": validation_score,
                }
            )
        audit_records.append(_audit_record(fold, model, grid, static, cap_parameters))

    new_metrics = pd.DataFrame(new_records)
    combined = pd.concat([old_metrics, new_metrics], ignore_index=True, sort=False)
    write_frame(output / "new_strategy_metrics.parquet", new_metrics)
    write_frame(output / "all_strategy_metrics.parquet", combined)
    pd.DataFrame(matching_records).to_csv(output / "matching_diagnostics.csv", index=False)
    pd.DataFrame(audit_records).to_csv(output / "assumption_audit.csv", index=False)
    summary = _summary(combined)
    summary.to_csv(output / "summary_table.csv", index=False)
    effects = _paired_fold_effects(combined, bootstrap_samples, seed)
    effects.to_csv(output / "paired_effects.csv", index=False)
    _make_figures(primary, output, features)
    write_json(
        output / "run_manifest.json",
        {
            "data": str(data_path),
            "primary": str(primary),
            "folds": len(fold_directories),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "adaptive_volatility_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "seed_aggregation": "within-fold median before fold bootstrap",
        },
    )


def _backtest(prices, index, signal, filtered, risk, cost_rate, *, effective_cap=None):
    trajectory = run_backtest(
        prices.loc[index], signal.loc[index], transaction_cost_rate=cost_rate
    )
    trajectory = trajectory.join(filtered).join(risk.drop(columns="exposure_cap"))
    if effective_cap is None:
        effective_cap = risk["exposure_cap"]
    trajectory["cap"] = effective_cap.loc[trajectory.index]
    trajectory["cap_binds"] = trajectory["exposure"] >= trajectory["cap"] - 1e-12
    trajectory["raw_action"] = np.divide(
        trajectory["exposure"],
        trajectory["cap"],
        out=np.full(len(trajectory), np.nan),
        where=trajectory["cap"].to_numpy() > 0,
    )
    return trajectory


def _record(fold, strategy, seed, trajectory, target_budget, gamma):
    return {
        "fold": fold,
        "strategy": strategy,
        "seed": seed,
        **calculate_metrics(trajectory),
        **calculate_risk_metrics(trajectory, target_budget=target_budget, gamma=gamma),
    }


def _store_trajectory(output, fold, strategy, seed, trajectory):
    suffix = "" if seed is None else f"_seed_{seed}"
    write_frame(output / f"fold_{fold:03d}" / "backtests" / f"{strategy}{suffix}.parquet", trajectory)


def _select_volatility_target(prices, features, risk, filtered, index, cost_rate):
    best: tuple[float, float] | None = None
    for target in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        utilization = volatility_target_utilization(
            features.loc[index, "realized_vol_20d"], target_volatility=target
        )
        trajectory = _backtest(
            prices,
            index,
            utilization * risk.loc[index, "exposure_cap"],
            filtered,
            risk,
            cost_rate,
        )
        score = calculate_metrics(trajectory)["sharpe_ratio"]
        score = -np.inf if not np.isfinite(score) else score
        candidate = (score, -target)
        if best is None or candidate > (best[1], -best[0]):
            best = (target, score)
    assert best is not None
    return best


def _audit_record(fold, model, grid, static, cap_parameters):
    means = model.parameters.means
    variances = model.parameters.variances
    second = variances + means**2
    differences = np.diff(grid.cvar_values)
    tolerance = 1e-10
    return {
        "fold": fold,
        "hmm_return_quantity": "daily_log_return",
        "cap_distribution_implemented": "Gaussian using HMM log-return parameters",
        "bear_mean": means[0],
        "bull_mean": means[1],
        "bear_volatility": np.sqrt(variances[0]),
        "bull_volatility": np.sqrt(variances[1]),
        "a1_pass": bool(means[1] > 0 > means[0] and (variances > 0).all()),
        "bear_raw_second_moment": second[0],
        "bull_raw_second_moment": second[1],
        "a2_pass": bool(second[0] >= second[1]),
        "minimum_conditional_cvar": float(grid.cvar_values.min()),
        "maximum_cvar_grid_increase": float(max(0.0, differences.max())),
        "a3_positive_nonincreasing_numeric": bool(
            (grid.cvar_values > 0).all() and (differences <= tolerance).all()
        ),
        "a3_tolerance": tolerance,
        "a7_bounded_simple_returns": False,
        "a7_reason": "untruncated Gaussian log-return model has unbounded support",
        "log_simple_distribution_consistent": False,
        "consistency_reason": "simple returns implied by Gaussian logs are shifted lognormal",
        "frictionless_theory_covers_costed_test": False,
        "hard_cap_pathwise_by_construction": True,
        "population_budget_guarantee_from_empirical_calibration": False,
        "regime_calibration_converged": bool(cap_parameters["converged"]),
        "regime_calibration_achieved_risk": float(cap_parameters["achieved_risk"]),
        "calibration_target_budget": float(cap_parameters["target_budget"]),
        "static_cap": static.cap,
        "static_calibration_achieved_risk": static.achieved_risk,
        "static_budget_feasible": static.achieved_risk <= static.target_budget + 1e-8,
    }


def _summary(metrics):
    fold_values = metrics.groupby(["fold", "strategy"], as_index=False)[list(METRICS)].median()
    records = []
    for strategy, group in fold_values.groupby("strategy"):
        for metric in METRICS:
            values = group[metric].dropna()
            if not values.empty:
                records.append(
                    {
                        "strategy": strategy,
                        "metric": metric,
                        "fold_median": values.median(),
                        "fold_q25": values.quantile(0.25),
                        "fold_q75": values.quantile(0.75),
                        "fold_mean": values.mean(),
                        "folds": len(values),
                        "mandate_exceedance_folds": int(
                            (values > 1.0).sum() if metric == "realized_risk_over_beta" else 0
                        ),
                    }
                )
    return pd.DataFrame(records)


def _paired_fold_effects(metrics, n_samples, seed):
    fold_values = metrics.groupby(["fold", "strategy"], as_index=False)[list(METRICS)].median()
    rng = np.random.default_rng(seed)
    records = []
    for left, right in PRIMARY_COMPARISONS:
        for metric in METRICS:
            paired = fold_values.pivot(index="fold", columns="strategy", values=metric)
            if left not in paired or right not in paired:
                continue
            values = paired[[left, right]].dropna()
            differences = values[left].to_numpy() - values[right].to_numpy()
            if len(differences) == 0:
                continue
            samples = np.asarray(
                [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(n_samples)]
            )
            records.append(
                {
                    "strategy": left,
                    "reference": right,
                    "metric": metric,
                    "effect_definition": "strategy minus reference",
                    "fold_aggregate": "median across seeds",
                    "sampling_unit": "chronological test fold",
                    "folds": len(differences),
                    "estimate_mean_difference": differences.mean(),
                    "ci95_lower": np.quantile(samples, 0.025),
                    "ci95_upper": np.quantile(samples, 0.975),
                    "bootstrap_samples": n_samples,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(records)


def _make_figures(primary: Path, output: Path, features: pd.DataFrame) -> None:
    """Create descriptive OOS cap/utilization decompositions from saved actor outputs."""
    import matplotlib.pyplot as plt

    rows = []
    for path in sorted(primary.glob("fold_*/backtests/full_seed_*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame.join(features[["realized_vol_20d"]], how="left")
        frame = frame[frame["cap"] > 1e-10].copy()
        rows.append(frame)
    if not rows:
        return
    data = pd.concat(rows)
    bins = pd.cut(data["bull_probability"], np.linspace(0, 1, 21), include_lowest=True)
    by_belief = data.groupby(bins, observed=True).agg(
        bull_probability=("bull_probability", "mean"),
        cap=("cap", "mean"),
        exposure=("exposure", "mean"),
        utilization=("raw_action", "mean"),
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(by_belief["bull_probability"], by_belief["cap"], label="regime cap")
    axes[0].plot(by_belief["bull_probability"], by_belief["exposure"], label="PPO exposure")
    axes[0].set(xlabel="Filtered bull probability", ylabel="Exposure", ylim=(0, 1))
    axes[0].legend(frameon=False)
    axes[1].plot(by_belief["bull_probability"], by_belief["utilization"], color="tab:green")
    axes[1].set(
        xlabel="Filtered bull probability",
        ylabel="PPO cap utilization",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(figures / "cap_policy_decomposition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hexbin(
        data["realized_vol_20d"], data["raw_action"], gridsize=35, mincnt=1, cmap="Blues"
    )
    axes[0].set(xlabel="Trailing 20-day annualized volatility", ylabel="PPO cap utilization")
    axes[1].hexbin(data["drawdown"], data["raw_action"], gridsize=35, mincnt=1, cmap="Blues")
    axes[1].set(xlabel="Portfolio drawdown", ylabel="PPO cap utilization")
    for axis in axes:
        axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figures / "utilization_associations.png", dpi=180)
    plt.close(fig)


def _read_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
