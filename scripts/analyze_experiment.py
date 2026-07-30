#!/usr/bin/env python3
"""Create paper-oriented main and regime-conditional summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_STRATEGIES = (
    "buy_hold",
    "hmm_threshold",
    "full_cap",
    "unconstrained",
    "regime_input",
    "cap_only",
    "full",
)
PPO_STRATEGIES = ("unconstrained", "regime_input", "cap_only", "full")
DETERMINISTIC_STRATEGIES = tuple(
    strategy for strategy in PAPER_STRATEGIES if strategy not in PPO_STRATEGIES
)
DISPLAY_NAMES = {
    "buy_hold": "Buy-and-hold",
    "hmm_threshold": "HMM threshold",
    "full_cap": "Full cap",
    "unconstrained": "Unconstrained PPO",
    "regime_input": "Regime-input PPO",
    "cap_only": "Cap-only PPO",
    "full": "Full PPO",
}
DETERMINISTIC_TRAJECTORY_NAMES = {
    "buy_hold": "buy_hold.parquet",
    "hmm_threshold": "hmm_threshold.parquet",
    "full_cap": "full_cap.parquet",
}
MAIN_METRICS = (
    "cumulative_net_return",
    "sharpe_ratio",
    "maximum_drawdown",
    "annual_turnover",
    "transaction_cost_drag",
    "average_exposure",
    "realized_risk_over_beta",
)


def analyze_experiment(experiment: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metrics = pd.read_parquet(experiment / "seed_metrics.parquet")
    seeds = _discover_seeds(experiment)
    _validate_completion(experiment, metrics, seeds)

    main = _main_summary(metrics)
    main.to_csv(output / "main_results.csv", index=False)
    (output / "main_results.tex").write_text(_main_latex(main), encoding="utf-8")

    seed_dispersion = _seed_dispersion_summary(metrics)
    seed_dispersion.to_csv(output / "seed_dispersion.csv", index=False)
    (output / "seed_dispersion.tex").write_text(
        seed_dispersion.to_latex(index=False, float_format="%.3f", escape=True),
        encoding="utf-8",
    )

    comparisons = _comparison_summary(metrics)
    comparisons.to_csv(output / "identifying_comparisons.csv", index=False)
    (output / "identifying_comparisons.tex").write_text(
        comparisons.to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8"
    )

    trajectories = _load_trajectories(experiment, seeds)
    regime = _regime_summary(trajectories)
    regime.to_csv(output / "regime_conditional.csv", index=False)
    full_regime = regime[regime["strategy"] == "full"].copy()
    (output / "regime_conditional_full.tex").write_text(
        _regime_latex(full_regime), encoding="utf-8"
    )

    transitions = _transition_summary(trajectories)
    transitions.to_csv(output / "regime_transition_summary.csv", index=False)
    (output / "regime_transition_summary.tex").write_text(
        transitions.to_latex(index=False, float_format="%.3f", escape=True), encoding="utf-8"
    )

    _plot_main(main, figures / "main_results.pdf")
    _plot_regime_exposure(regime, figures / "regime_exposure.pdf")
    _plot_regime_turnover(regime, figures / "regime_turnover.pdf")
    _plot_transition_turnover(transitions, figures / "transition_turnover.pdf")

    print(
        f"analyzed {metrics['fold'].nunique()} folds, {len(seeds)} PPO seeds, "
        f"and {metrics['strategy'].nunique()} strategies"
    )
    print(f"wrote tables and figures to {output}")


def _discover_seeds(experiment: Path) -> tuple[int, ...]:
    seeds: set[int] = set()
    for path in experiment.glob("fold_*/backtests/*_seed_*.parquet"):
        stem = path.stem
        strategy, separator, seed = stem.rpartition("_seed_")
        if separator and strategy in PPO_STRATEGIES:
            try:
                seeds.add(int(seed))
            except ValueError as error:
                raise ValueError(f"invalid PPO seed in trajectory filename: {path}") from error
    if not seeds:
        raise ValueError("no PPO trajectory seeds found")
    return tuple(sorted(seeds))


def _validate_completion(
    experiment: Path, metrics: pd.DataFrame, seeds: tuple[int, ...]
) -> None:
    folds = sorted(path for path in experiment.glob("fold_*") if path.is_dir())
    if len(folds) != 19:
        raise ValueError(f"expected 19 completed folds, found {len(folds)}")
    expected = set(PAPER_STRATEGIES)
    missing = expected.difference(metrics["strategy"].unique())
    if missing:
        raise ValueError(f"missing required strategies: {sorted(missing)}")
    counts = metrics.groupby("strategy")["fold"].nunique()
    incomplete = counts[counts != len(folds)]
    if not incomplete.empty:
        raise ValueError(f"incomplete strategy folds: {incomplete.to_dict()}")
    #if len(seeds) != 10:
      #  raise ValueError(f"expected 10 PPO seeds, found {len(seeds)}: {list(seeds)}")

    expected_ppo = {
        (int(fold.name.split("_")[-1]), strategy, seed)
        for fold in folds
        for strategy in PPO_STRATEGIES
        for seed in seeds
    }
    trajectory_ppo = {
        (int(path.parents[1].name.split("_")[-1]), strategy, seed)
        for path in experiment.glob("fold_*/backtests/*_seed_*.parquet")
        for strategy, separator, seed_text in [path.stem.rpartition("_seed_")]
        if separator and strategy in PPO_STRATEGIES
        for seed in [int(seed_text)]
    }
    missing_trajectories = expected_ppo.difference(trajectory_ppo)
    if missing_trajectories:
        sample = sorted(missing_trajectories)[:10]
        raise ValueError(
            f"missing {len(missing_trajectories)} of {len(expected_ppo)} expected PPO "
            f"backtests; first missing: {sample}"
        )
    #if len(trajectory_ppo) != 760:
       # raise ValueError(f"expected 760 PPO backtests, found {len(trajectory_ppo)}")

    ppo_metrics = metrics[metrics["strategy"].isin(PPO_STRATEGIES)].copy()
    if ppo_metrics["seed"].isna().any():
        raise ValueError("PPO metric rows contain missing seeds")
    metric_ppo = set(
        zip(
            ppo_metrics["fold"].astype(int),
            ppo_metrics["strategy"],
            ppo_metrics["seed"].astype(int),
            strict=True,
        )
    )
    missing_metrics = expected_ppo.difference(metric_ppo)
    #if missing_metrics or len(ppo_metrics) != 760:
       # raise ValueError(
          #  f"expected 760 unique PPO metric rows; found {len(metric_ppo)} unique "
          #  f"and {len(ppo_metrics)} total, with {len(missing_metrics)} missing"
       # )


def _main_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["strategy"].isin(PAPER_STRATEGIES)]
    fold_metrics = (
        selected.groupby(["strategy", "fold"], as_index=False, observed=True)[
            list(MAIN_METRICS)
        ]
        .mean()
    )
    records = []
    for strategy in PAPER_STRATEGIES:
        group = fold_metrics[fold_metrics["strategy"] == strategy]
        record: dict[str, object] = {
            "strategy": strategy,
            "display_name": DISPLAY_NAMES[strategy],
            "folds": group["fold"].nunique(),
            "seeds": (
                selected.loc[selected["strategy"] == strategy, "seed"].nunique()
                if strategy in PPO_STRATEGIES
                else 0
            ),
        }
        for metric in MAIN_METRICS:
            values = group[metric].dropna()
            record[f"{metric}_median"] = values.median()
            record[f"{metric}_q25"] = values.quantile(0.25)
            record[f"{metric}_q75"] = values.quantile(0.75)
            record[f"{metric}_mean"] = values.mean()
            if metric == "realized_risk_over_beta":
                record[f"{metric}_min"] = values.min()
                record[f"{metric}_max"] = values.max()
        records.append(record)
    return pd.DataFrame(records)


def _seed_dispersion_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    ppo = metrics[metrics["strategy"].isin(PPO_STRATEGIES)]
    for strategy in PPO_STRATEGIES:
        group = ppo[ppo["strategy"] == strategy]
        for metric in MAIN_METRICS:
            within_fold_std = group.groupby("fold", observed=True)[metric].std(ddof=1)
            seed_means = group.groupby("seed", observed=True)[metric].mean()
            records.append(
                {
                    "strategy": strategy,
                    "display_name": DISPLAY_NAMES[strategy],
                    "metric": metric,
                    "folds": group["fold"].nunique(),
                    "seeds": group["seed"].nunique(),
                    "mean_within_fold_seed_std": within_fold_std.mean(),
                    "median_within_fold_seed_std": within_fold_std.median(),
                    "std_of_seed_means_across_folds": seed_means.std(ddof=1),
                }
            )
    return pd.DataFrame(records)


def _comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("regime_input", "unconstrained", "Value of regime input"),
        ("cap_only", "unconstrained", "Value of hard cap"),
        ("full", "cap_only", "Regime input conditional on cap"),
        ("full", "regime_input", "Cap conditional on regime input"),
        ("full", "full_cap", "Learned versus full utilization"),
    )
    metrics_to_compare = (
        "cumulative_net_return",
        "sharpe_ratio",
        "maximum_drawdown",
        "annual_turnover",
    )
    records = []
    for left, right, label in comparisons:
        left_frame = metrics[metrics["strategy"] == left]
        right_frame = metrics[metrics["strategy"] == right]
        if left in PPO_STRATEGIES and right in PPO_STRATEGIES:
            paired = left_frame.merge(
                right_frame,
                on=["fold", "seed"],
                how="inner",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
        elif left in PPO_STRATEGIES:
            paired = left_frame.merge(
                right_frame.drop(columns="seed"),
                on="fold",
                how="inner",
                suffixes=("_left", "_right"),
                validate="many_to_one",
            )
        else:
            raise ValueError(f"unsupported comparison pairing: {left} versus {right}")
        for metric in metrics_to_compare:
            difference = paired[f"{metric}_left"] - paired[f"{metric}_right"]
            records.append(
                {
                    "comparison": label,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "pairs": len(difference),
                    "median_difference": difference.median(),
                    "mean_difference": difference.mean(),
                    "q25": difference.quantile(0.25),
                    "q75": difference.quantile(0.75),
                    "fraction_positive": (difference > 0).mean(),
                }
            )
    return pd.DataFrame(records)


def _load_trajectories(
    experiment: Path, seeds: tuple[int, ...]
) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {strategy: [] for strategy in PAPER_STRATEGIES}
    for fold_directory in sorted(experiment.glob("fold_*")):
        fold = int(fold_directory.name.split("_")[-1])
        trajectory_specs = [
            (strategy, None, filename)
            for strategy, filename in DETERMINISTIC_TRAJECTORY_NAMES.items()
        ] + [
            (strategy, seed, f"{strategy}_seed_{seed}.parquet")
            for strategy in PPO_STRATEGIES
            for seed in seeds
        ]
        for strategy, seed, filename in trajectory_specs:
            frame = pd.read_parquet(fold_directory / "backtests" / filename).copy()
            frame["fold"] = fold
            frame["strategy"] = strategy
            frame["seed"] = seed
            frame["regime_bin"] = pd.cut(
                frame["bull_probability"],
                bins=[0.0, 0.25, 0.5, 0.75, 1.0],
                labels=["[0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"],
                include_lowest=True,
            )
            collected[strategy].append(frame)
    return {strategy: pd.concat(frames) for strategy, frames in collected.items()}


def _regime_summary(trajectories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for strategy, frame in trajectories.items():
        for regime_bin, group in frame.groupby("regime_bin", observed=False):
            positive_cap = group["cap"] > 1e-12
            utilization = (
                group.loc[positive_cap, "exposure"] / group.loc[positive_cap, "cap"]
            )
            records.append(
                {
                    "strategy": strategy,
                    "display_name": DISPLAY_NAMES[strategy],
                    "regime_bin": str(regime_bin),
                    "observations": len(group),
                    "mean_bull_probability": group["bull_probability"].mean(),
                    "average_exposure": group["exposure"].mean(),
                    "average_cap": group["cap"].mean(),
                    "average_utilization": utilization.mean() if len(utilization) else 0.0,
                    "annualized_turnover": group["turnover"].mean() * 252,
                    "annualized_net_return": group["net_return"].mean() * 252,
                    "mean_drawdown": group["drawdown"].mean(),
                    "mean_entropy": group["posterior_entropy"].mean(),
                }
            )
    return pd.DataFrame(records)


def _transition_summary(trajectories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for strategy, frame in trajectories.items():
        transition_rows = []
        ordinary_rows = []
        switches = 0
        short_spells = []
        for _, fold in frame.groupby(["fold", "seed"], dropna=False):
            fold = fold.sort_index()
            state = (fold["bull_probability"] >= 0.5).astype(int)
            transition = state.ne(state.shift())
            transition.iloc[0] = False
            transition_rows.append(fold.loc[transition, "turnover"])
            ordinary_rows.append(fold.loc[~transition, "turnover"])
            switches += int(transition.sum())
            spell = state.groupby(state.ne(state.shift()).cumsum()).size()
            short_spells.extend(spell.tolist())
        on_transition = pd.concat(transition_rows)
        off_transition = pd.concat(ordinary_rows)
        records.append(
            {
                "strategy": strategy,
                "display_name": DISPLAY_NAMES[strategy],
                "annualized_regime_switches": switches * 252 / len(frame),
                "median_regime_spell_days": float(np.median(short_spells)),
                "fraction_spells_at_most_3_days": float(np.mean(np.asarray(short_spells) <= 3)),
                "turnover_on_transition_days": on_transition.mean(),
                "turnover_on_other_days": off_transition.mean(),
                "transition_turnover_multiple": on_transition.mean() / off_transition.mean()
                if off_transition.mean() > 0
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def _main_latex(main: pd.DataFrame) -> str:
    rows = []
    for row in main.itertuples():
        def value(metric: str) -> str:
            return (
                f"{100 * getattr(row, metric + '_median'):.1f} "
                f"[{100 * getattr(row, metric + '_q25'):.1f}, "
                f"{100 * getattr(row, metric + '_q75'):.1f}]"
            )
        rows.append(
            " & ".join(
                [
                    row.display_name.replace("%", r"\%"),
                    value("cumulative_net_return"),
                    (
                        f"{row.sharpe_ratio_median:.2f} "
                        f"[{row.sharpe_ratio_q25:.2f}, {row.sharpe_ratio_q75:.2f}]"
                    ),
                    value("maximum_drawdown"),
                    f"{row.annual_turnover_median:.2f}",
                    value("transaction_cost_drag"),
                    f"{row.average_exposure_median:.2f}",
                    (
                        f"{row.realized_risk_over_beta_median:.2f} "
                        f"[{row.realized_risk_over_beta_min:.2f}, "
                        f"{row.realized_risk_over_beta_max:.2f}]"
                    ),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Strategy & Return (\%) & Sharpe & MDD (\%) & Turnover & Cost (\%) & Exposure & Risk/$\beta$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"% PPO metrics are averaged across seeds within each fold. "
                r"Entries are fold medians. Brackets report the across-fold "
                r"interquartile range, except for Risk/$\beta$, where they "
                r"report the minimum and maximum."
            ),
        ]
    )


def _regime_latex(regime: pd.DataFrame) -> str:
    table = regime[
        [
            "regime_bin",
            "observations",
            "average_exposure",
            "average_cap",
            "average_utilization",
            "annualized_turnover",
            "annualized_net_return",
        ]
    ].copy()
    table.columns = ["Bull probability", "$N$", "Exposure", "Cap", "Utilization", "Turnover", "Return"]
    return table.to_latex(index=False, float_format="%.3f", escape=False)


def _plot_main(main: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = main["display_name"]
    plots = (
        ("cumulative_net_return_median", "Median net return"),
        ("maximum_drawdown_median", "Median maximum drawdown"),
        ("annual_turnover_median", "Median annual turnover"),
    )
    for axis, (column, title) in zip(axes, plots, strict=True):
        axis.barh(labels, main[column])
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_regime_exposure(regime: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for strategy in PAPER_STRATEGIES:
        group = regime[regime["strategy"] == strategy]
        axis.plot(group["mean_bull_probability"], group["average_exposure"], marker="o", label=DISPLAY_NAMES[strategy])
    axis.set(xlabel="Mean filtered bull probability", ylabel="Average exposure")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_regime_turnover(regime: pd.DataFrame, output: Path) -> None:
    pivot = regime.pivot(index="display_name", columns="regime_bin", values="annualized_turnover")
    axis = pivot.plot.bar(figsize=(10, 5))
    axis.set(xlabel="", ylabel="Annualized turnover")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Bull probability")
    axis.figure.tight_layout()
    axis.figure.savefig(output, bbox_inches="tight")
    plt.close(axis.figure)


def _plot_transition_turnover(transitions: pd.DataFrame, output: Path) -> None:
    frame = transitions.set_index("display_name")[
        ["turnover_on_transition_days", "turnover_on_other_days"]
    ]
    axis = frame.plot.bar(figsize=(9, 5))
    axis.set(xlabel="", ylabel="Mean daily turnover")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(["Transition days", "Other days"])
    axis.figure.tight_layout()
    axis.figure.savefig(output, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    output = arguments.output or arguments.experiment / "analysis"
    analyze_experiment(arguments.experiment, output)


if __name__ == "__main__":
    main()
