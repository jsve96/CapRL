"""Chronological fold orchestration for the primary experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fcntl
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from caprl.agents.checkpointing import ValidationCheckpoint, save_agent, select_validation_checkpoint
from caprl.agents.evaluation import evaluate_agent
from caprl.agents.ppo import PPOConfig, train_ppo
from caprl.baselines import (
    buy_hold_exposure,
    cash_exposure,
    full_cap_exposure,
    hmm_threshold_exposure,
    momentum_exposure,
    moving_average_exposure,
    quadratic_regime_exposure,
)
from caprl.data.features import build_market_features
from caprl.data.scaling import RobustFeatureScaler
from caprl.data.splits import WalkForwardFold, make_walk_forward_folds, select_half_open
from caprl.envs.wrappers import RLVariant, make_variant_env
from caprl.evaluation.backtest import run_backtest
from caprl.evaluation.aggregate import aggregate_metrics
from caprl.evaluation.bootstrap import paired_block_bootstrap
from caprl.evaluation.metrics import calculate_metrics
from caprl.evaluation.regime_metrics import calculate_regime_metrics
from caprl.evaluation.risk_metrics import calculate_risk_metrics
from caprl.regimes.gaussian_hmm import GaussianHMM, fit_gaussian_hmm
from caprl.risk.calibration import CalibrationEpisode, calibrate_cap
from caprl.risk.conditional import build_conditional_risk_frame
from caprl.risk.mixture_cvar import TwoStateCVarGrid
from caprl.utils.hashing import hash_frame, hash_mapping, hash_source_tree
from caprl.utils.serialization import write_frame, write_json, write_pickle


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    train_years: int = 10
    calibration_years: int = 2
    validation_years: int = 1
    test_years: int = 1
    step_years: int = 1
    hmm_initializations: int = 20
    hmm_max_iterations: int = 300
    variance_floor: float = 1e-8
    min_state_occupancy: float = 0.03
    hmm_transition_stickiness: float = 0.0
    cvar_confidence: float = 0.95
    cvar_grid_size: int = 2001
    risk_fraction: float = 0.5
    safety_margin: float = 0.05
    risk_gamma: float = 1.0
    transaction_cost_bps: float = 10.0
    episode_length: int = 252
    rl_seeds: tuple[int, ...] = tuple(range(10))
    variants: tuple[str, ...] = tuple(variant.value for variant in RLVariant)
    bootstrap_samples: int = 2000
    bootstrap_block_length: int = 40
    ppo: PPOConfig = field(default_factory=PPOConfig)


@dataclass(frozen=True)
class FoldResult:
    fold: int
    output_directory: Path
    metrics: pd.DataFrame
    validation_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_experiment(
    prices: pd.Series,
    output_directory: str | Path,
    *,
    config: ExperimentConfig | None = None,
    run_rl: bool = True,
    fold_indices: tuple[int, ...] | None = None,
) -> list[FoldResult]:
    config = config or ExperimentConfig()
    features = build_market_features(prices)
    aligned_prices = prices.loc[features.index]
    folds = make_walk_forward_folds(
        features.index,
        train_years=config.train_years,
        calibration_years=config.calibration_years,
        validation_years=config.validation_years,
        test_years=config.test_years,
        step_years=config.step_years,
    )
    total_fold_count = len(folds)
    if fold_indices is not None:
        requested = set(fold_indices)
        available = {fold.fold for fold in folds}
        missing = requested.difference(available)
        if missing:
            raise ValueError(
                f"requested folds are unavailable: {sorted(missing)}; "
                f"available folds are {sorted(available)}"
            )
        folds = [fold for fold in folds if fold.fold in requested]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "config.yaml"
    if not config_path.exists():
        write_json(config_path, _config_dictionary(config))
    source_root = Path(__file__).resolve().parents[1]
    metadata_path = output_directory / "metadata.json"
    if not metadata_path.exists():
        write_json(
            metadata_path,
            {
                "python": sys.version,
                "platform": platform.platform(),
                "data_hash": hash_frame(aligned_prices),
                "config_hash": hash_mapping(_config_dictionary(config)),
                "code_hash": hash_source_tree(source_root),
                "fold_count": total_fold_count,
            },
        )
    LOGGER.info(
        "experiment start: folds=%d variants=%d seeds=%d PPO_steps=%d output=%s",
        len(folds),
        len(config.variants),
        len(config.rl_seeds) if run_rl else 0,
        config.ppo.training_steps,
        output_directory,
    )
    experiment_started = time.monotonic()
    results = []
    for position, fold in enumerate(folds, start=1):
        fold_started = time.monotonic()
        LOGGER.info(
            "fold start: fold=%03d (%d/%d) test=[%s, %s)",
            fold.fold,
            position,
            len(folds),
            fold.test.start.date(),
            fold.test.stop.date(),
        )
        result = run_fold(
            aligned_prices,
            features,
            fold,
            output_directory / f"fold_{fold.fold:03d}",
            config=config,
            run_rl=run_rl,
        )
        results.append(result)
        LOGGER.info(
            "fold complete: fold=%03d (%d/%d) elapsed=%s",
            fold.fold,
            position,
            len(folds),
            _format_duration(time.monotonic() - fold_started),
        )
    if results:
        aggregate_completed_folds(output_directory)
    LOGGER.info(
        "experiment complete: folds=%d elapsed=%s output=%s",
        len(results),
        _format_duration(time.monotonic() - experiment_started),
        output_directory,
    )
    return results


def aggregate_completed_folds(output_directory: str | Path) -> int:
    """Rebuild experiment-level metrics from every completed fold on disk."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / ".aggregate.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        metric_paths = sorted(output_directory.glob("fold_*/summary_metrics.parquet"))
        if not metric_paths:
            return 0
        aggregate = pd.concat(
            [pd.read_parquet(path) for path in metric_paths],
            ignore_index=True,
        )
        write_frame(output_directory / "seed_metrics.parquet", aggregate)
        write_frame(output_directory / "summary_metrics.parquet", aggregate_metrics(aggregate))

        validation_paths = sorted(
            output_directory.glob("fold_*/validation_summary_metrics.parquet")
        )
        if validation_paths:
            write_frame(
                output_directory / "validation_seed_metrics.parquet",
                pd.concat(
                    [pd.read_parquet(path) for path in validation_paths],
                    ignore_index=True,
                ),
            )
        return len(metric_paths)


def run_fold(
    prices: pd.Series,
    features: pd.DataFrame,
    fold: WalkForwardFold,
    output_directory: str | Path,
    *,
    config: ExperimentConfig,
    run_rl: bool = True,
    pretrained_hmm: GaussianHMM | None = None,
) -> FoldResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    indices = {
        "train": select_half_open(features.index, fold.train),
        "calibration": select_half_open(features.index, fold.calibration),
        "validation": select_half_open(features.index, fold.validation),
        "test": select_half_open(features.index, fold.test),
    }
    if any(len(index) < 2 for index in indices.values()):
        raise ValueError("every fold segment must contain at least two feature rows")
    write_json(
        output_directory / "split_definition.json",
        {
            name: {"start": index[0], "end_exclusive": interval.stop, "observations": len(index)}
            for (name, index), interval in zip(
                indices.items(), (fold.train, fold.calibration, fold.validation, fold.test), strict=True
            )
        },
    )

    scaler = RobustFeatureScaler.fit(features.loc[indices["train"]])
    scaled_features = scaler.transform(features)
    write_json(output_directory / "scaler.json", _scaler_dictionary(scaler))
    write_pickle(output_directory / "scaler.pkl", scaler)
    log_returns = features["log_return_1d"]
    model = pretrained_hmm
    if model is None:
        model = fit_gaussian_hmm(
            log_returns.loc[indices["train"]],
            n_initializations=config.hmm_initializations,
            variance_floor=config.variance_floor,
            min_state_occupancy=config.min_state_occupancy,
            transition_stickiness=config.hmm_transition_stickiness,
            max_iterations=config.hmm_max_iterations,
            seed=config.seed + fold.fold,
        )
    model.save(output_directory / "hmm_parameters.json")
    write_pickle(output_directory / "hmm_model.pkl", model)
    complete_index = features.index[
        (features.index >= fold.train.start) & (features.index < fold.test.stop)
    ]
    filtered = model.filter_frame(log_returns.loc[complete_index])
    write_frame(output_directory / "filtered_probabilities.parquet", filtered)
    cvar_grid = TwoStateCVarGrid.build(
        model.parameters.means,
        model.parameters.variances,
        confidence=config.cvar_confidence,
        grid_size=config.cvar_grid_size,
    )
    provisional_risk = build_conditional_risk_frame(
        filtered,
        model.parameters.means,
        model.parameters.variances,
        eta=0.0,
        confidence=config.cvar_confidence,
        cvar_grid=cvar_grid,
    )
    calibration_values = provisional_risk.loc[indices["calibration"]]
    episode = CalibrationEpisode(
        calibration_values["conditional_mean"].to_numpy(),
        calibration_values["conditional_second_moment"].to_numpy(),
        calibration_values["conditional_cvar"].to_numpy(),
    )
    calibration = calibrate_cap(
        [episode],
        risk_fraction=config.risk_fraction,
        safety_margin=config.safety_margin,
        gamma=config.risk_gamma,
    )
    write_json(output_directory / "cap_parameters.json", calibration.to_dict())
    risk = build_conditional_risk_frame(
        filtered,
        model.parameters.means,
        model.parameters.variances,
        eta=calibration.eta,
        confidence=config.cvar_confidence,
        cvar_grid=cvar_grid,
    )
    caps = risk["exposure_cap"]
    metrics, bootstrap_records, reference_returns = _run_baselines(
        prices,
        filtered,
        risk,
        caps,
        indices["test"],
        output_directory,
        config,
        calibration.target_budget,
    )
    validation_metrics: list[dict[str, Any]] = []
    if run_rl:
        rl_metrics, rl_bootstrap, validation_metrics = _run_rl_variants(
            prices,
            scaled_features,
            filtered,
            risk,
            caps,
            indices,
            output_directory,
            config,
            calibration.target_budget,
            reference_returns,
        )
        metrics.extend(rl_metrics)
        bootstrap_records.extend(rl_bootstrap)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.insert(0, "fold", fold.fold)
    validation_metrics_frame = pd.DataFrame(validation_metrics)
    if not validation_metrics_frame.empty:
        validation_metrics_frame.insert(0, "fold", fold.fold)
        write_frame(
            output_directory / "validation_summary_metrics.parquet",
            validation_metrics_frame,
        )
    write_frame(output_directory / "summary_metrics.parquet", metrics_frame)
    write_frame(output_directory / "bootstrap_results.parquet", pd.DataFrame(bootstrap_records))
    return FoldResult(
        fold.fold,
        output_directory,
        metrics_frame,
        validation_metrics_frame,
    )


def _run_baselines(
    prices: pd.Series,
    filtered: pd.DataFrame,
    risk: pd.DataFrame,
    caps: pd.Series,
    test_index: pd.DatetimeIndex,
    output_directory: Path,
    config: ExperimentConfig,
    target_budget: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.Series]:
    all_signals = {
        "cash": cash_exposure(prices),
        "buy_hold": buy_hold_exposure(prices),
        "moving_average": moving_average_exposure(prices),
        "momentum": momentum_exposure(prices),
        "hmm_threshold": hmm_threshold_exposure(filtered["bull_probability"]).reindex(prices.index, fill_value=0.0),
        "quadratic_regime": quadratic_regime_exposure(
            risk["conditional_mean"], risk["conditional_second_moment"]
        ).reindex(prices.index, fill_value=0.0),
        "full_cap": full_cap_exposure(caps).reindex(prices.index, fill_value=0.0),
    }
    results = []
    trajectories: dict[str, pd.DataFrame] = {}
    test_prices = prices.loc[test_index]
    for strategy, signal in all_signals.items():
        trajectory = run_backtest(
            test_prices,
            signal.loc[test_index],
            transaction_cost_rate=config.transaction_cost_bps / 10_000.0,
        )
        trajectory = _enrich_trajectory(trajectory, filtered, risk, caps)
        trajectories[strategy] = trajectory
        write_frame(output_directory / "backtests" / f"{strategy}.parquet", trajectory)
        write_frame(
            output_directory / "regime_metrics" / f"{strategy}.parquet",
            calculate_regime_metrics(trajectory),
        )
        results.append(
            _metric_record(strategy, None, trajectory, target_budget, config.risk_gamma)
        )
    reference = trajectories["full_cap"]["net_return"]
    bootstrap = [
        _bootstrap_record(strategy, None, trajectory["net_return"], reference, config)
        for strategy, trajectory in trajectories.items()
        if strategy != "full_cap"
    ]
    return results, bootstrap, reference


def _run_rl_variants(
    prices: pd.Series,
    scaled_features: pd.DataFrame,
    filtered: pd.DataFrame,
    risk: pd.DataFrame,
    caps: pd.Series,
    indices: dict[str, pd.DatetimeIndex],
    output_directory: Path,
    config: ExperimentConfig,
    target_budget: float,
    reference_returns: pd.Series,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    bootstrap = []
    validation_results = []
    for variant_value in config.variants:
        variant = RLVariant(variant_value)
        for seed in config.rl_seeds:
            training_started = time.monotonic()
            report_every = max(config.ppo.rollout_steps, config.ppo.training_steps // 10)
            next_report = report_every

            def report_progress(metric: Any) -> None:
                nonlocal next_report
                if metric.steps < next_report and metric.steps < config.ppo.training_steps:
                    return
                LOGGER.info(
                    "PPO progress: fold=%s variant=%s seed=%d steps=%d/%d (%.1f%%) "
                    "reward=%.6f policy_loss=%.6f value_loss=%.6f entropy=%.6f elapsed=%s",
                    output_directory.name.split("_")[-1],
                    variant.value,
                    seed,
                    metric.steps,
                    config.ppo.training_steps,
                    100 * metric.steps / config.ppo.training_steps,
                    metric.mean_reward,
                    metric.policy_loss,
                    metric.value_loss,
                    metric.entropy,
                    _format_duration(time.monotonic() - training_started),
                )
                while next_report <= metric.steps:
                    next_report += report_every

            LOGGER.info(
                "PPO start: fold=%s variant=%s seed=%d steps=%d",
                output_directory.name.split("_")[-1],
                variant.value,
                seed,
                config.ppo.training_steps,
            )
            training_factory = _environment_factory(
                variant,
                prices,
                scaled_features,
                filtered,
                risk,
                caps,
                indices["train"],
                config,
                training=True,
            )
            agent, training_metrics = train_ppo(
                training_factory,
                config=config.ppo,
                seed=seed,
                progress_callback=report_progress,
            )
            checkpoint_path = output_directory / "models" / f"{variant.value}_seed_{seed}.pt"
            save_agent(agent, checkpoint_path)
            validation = evaluate_agent(
                agent,
                _environment_factory(
                    variant,
                    prices,
                    scaled_features,
                    filtered,
                    risk,
                    caps,
                    indices["validation"],
                    config,
                    training=False,
                ),
                seed=seed,
            )
            validation_results.append(
                _metric_record(
                    variant.value,
                    seed,
                    validation,
                    target_budget,
                    config.risk_gamma,
                )
            )
            candidate = ValidationCheckpoint(
                checkpoint_path,
                float(validation["reward"].sum()),
                float(validation["turnover"].sum()),
                bool((validation["exposure"] <= validation["cap"] + 1e-12).all()),
            )
            select_validation_checkpoint([candidate])
            test_trajectory = evaluate_agent(
                agent,
                _environment_factory(
                    variant,
                    prices,
                    scaled_features,
                    filtered,
                    risk,
                    caps,
                    indices["test"],
                    config,
                    training=False,
                ),
                seed=seed,
            )
            write_frame(
                output_directory / "backtests" / f"{variant.value}_seed_{seed}.parquet",
                test_trajectory,
            )
            write_json(
                output_directory / "training" / f"{variant.value}_seed_{seed}.json",
                [asdict(metric) for metric in training_metrics],
            )
            results.append(
                _metric_record(
                    variant.value, seed, test_trajectory, target_budget, config.risk_gamma
                )
            )
            bootstrap.append(
                _bootstrap_record(
                    variant.value, seed, test_trajectory["net_return"], reference_returns, config
                )
            )
            LOGGER.info(
                "PPO complete: fold=%s variant=%s seed=%d elapsed=%s test_return=%.6f",
                output_directory.name.split("_")[-1],
                variant.value,
                seed,
                _format_duration(time.monotonic() - training_started),
                float(test_trajectory["wealth"].iloc[-1] - 1.0),
            )
    return results, bootstrap, validation_results


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _environment_factory(
    variant: RLVariant,
    prices: pd.Series,
    features: pd.DataFrame,
    filtered: pd.DataFrame,
    risk: pd.DataFrame,
    caps: pd.Series,
    index: pd.DatetimeIndex,
    config: ExperimentConfig,
    *,
    training: bool,
):
    def factory():
        return make_variant_env(
            variant,
            prices=prices.loc[index],
            market_features=features.loc[index],
            regime_data=filtered.loc[index],
            risk_data=risk.loc[index],
            caps=caps.loc[index],
            episode_length=config.episode_length if training else None,
            transaction_cost_rate=config.transaction_cost_bps / 10_000.0,
            random_start=training,
        )

    return factory


def _enrich_trajectory(
    trajectory: pd.DataFrame,
    filtered: pd.DataFrame,
    risk: pd.DataFrame,
    caps: pd.Series,
) -> pd.DataFrame:
    result = trajectory.join(filtered).join(risk.drop(columns="exposure_cap"))
    result["cap"] = caps.reindex(result.index)
    result["cap_binds"] = result["exposure"] >= result["cap"] - 1e-12
    result["raw_action"] = np.divide(
        result["exposure"],
        result["cap"],
        out=np.zeros(len(result)),
        where=result["cap"].to_numpy() > 0,
    )
    return result


def _metric_record(
    strategy: str,
    seed: int | None,
    trajectory: pd.DataFrame,
    target_budget: float,
    gamma: float,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "seed": seed,
        **calculate_metrics(trajectory),
        **calculate_risk_metrics(trajectory, target_budget=target_budget, gamma=gamma),
    }


def _bootstrap_record(
    strategy: str,
    seed: int | None,
    returns: pd.Series,
    reference_returns: pd.Series,
    config: ExperimentConfig,
) -> dict[str, Any]:
    if not returns.index.equals(reference_returns.index):
        raise ValueError("paired bootstrap strategies must use identical test dates")
    block_length = min(config.bootstrap_block_length, len(returns))
    result = paired_block_bootstrap(
        returns.to_numpy(),
        reference_returns.to_numpy(),
        block_length=block_length,
        n_samples=config.bootstrap_samples,
        seed=config.seed + (seed or 0),
    )
    return {
        "strategy": strategy,
        "seed": seed,
        "reference": "full_cap",
        "statistic": "mean_daily_net_return_difference",
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "samples": config.bootstrap_samples,
        "block_length": block_length,
    }


def _scaler_dictionary(scaler: RobustFeatureScaler) -> dict[str, Any]:
    return {
        "columns": scaler.columns,
        "lower": scaler.lower.to_dict(),
        "upper": scaler.upper.to_dict(),
        "center": scaler.center.to_dict(),
        "scale": scaler.scale.to_dict(),
        "fitted_start": scaler.fitted_start,
        "fitted_end": scaler.fitted_end,
    }


def _config_dictionary(config: ExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    values["ppo"]["hidden_layers"] = list(config.ppo.hidden_layers)
    return values
