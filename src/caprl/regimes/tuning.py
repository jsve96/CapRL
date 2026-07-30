"""Training/validation-only tuning of HMM transition persistence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from caprl.regimes.gaussian_hmm import GaussianHMM, fit_gaussian_hmm


@dataclass(frozen=True)
class PersistenceTuningResult:
    diagnostics: pd.DataFrame
    models: dict[float, GaussianHMM]


def tune_transition_stickiness(
    observations: pd.Series,
    *,
    train_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
    candidates: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0, 100.0),
    n_initializations: int = 20,
    variance_floor: float = 1e-8,
    min_state_occupancy: float = 0.03,
    max_iterations: int = 300,
    seed: int = 42,
) -> PersistenceTuningResult:
    """Fit on training data and compare causal behavior on validation data only."""
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("stickiness candidates must be non-empty and non-negative")
    if not train_index.isin(observations.index).all() or not validation_index.isin(
        observations.index
    ).all():
        raise ValueError("train and validation dates must exist in observations")
    if train_index.max() >= validation_index.min():
        raise ValueError("training dates must precede validation dates")
    context_index = observations.index[
        (observations.index >= train_index.min()) & (observations.index <= validation_index.max())
    ]
    validation_start = int(context_index.get_loc(validation_index.min()))
    context = observations.loc[context_index]
    prefix = context.iloc[:validation_start]
    records: list[dict[str, float]] = []
    models: dict[float, GaussianHMM] = {}
    for candidate in candidates:
        model = fit_gaussian_hmm(
            observations.loc[train_index],
            n_initializations=n_initializations,
            variance_floor=variance_floor,
            min_state_occupancy=min_state_occupancy,
            max_iterations=max_iterations,
            transition_stickiness=candidate,
            seed=seed,
        )
        models[candidate] = model
        probabilities = model.filter_frame(context).iloc[validation_start:]
        bull_probability = probabilities["bull_probability"]
        hard_state = (bull_probability >= 0.5).astype(int)
        spell_ids = hard_state.ne(hard_state.shift()).cumsum()
        spell_lengths = hard_state.groupby(spell_ids).size()
        validation_log_score = model.score(context) - model.score(prefix)
        records.append(
            {
                "transition_stickiness": candidate,
                "validation_log_score_per_observation": validation_log_score
                / len(validation_index),
                "annualized_switches": float(hard_state.diff().abs().sum() * 252 / len(hard_state)),
                "median_spell_days": float(spell_lengths.median()),
                "short_spell_fraction": float((spell_lengths <= 3).mean()),
                "mean_entropy": float(probabilities["posterior_entropy"].mean()),
                "ambiguous_probability_fraction": float(bull_probability.between(0.4, 0.6).mean()),
                "extreme_probability_fraction": float(
                    ((bull_probability < 0.1) | (bull_probability > 0.9)).mean()
                ),
                "bear_occupancy": float((1.0 - bull_probability).mean()),
                "bear_expected_duration": float(model.diagnostics.expected_durations[0]),
                "bull_expected_duration": float(model.diagnostics.expected_durations[1]),
            }
        )
    return PersistenceTuningResult(
        diagnostics=pd.DataFrame(records).set_index("transition_stickiness"), models=models
    )


def select_stickiness(
    diagnostics: pd.DataFrame,
    *,
    max_log_score_loss_per_observation: float = 0.01,
    max_bear_expected_duration: float | None = None,
    max_bull_expected_duration: float | None = None,
) -> float:
    """Choose the least-switching model within a validation log-score tolerance."""
    required = {"validation_log_score_per_observation", "annualized_switches", "short_spell_fraction"}
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"diagnostics are missing columns: {sorted(missing)}")
    if max_log_score_loss_per_observation < 0:
        raise ValueError("log-score tolerance must be non-negative")
    best_score = diagnostics["validation_log_score_per_observation"].max()
    eligible = diagnostics[
        diagnostics["validation_log_score_per_observation"]
        >= best_score - max_log_score_loss_per_observation
    ]
    if max_bear_expected_duration is not None:
        eligible = eligible[
            eligible["bear_expected_duration"] <= max_bear_expected_duration
        ]
    if max_bull_expected_duration is not None:
        eligible = eligible[
            eligible["bull_expected_duration"] <= max_bull_expected_duration
        ]
    if eligible.empty:
        raise ValueError("no persistence candidate satisfies the selection constraints")
    selected = eligible.sort_values(
        ["annualized_switches", "short_spell_fraction", "validation_log_score_per_observation"],
        ascending=[True, True, False],
    ).index[0]
    return float(selected)
