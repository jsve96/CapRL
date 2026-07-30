"""Two-state univariate Gaussian HMM fitted by multi-start EM."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from caprl.regimes.diagnostics import HMMFitAttempt, HMMFitDiagnostics
from caprl.regimes.filtering import (
    forward_log_messages,
    forward_log_probabilities,
    gaussian_log_emissions,
    logsumexp,
)
from caprl.regimes.state_ordering import reorder_parameters


@dataclass(frozen=True)
class GaussianHMMParameters:
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def to_dict(self) -> dict[str, list[float] | list[list[float]]]:
        return {
            "initial_probabilities": self.initial_probabilities.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GaussianHMMParameters":
        return cls(
            initial_probabilities=np.asarray(values["initial_probabilities"], dtype=float),
            transition_matrix=np.asarray(values["transition_matrix"], dtype=float),
            means=np.asarray(values["means"], dtype=float),
            variances=np.asarray(values["variances"], dtype=float),
        )


@dataclass(frozen=True)
class GaussianHMM:
    """Fitted model with bear state 0 and bull state 1."""

    parameters: GaussianHMMParameters
    diagnostics: HMMFitDiagnostics

    def filter(self, observations: np.ndarray | pd.Series) -> np.ndarray:
        values = _validate_observations(observations)
        parameters = self.parameters
        emissions = gaussian_log_emissions(values, parameters.means, parameters.variances)
        probabilities, _ = forward_log_probabilities(
            emissions, parameters.initial_probabilities, parameters.transition_matrix
        )
        return probabilities

    def filter_frame(self, observations: pd.Series) -> pd.DataFrame:
        """Return semantically named causal probabilities and entropy."""
        probabilities = self.filter(observations)
        entropy = -np.sum(probabilities * np.log(probabilities + np.finfo(float).eps), axis=1)
        return pd.DataFrame(
            {
                "bear_probability": probabilities[:, 0],
                "bull_probability": probabilities[:, 1],
                "most_likely_state": np.argmax(probabilities, axis=1),
                "posterior_entropy": entropy,
            },
            index=observations.index,
        )

    def score(self, observations: np.ndarray | pd.Series) -> float:
        values = _validate_observations(observations)
        parameters = self.parameters
        emissions = gaussian_log_emissions(values, parameters.means, parameters.variances)
        _, log_likelihood = forward_log_probabilities(
            emissions, parameters.initial_probabilities, parameters.transition_matrix
        )
        return log_likelihood

    def to_dict(self) -> dict[str, object]:
        return {"parameters": self.parameters.to_dict(), "diagnostics": self.diagnostics.to_dict()}

    def save(self, path: str | Path) -> None:
        """Persist fitted parameters and all initialization diagnostics as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GaussianHMM":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        diagnostics_values = values["diagnostics"]
        attempts = tuple(HMMFitAttempt(**attempt) for attempt in diagnostics_values["attempts"])
        diagnostics = HMMFitDiagnostics(
            selected_initialization=diagnostics_values["selected_initialization"],
            selected_seed=diagnostics_values["selected_seed"],
            log_likelihood=diagnostics_values["log_likelihood"],
            iterations=diagnostics_values["iterations"],
            state_occupancies=tuple(diagnostics_values["state_occupancies"]),
            expected_durations=tuple(diagnostics_values["expected_durations"]),
            attempts=attempts,
        )
        return cls(GaussianHMMParameters.from_dict(values["parameters"]), diagnostics)


def fit_gaussian_hmm(
    observations: np.ndarray | pd.Series,
    *,
    n_states: int = 2,
    n_initializations: int = 20,
    variance_floor: float = 1e-8,
    min_state_occupancy: float = 0.03,
    max_iterations: int = 300,
    tolerance: float = 1e-6,
    transition_stickiness: float = 0.0,
    seed: int = 42,
) -> GaussianHMM:
    """Fit seeded EM starts and retain the valid model with highest likelihood."""
    values = _validate_observations(observations)
    if n_states != 2:
        raise ValueError("the primary implementation currently requires exactly two states")
    if n_initializations < 1 or max_iterations < 1:
        raise ValueError("initialization and iteration counts must be positive")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    if transition_stickiness < 0 or not np.isfinite(transition_stickiness):
        raise ValueError("transition_stickiness must be finite and non-negative")
    if not 0 < min_state_occupancy < 1 / n_states:
        raise ValueError("min_state_occupancy must be between 0 and 1 / n_states")
    if len(values) < max(20, n_states * 5):
        raise ValueError("too few observations to fit a Gaussian HMM")

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(n_initializations)
    attempts: list[HMMFitAttempt] = []
    valid_results: list[tuple[float, int, int, GaussianHMMParameters, np.ndarray, int]] = []
    for initialization, child_sequence in enumerate(child_sequences):
        child_seed = int(child_sequence.generate_state(1)[0])
        try:
            parameters, occupancies, log_likelihood, iterations, converged = _fit_single_start(
                values,
                n_states=n_states,
                variance_floor=variance_floor,
                max_iterations=max_iterations,
                tolerance=tolerance,
                transition_stickiness=transition_stickiness,
                rng=np.random.default_rng(child_sequence),
            )
            reason = _invalid_reason(parameters, occupancies, log_likelihood, min_state_occupancy)
            valid = converged and reason is None
            if not converged and reason is None:
                reason = "EM did not converge"
            attempts.append(
                HMMFitAttempt(
                    initialization=initialization,
                    seed=child_seed,
                    converged=converged,
                    valid=valid,
                    iterations=iterations,
                    log_likelihood=log_likelihood,
                    minimum_occupancy=float(np.min(occupancies)),
                    reason=reason,
                )
            )
            if valid:
                valid_results.append(
                    (log_likelihood, initialization, child_seed, parameters, occupancies, iterations)
                )
        except (FloatingPointError, ValueError) as exc:
            attempts.append(
                HMMFitAttempt(
                    initialization=initialization,
                    seed=child_seed,
                    converged=False,
                    valid=False,
                    iterations=0,
                    log_likelihood=-math.inf,
                    minimum_occupancy=0.0,
                    reason=str(exc),
                )
            )
    if not valid_results:
        reasons = sorted({attempt.reason for attempt in attempts})
        raise RuntimeError(f"no valid HMM fit found; reasons={reasons}")

    log_likelihood, selected, selected_seed, raw_parameters, raw_occupancies, iterations = max(
        valid_results, key=lambda item: item[0]
    )
    permutation = np.argsort(raw_parameters.means, kind="stable")
    ordered = reorder_parameters(
        raw_parameters.initial_probabilities,
        raw_parameters.transition_matrix,
        raw_parameters.means,
        raw_parameters.variances,
    )
    parameters = GaussianHMMParameters(*ordered)
    occupancies = raw_occupancies[permutation]
    expected_durations = 1.0 / (1.0 - np.diag(parameters.transition_matrix))
    diagnostics = HMMFitDiagnostics(
        selected_initialization=selected,
        selected_seed=selected_seed,
        log_likelihood=log_likelihood,
        iterations=iterations,
        state_occupancies=tuple(float(value) for value in occupancies),
        expected_durations=tuple(float(value) for value in expected_durations),
        attempts=tuple(attempts),
    )
    return GaussianHMM(parameters=parameters, diagnostics=diagnostics)


def _validate_observations(observations: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    return values


def _initialize_parameters(
    observations: np.ndarray, n_states: int, variance_floor: float, rng: np.random.Generator
) -> GaussianHMMParameters:
    quantiles = np.linspace(0.2, 0.8, n_states)
    data_scale = max(float(np.std(observations)), math.sqrt(variance_floor))
    means = np.quantile(observations, quantiles) + rng.normal(0.0, 0.15 * data_scale, n_states)
    base_variance = max(float(np.var(observations)), variance_floor)
    variances = base_variance * rng.uniform(0.5, 1.5, size=n_states)
    diagonal = rng.uniform(0.85, 0.98, size=n_states)
    transition = np.empty((n_states, n_states), dtype=float)
    for state in range(n_states):
        off_diagonal = rng.dirichlet(np.ones(n_states - 1)) * (1.0 - diagonal[state])
        transition[state] = np.insert(off_diagonal, state, diagonal[state])
    initial = rng.dirichlet(np.ones(n_states))
    return GaussianHMMParameters(initial, transition, means, variances)


def _fit_single_start(
    observations: np.ndarray,
    *,
    n_states: int,
    variance_floor: float,
    max_iterations: int,
    tolerance: float,
    transition_stickiness: float,
    rng: np.random.Generator,
) -> tuple[GaussianHMMParameters, np.ndarray, float, int, bool]:
    parameters = _initialize_parameters(observations, n_states, variance_floor, rng)
    previous_likelihood = -math.inf
    converged = False
    occupancies = np.full(n_states, 1 / n_states)
    for iteration in range(1, max_iterations + 1):
        emissions = gaussian_log_emissions(observations, parameters.means, parameters.variances)
        alpha, log_likelihood = forward_log_messages(
            emissions, parameters.initial_probabilities, parameters.transition_matrix
        )
        beta = _backward_messages(emissions, parameters.transition_matrix)
        gamma = np.exp(alpha + beta - log_likelihood)
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_sum = _expected_transition_counts(
            alpha, beta, emissions, parameters.transition_matrix, log_likelihood
        )
        weights = gamma.sum(axis=0)
        means = (gamma * observations[:, None]).sum(axis=0) / weights
        variances = (
            gamma * (observations[:, None] - means[None, :]) ** 2
        ).sum(axis=0) / weights
        variances = np.maximum(variances, variance_floor)
        transition_counts = xi_sum + transition_stickiness * np.eye(n_states)
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        parameters = GaussianHMMParameters(gamma[0], transition, means, variances)
        occupancies = gamma.mean(axis=0)
        if np.isfinite(previous_likelihood) and abs(log_likelihood - previous_likelihood) <= tolerance * (
            1.0 + abs(previous_likelihood)
        ):
            converged = True
            break
        previous_likelihood = log_likelihood
    final_emissions = gaussian_log_emissions(
        observations, parameters.means, parameters.variances
    )
    _, final_likelihood = forward_log_messages(
        final_emissions, parameters.initial_probabilities, parameters.transition_matrix
    )
    return parameters, occupancies, final_likelihood, iteration, converged


def _backward_messages(log_emissions: np.ndarray, transition: np.ndarray) -> np.ndarray:
    time_steps, states = log_emissions.shape
    log_transition = np.log(transition)
    beta = np.zeros((time_steps, states), dtype=float)
    for time in range(time_steps - 2, -1, -1):
        beta[time] = logsumexp(
            log_transition + log_emissions[time + 1][None, :] + beta[time + 1][None, :], axis=1
        )
    return beta


def _expected_transition_counts(
    alpha: np.ndarray,
    beta: np.ndarray,
    log_emissions: np.ndarray,
    transition: np.ndarray,
    log_likelihood: float,
) -> np.ndarray:
    log_transition = np.log(transition)
    counts = np.zeros_like(transition)
    for time in range(len(log_emissions) - 1):
        log_xi = (
            alpha[time][:, None]
            + log_transition
            + log_emissions[time + 1][None, :]
            + beta[time + 1][None, :]
            - log_likelihood
        )
        counts += np.exp(log_xi)
    return counts


def _invalid_reason(
    parameters: GaussianHMMParameters,
    occupancies: np.ndarray,
    log_likelihood: float,
    min_state_occupancy: float,
) -> str | None:
    if not np.isfinite(log_likelihood):
        return "non-finite likelihood"
    arrays = (
        parameters.initial_probabilities,
        parameters.transition_matrix,
        parameters.means,
        parameters.variances,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        return "non-finite parameters"
    if (parameters.variances <= 0).any():
        return "non-positive variance"
    if (parameters.transition_matrix < 0).any() or not np.allclose(
        parameters.transition_matrix.sum(axis=1), 1.0, atol=1e-8
    ):
        return "invalid transition matrix"
    if float(np.min(occupancies)) < min_state_occupancy:
        return "state occupancy below threshold"
    return None
