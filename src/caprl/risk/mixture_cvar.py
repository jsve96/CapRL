"""VaR and CVaR for conditional Gaussian-mixture losses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np

from caprl.risk.mixture_moments import validate_mixture_inputs


_STANDARD_NORMAL = NormalDist()


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def mixture_loss_cdf(
    loss: float, probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> float:
    probabilities, means, variances = validate_mixture_inputs(probabilities, means, variances)
    if probabilities.ndim != 1:
        raise ValueError("mixture_loss_cdf expects one probability vector")
    standard_deviations = np.sqrt(variances)
    values = [
        weight * _STANDARD_NORMAL.cdf((loss + mean) / standard_deviation)
        for weight, mean, standard_deviation in zip(
            probabilities, means, standard_deviations, strict=True
        )
    ]
    return float(sum(values))


def mixture_var(
    probabilities: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    confidence: float = 0.95,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Solve the mixture-loss quantile with safeguarded bracketed bisection."""
    probabilities, means, variances = validate_mixture_inputs(probabilities, means, variances)
    if probabilities.ndim != 1:
        raise ValueError("mixture_var expects one probability vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    standard_deviations = np.sqrt(variances)
    lower = float(np.min(-means - 8.0 * standard_deviations))
    upper = float(np.max(-means + 8.0 * standard_deviations))
    expansion = max(1.0, upper - lower)
    while mixture_loss_cdf(lower, probabilities, means, variances) > confidence:
        lower -= expansion
        expansion *= 2.0
    expansion = max(1.0, upper - lower)
    while mixture_loss_cdf(upper, probabilities, means, variances) < confidence:
        upper += expansion
        expansion *= 2.0
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        if mixture_loss_cdf(midpoint, probabilities, means, variances) < confidence:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance:
            break
    return 0.5 * (lower + upper)


def mixture_cvar(
    probabilities: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    confidence: float = 0.95,
) -> float:
    """Return expected loss conditional on exceeding mixture VaR."""
    probabilities, means, variances = validate_mixture_inputs(probabilities, means, variances)
    if probabilities.ndim != 1:
        raise ValueError("mixture_cvar expects one probability vector")
    value_at_risk = mixture_var(
        probabilities, means, variances, confidence=confidence
    )
    standard_deviations = np.sqrt(variances)
    tail_total = 0.0
    for weight, mean, standard_deviation in zip(
        probabilities, means, standard_deviations, strict=True
    ):
        z_score = (value_at_risk + mean) / standard_deviation
        tail_total += weight * (
            -mean * (1.0 - _STANDARD_NORMAL.cdf(z_score))
            + standard_deviation * normal_pdf(z_score)
        )
    return float(tail_total / (1.0 - confidence))


def mixture_cvar_many(
    probabilities: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    confidence: float = 0.95,
) -> np.ndarray:
    probabilities, means, variances = validate_mixture_inputs(probabilities, means, variances)
    if probabilities.ndim == 1:
        return np.asarray(mixture_cvar(probabilities, means, variances, confidence=confidence))
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    values = [mixture_cvar(row, means, variances, confidence=confidence) for row in flat]
    return np.asarray(values).reshape(probabilities.shape[:-1])


@dataclass(frozen=True)
class TwoStateCVarGrid:
    """Dense deterministic interpolation cache indexed by bull probability."""

    bull_probabilities: np.ndarray
    cvar_values: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    confidence: float

    @classmethod
    def build(
        cls,
        means: np.ndarray,
        variances: np.ndarray,
        *,
        confidence: float = 0.95,
        grid_size: int = 2001,
    ) -> "TwoStateCVarGrid":
        if grid_size < 2:
            raise ValueError("grid_size must be at least two")
        means = np.asarray(means, dtype=float)
        variances = np.asarray(variances, dtype=float)
        bull_probabilities = np.linspace(0.0, 1.0, grid_size)
        probabilities = np.column_stack((1.0 - bull_probabilities, bull_probabilities))
        values = mixture_cvar_many(
            probabilities, means, variances, confidence=confidence
        )
        return cls(bull_probabilities, values, means.copy(), variances.copy(), confidence)

    def __call__(self, bull_probability: np.ndarray | float) -> np.ndarray:
        values = np.asarray(bull_probability, dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("bull probabilities must be finite and in [0, 1]")
        return np.interp(values, self.bull_probabilities, self.cvar_values)

