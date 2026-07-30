"""Conditional moments of finite Gaussian return mixtures."""

from __future__ import annotations

import numpy as np


def validate_mixture_inputs(
    probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if means.ndim != 1 or variances.shape != means.shape:
        raise ValueError("means and variances must be matching one-dimensional arrays")
    if probabilities.shape[-1:] != means.shape:
        raise ValueError("the final probability dimension must match the number of regimes")
    if not all(np.isfinite(values).all() for values in (probabilities, means, variances)):
        raise ValueError("mixture inputs must be finite")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-10):
        raise ValueError("probabilities must be non-negative and sum to one")
    if (variances <= 0).any():
        raise ValueError("variances must be positive")
    return probabilities, means, variances


def conditional_mean(
    probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    probabilities, means, _ = validate_mixture_inputs(probabilities, means, variances)
    return np.sum(probabilities * means, axis=-1)


def conditional_second_moment(
    probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    probabilities, means, variances = validate_mixture_inputs(probabilities, means, variances)
    return np.sum(probabilities * (variances + means**2), axis=-1)

