"""Stable semantic ordering of otherwise exchangeable HMM state labels."""

from __future__ import annotations

import numpy as np


def mean_order_permutation(means: np.ndarray) -> np.ndarray:
    """Return states ordered from lowest (bear) to highest (bull) mean."""
    means = np.asarray(means, dtype=float)
    if means.ndim != 1 or len(means) < 2:
        raise ValueError("means must be a one-dimensional array with at least two states")
    return np.argsort(means, kind="stable")


def reorder_parameters(
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    permutation = mean_order_permutation(means)
    return (
        initial_probabilities[permutation],
        transition_matrix[np.ix_(permutation, permutation)],
        means[permutation],
        variances[permutation],
    )

