"""Regime-dependent analytical hard exposure cap."""

from __future__ import annotations

import numpy as np


def exposure_cap(
    conditional_means: np.ndarray | float,
    second_moments: np.ndarray | float,
    conditional_cvars: np.ndarray | float,
    *,
    eta: float,
) -> np.ndarray:
    means = np.asarray(conditional_means, dtype=float)
    moments = np.asarray(second_moments, dtype=float)
    cvars = np.asarray(conditional_cvars, dtype=float)
    try:
        means, moments, cvars = np.broadcast_arrays(means, moments, cvars)
    except ValueError as exc:
        raise ValueError("cap inputs must be broadcast-compatible") from exc
    if not all(np.isfinite(values).all() for values in (means, moments, cvars)):
        raise ValueError("cap inputs must be finite")
    if (moments <= 0).any():
        raise ValueError("second moments must be positive")
    if (cvars < 0).any():
        raise ValueError("conditional CVaR must be non-negative")
    if eta < 0 or not np.isfinite(eta):
        raise ValueError("eta must be finite and non-negative")
    raw_cap = np.maximum(means - eta * cvars, 0.0) / moments
    return np.clip(raw_cap, 0.0, 1.0)


def cap_from_probabilities(
    probabilities: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    conditional_cvars: np.ndarray,
    *,
    eta: float,
) -> np.ndarray:
    from caprl.risk.mixture_moments import conditional_mean, conditional_second_moment

    return exposure_cap(
        conditional_mean(probabilities, means, variances),
        conditional_second_moment(probabilities, means, variances),
        conditional_cvars,
        eta=eta,
    )

