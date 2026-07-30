"""Offline calibration of the hard-cap risk-aversion parameter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from caprl.risk.exposure_cap import exposure_cap


@dataclass(frozen=True)
class CalibrationEpisode:
    conditional_means: np.ndarray
    second_moments: np.ndarray
    conditional_cvars: np.ndarray

    def __post_init__(self) -> None:
        means = np.asarray(self.conditional_means, dtype=float)
        moments = np.asarray(self.second_moments, dtype=float)
        cvars = np.asarray(self.conditional_cvars, dtype=float)
        if means.ndim != 1 or moments.shape != means.shape or cvars.shape != means.shape:
            raise ValueError("episode risk inputs must be matching one-dimensional arrays")
        if means.size == 0:
            raise ValueError("calibration episodes must not be empty")
        if not all(np.isfinite(values).all() for values in (means, moments, cvars)):
            raise ValueError("episode risk inputs must be finite")
        if (moments <= 0).any() or (cvars < 0).any():
            raise ValueError("second moments must be positive and CVaR non-negative")
        object.__setattr__(self, "conditional_means", means)
        object.__setattr__(self, "second_moments", moments)
        object.__setattr__(self, "conditional_cvars", cvars)


@dataclass(frozen=True)
class CapCalibrationResult:
    eta: float
    achieved_risk: float
    target_budget: float
    unconstrained_cap_risk: float
    risk_fraction: float
    safety_margin: float
    converged: bool
    iterations: int
    upper_bracket: float

    def to_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


def episode_risk_charge(
    exposures: np.ndarray,
    conditional_cvars: np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    exposures = np.asarray(exposures, dtype=float)
    cvars = np.asarray(conditional_cvars, dtype=float)
    if exposures.ndim != 1 or cvars.shape != exposures.shape:
        raise ValueError("exposures and CVaR must be matching one-dimensional arrays")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not np.isfinite(exposures).all() or not np.isfinite(cvars).all():
        raise ValueError("risk-charge inputs must be finite")
    if not ((0 <= exposures) & (exposures <= 1)).all() or (cvars < 0).any():
        raise ValueError("exposures must be in [0, 1] and CVaR non-negative")
    discounts = np.power(gamma, np.arange(len(exposures), dtype=float))
    return float(np.sum(discounts * exposures * cvars))


def empirical_cap_risk(
    eta: float, episodes: Sequence[CalibrationEpisode], *, gamma: float = 1.0
) -> float:
    if not episodes:
        raise ValueError("at least one calibration episode is required")
    charges = []
    for episode in episodes:
        caps = exposure_cap(
            episode.conditional_means,
            episode.second_moments,
            episode.conditional_cvars,
            eta=eta,
        )
        charges.append(episode_risk_charge(caps, episode.conditional_cvars, gamma=gamma))
    return float(np.mean(charges))


def calibrate_cap(
    episodes: Sequence[CalibrationEpisode],
    *,
    risk_fraction: float = 0.5,
    safety_margin: float = 0.05,
    gamma: float = 1.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    initial_upper: float = 1.0,
) -> CapCalibrationResult:
    """Choose the smallest numerically resolved eta satisfying the empirical budget."""
    if not 0 < risk_fraction <= 1:
        raise ValueError("risk_fraction must be in (0, 1]")
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be in [0, 1)")
    if tolerance <= 0 or max_iterations < 1 or initial_upper <= 0:
        raise ValueError("bisection settings must be positive")
    phi_zero = empirical_cap_risk(0.0, episodes, gamma=gamma)
    target = (1.0 - safety_margin) * risk_fraction * phi_zero
    if phi_zero <= target + tolerance:
        return CapCalibrationResult(
            eta=0.0,
            achieved_risk=phi_zero,
            target_budget=target,
            unconstrained_cap_risk=phi_zero,
            risk_fraction=risk_fraction,
            safety_margin=safety_margin,
            converged=True,
            iterations=0,
            upper_bracket=0.0,
        )

    lower = 0.0
    upper = initial_upper
    upper_risk = empirical_cap_risk(upper, episodes, gamma=gamma)
    bracket_iterations = 0
    while upper_risk > target:
        lower = upper
        upper *= 2.0
        bracket_iterations += 1
        if bracket_iterations >= max_iterations or not np.isfinite(upper):
            return CapCalibrationResult(
                eta=upper,
                achieved_risk=upper_risk,
                target_budget=target,
                unconstrained_cap_risk=phi_zero,
                risk_fraction=risk_fraction,
                safety_margin=safety_margin,
                converged=False,
                iterations=bracket_iterations,
                upper_bracket=upper,
            )
        upper_risk = empirical_cap_risk(upper, episodes, gamma=gamma)

    iterations = bracket_iterations
    while iterations < max_iterations and upper - lower > tolerance:
        midpoint = 0.5 * (lower + upper)
        midpoint_risk = empirical_cap_risk(midpoint, episodes, gamma=gamma)
        if midpoint_risk <= target:
            upper = midpoint
            upper_risk = midpoint_risk
        else:
            lower = midpoint
        iterations += 1
    achieved = empirical_cap_risk(upper, episodes, gamma=gamma)
    return CapCalibrationResult(
        eta=upper,
        achieved_risk=achieved,
        target_budget=target,
        unconstrained_cap_risk=phi_zero,
        risk_fraction=risk_fraction,
        safety_margin=safety_margin,
        converged=upper - lower <= tolerance and achieved <= target + tolerance,
        iterations=iterations,
        upper_bracket=upper,
    )


def verify_non_increasing_risk(
    etas: np.ndarray, episodes: Sequence[CalibrationEpisode], *, gamma: float = 1.0
) -> bool:
    etas = np.asarray(etas, dtype=float)
    if etas.ndim != 1 or len(etas) < 2 or np.any(np.diff(etas) < 0):
        raise ValueError("etas must be an increasing one-dimensional grid")
    risks = np.asarray([empirical_cap_risk(eta, episodes, gamma=gamma) for eta in etas])
    return bool(np.all(np.diff(risks) <= 1e-12))

