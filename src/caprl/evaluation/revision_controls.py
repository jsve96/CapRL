"""Leakage-safe deterministic controls used by the SAC revision experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from caprl.risk.calibration import episode_risk_charge


@dataclass(frozen=True)
class StaticCapCalibration:
    cap: float
    achieved_risk: float
    target_budget: float
    full_exposure_risk: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calibrate_static_cap(
    conditional_cvars: np.ndarray,
    *,
    target_budget: float,
    gamma: float = 1.0,
) -> StaticCapCalibration:
    """Calibrate a constant exposure using the same additive charge as the regime cap."""
    cvars = np.asarray(conditional_cvars, dtype=float)
    if cvars.ndim != 1 or cvars.size == 0 or not np.isfinite(cvars).all():
        raise ValueError("conditional CVaRs must be a non-empty finite vector")
    if (cvars < 0).any() or target_budget < 0 or not np.isfinite(target_budget):
        raise ValueError("CVaRs and target budget must be finite and non-negative")
    full_risk = episode_risk_charge(np.ones_like(cvars), cvars, gamma=gamma)
    cap = 1.0 if full_risk == 0.0 else float(np.clip(target_budget / full_risk, 0.0, 1.0))
    achieved = episode_risk_charge(np.full_like(cvars, cap), cvars, gamma=gamma)
    return StaticCapCalibration(cap, achieved, float(target_budget), full_risk)


def match_constant_utilization(target_mean_exposure: float, caps: pd.Series) -> float:
    """Match validation exposure with a constant fraction of the regime cap."""
    values = caps.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("caps must be a non-empty finite Series in [0, 1]")
    if not np.isfinite(target_mean_exposure) or not 0 <= target_mean_exposure <= 1:
        raise ValueError("target mean exposure must be in [0, 1]")
    mean_cap = float(values.mean())
    if mean_cap == 0.0:
        return 0.0
    return float(np.clip(target_mean_exposure / mean_cap, 0.0, 1.0))


def matched_fixed_exposure(target_mean_exposure: float, index: pd.Index) -> pd.Series:
    """Return a constant, validation-selected exposure on the requested dates."""
    if not np.isfinite(target_mean_exposure) or not 0 <= target_mean_exposure <= 1:
        raise ValueError("target mean exposure must be in [0, 1]")
    return pd.Series(float(target_mean_exposure), index=index, dtype=float)


def volatility_target_utilization(
    annualized_volatility: pd.Series, *, target_volatility: float
) -> pd.Series:
    """Causal utilization min(1, target/trailing-volatility)."""
    values = annualized_volatility.astype(float)
    if values.empty or not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("trailing volatility must be finite and non-negative")
    if target_volatility <= 0 or not np.isfinite(target_volatility):
        raise ValueError("target volatility must be finite and positive")
    utilization = np.divide(
        target_volatility,
        values.to_numpy(),
        out=np.ones(len(values), dtype=float),
        where=values.to_numpy() > 0,
    )
    return pd.Series(np.clip(utilization, 0.0, 1.0), index=values.index)
