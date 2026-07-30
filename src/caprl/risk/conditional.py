"""Dated conditional-risk outputs derived from filtered regime probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from caprl.risk.exposure_cap import exposure_cap
from caprl.risk.mixture_cvar import TwoStateCVarGrid, mixture_cvar_many
from caprl.risk.mixture_moments import conditional_mean, conditional_second_moment


def build_conditional_risk_frame(
    filtered_probabilities: pd.DataFrame,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    eta: float,
    confidence: float = 0.95,
    cvar_grid: TwoStateCVarGrid | None = None,
) -> pd.DataFrame:
    """Compute conditional moments, CVaR, and cap without changing timestamps."""
    required = ["bear_probability", "bull_probability"]
    if any(column not in filtered_probabilities for column in required):
        raise ValueError(f"filtered probabilities must contain {required}")
    probabilities = filtered_probabilities.loc[:, required].to_numpy(dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    conditional_means = conditional_mean(probabilities, means, variances)
    second_moments = conditional_second_moment(probabilities, means, variances)
    if cvar_grid is None:
        cvars = mixture_cvar_many(probabilities, means, variances, confidence=confidence)
    else:
        if not np.allclose(cvar_grid.means, means) or not np.allclose(cvar_grid.variances, variances):
            raise ValueError("CVaR grid parameters do not match the requested mixture")
        if not np.isclose(cvar_grid.confidence, confidence):
            raise ValueError("CVaR grid confidence does not match the requested confidence")
        cvars = cvar_grid(probabilities[:, 1])
    caps = exposure_cap(conditional_means, second_moments, cvars, eta=eta)
    return pd.DataFrame(
        {
            "conditional_mean": conditional_means,
            "conditional_second_moment": second_moments,
            "conditional_cvar": cvars,
            "exposure_cap": caps,
        },
        index=filtered_probabilities.index,
    )

