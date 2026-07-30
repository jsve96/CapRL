"""Filtered HMM probability-threshold baseline."""

import pandas as pd


def hmm_threshold_exposure(
    bull_probability: pd.Series, *, threshold: float = 0.5
) -> pd.Series:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if bull_probability.isna().any() or not bull_probability.between(0, 1).all():
        raise ValueError("bull probabilities must be non-missing and in [0, 1]")
    return (bull_probability >= threshold).astype(float).rename("exposure")

