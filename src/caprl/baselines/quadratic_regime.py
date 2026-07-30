"""Frictionless quadratic regime target exposure."""

import numpy as np
import pandas as pd


def quadratic_regime_exposure(
    conditional_mean: pd.Series, conditional_second_moment: pd.Series
) -> pd.Series:
    if not conditional_mean.index.equals(conditional_second_moment.index):
        raise ValueError("conditional moments must have identical indices")
    if (conditional_second_moment <= 0).any():
        raise ValueError("conditional second moments must be positive")
    values = np.clip(conditional_mean / conditional_second_moment, 0.0, 1.0)
    return pd.Series(values, index=conditional_mean.index, name="exposure")

