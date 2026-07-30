"""Long-only time-series momentum baseline."""

import numpy as np
import pandas as pd


def momentum_exposure(prices: pd.Series, *, lookback: int = 252) -> pd.Series:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    log_returns = np.log(prices / prices.shift(1))
    momentum = log_returns.rolling(lookback, min_periods=lookback).sum()
    exposure = (momentum > 0).astype(float)
    exposure.loc[momentum.isna()] = 0.0
    return exposure.rename("exposure")

