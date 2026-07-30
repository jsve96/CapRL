"""Long-only moving-average trend baseline."""

import pandas as pd


def moving_average_exposure(
    prices: pd.Series, *, short_window: int = 50, long_window: int = 200
) -> pd.Series:
    """Invest when the trailing short average is above the trailing long average."""
    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        raise ValueError("windows must satisfy 0 < short_window < long_window")
    short_average = prices.rolling(short_window, min_periods=short_window).mean()
    long_average = prices.rolling(long_window, min_periods=long_window).mean()
    signal = (short_average > long_average).astype(float)
    signal.loc[long_average.isna()] = 0.0
    return signal.rename("exposure")

