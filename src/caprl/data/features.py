"""Causal market-feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_FEATURES = (
    "log_return_1d",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "realized_vol_20d",
    "realized_vol_60d",
    "ma_distance_50d",
    "ma_distance_200d",
)


def build_market_features(prices: pd.Series) -> pd.DataFrame:
    """Build trailing features using prices available at each row's timestamp."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price timestamps must be unique and increasing")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("prices must be non-missing and positive")

    prices = prices.astype(float)
    log_returns = np.log(prices / prices.shift(1))
    result = pd.DataFrame(index=prices.index)
    result["log_return_1d"] = log_returns
    for window in (5, 20, 60):
        result[f"momentum_{window}d"] = log_returns.rolling(window).sum()
    for window in (20, 60):
        result[f"realized_vol_{window}d"] = (
            log_returns.rolling(window).std(ddof=1) * np.sqrt(252.0)
        )
    for window in (50, 200):
        moving_average = prices.rolling(window).mean()
        result[f"ma_distance_{window}d"] = prices / moving_average - 1.0
    return result.loc[:, MARKET_FEATURES].dropna()

