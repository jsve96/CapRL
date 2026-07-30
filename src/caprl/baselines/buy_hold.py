"""Constant-exposure buy-and-hold baselines."""

import pandas as pd


def buy_hold_exposure(prices: pd.Series, *, exposure: float = 1.0) -> pd.Series:
    """Return a constant target exposure, fully invested by default."""
    if not 0.0 <= exposure <= 1.0:
        raise ValueError("exposure must be in [0, 1]")
    return pd.Series(float(exposure), index=prices.index, name="exposure")
