"""Validation rules for daily single-asset market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("adjusted_close",)


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy without silently repairing market data."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("market data must use a DatetimeIndex")
    if data.index.tz is not None:
        raise ValueError("market data timestamps must be timezone-naive")
    if not data.index.is_monotonic_increasing:
        raise ValueError("market data timestamps must be increasing")
    if not data.index.is_unique:
        raise ValueError("market data timestamps must be unique")
    missing_columns = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("market data must not be empty")

    result = data.copy()
    numeric_columns = [column for column in result.columns if column != "symbol"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"column {column!r} contains non-finite values")
    if (result["adjusted_close"] <= 0).any():
        raise ValueError("adjusted_close must be positive")
    if "volume" in result and (result["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    return result

