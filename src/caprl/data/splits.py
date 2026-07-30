"""Nested chronological walk-forward split generation."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: slice
    calibration: slice
    validation: slice
    test: slice


def make_walk_forward_folds(
    dates: pd.DatetimeIndex,
    *,
    train_years: int = 10,
    calibration_years: int = 2,
    validation_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
) -> list[WalkForwardFold]:
    """Create rolling-window folds whose intervals are disjoint and ordered."""
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("dates must be unique and increasing")
    values = (train_years, calibration_years, validation_years, test_years, step_years)
    if any(value <= 0 for value in values):
        raise ValueError("all durations must be positive")
    if dates.empty:
        return []

    start = dates[0].normalize()
    folds: list[WalkForwardFold] = []
    fold_number = 0
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        calibration_end = train_end + pd.DateOffset(years=calibration_years)
        validation_end = calibration_end + pd.DateOffset(years=validation_years)
        test_end = validation_end + pd.DateOffset(years=test_years)
        if test_end > dates[-1] + pd.Timedelta(days=1):
            break
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train=slice(start, train_end),
                calibration=slice(train_end, calibration_end),
                validation=slice(calibration_end, validation_end),
                test=slice(validation_end, test_end),
            )
        )
        fold_number += 1
        start += pd.DateOffset(years=step_years)
    return folds


def select_half_open(dates: pd.DatetimeIndex, interval: slice) -> pd.DatetimeIndex:
    """Select timestamps in [start, stop), preventing boundary overlap."""
    return dates[(dates >= interval.start) & (dates < interval.stop)]

