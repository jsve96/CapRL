"""File-based market-data loading with an explicit canonical schema."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from caprl.data.validation import validate_market_data


def load_price_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    adjusted_close_column: str = "adjusted_close",
) -> pd.DataFrame:
    """Load daily prices from CSV; missing observations are never forward-filled."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    missing = {date_column, adjusted_close_column}.difference(data.columns)
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")

    try:
        dates = pd.to_datetime(data.pop(date_column), errors="raise", utc=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("date column contains invalid timestamps") from exc
    if dates.dt.tz is not None:
        raise ValueError("date column must be timezone-naive")
    data = data.rename(columns={adjusted_close_column: "adjusted_close"})
    data.index = pd.DatetimeIndex(dates, name="date")
    return validate_market_data(data)

