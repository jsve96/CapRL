"""Artifact writers with NumPy- and pandas-aware JSON conversion."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: str | Path, values: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, default=_json_default), encoding="utf-8")


def write_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write Parquet when available and fall back explicitly to compressed CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=True)
        return path
    except ImportError:
        fallback = path.with_suffix(".csv.gz")
        frame.to_csv(fallback, compression="gzip", index=True)
        return fallback


def write_pickle(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
