"""Analytical full-cap strategy with cap utilization fixed at one."""

import numpy as np
import pandas as pd


def full_cap_exposure(caps: pd.Series | np.ndarray, index: pd.Index | None = None) -> pd.Series:
    """Return target exposure w_t = w_cap(pi_t)."""
    if isinstance(caps, pd.Series):
        values = caps.to_numpy(dtype=float)
        output_index = caps.index
    else:
        values = np.asarray(caps, dtype=float)
        output_index = index
    if values.ndim != 1 or not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("caps must be a finite one-dimensional sequence in [0, 1]")
    if output_index is None:
        output_index = pd.RangeIndex(len(values))
    if len(output_index) != len(values):
        raise ValueError("index length must match caps")
    return pd.Series(values.copy(), index=output_index, name="exposure")

