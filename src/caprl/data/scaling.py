"""Training-only robust scaling for unstable market features."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustFeatureScaler:
    """Winsorize and scale using statistics learned from one training interval."""

    columns: tuple[str, ...]
    lower: pd.Series
    upper: pd.Series
    center: pd.Series
    scale: pd.Series
    fitted_start: pd.Timestamp
    fitted_end: pd.Timestamp

    @classmethod
    def fit(
        cls,
        training_features: pd.DataFrame,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> "RobustFeatureScaler":
        if training_features.empty:
            raise ValueError("training_features must not be empty")
        if not isinstance(training_features.index, pd.DatetimeIndex):
            raise TypeError("training_features must use a DatetimeIndex")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("winsorization quantiles must satisfy 0 <= lower < upper <= 1")
        values = training_features.astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("training_features must be finite")
        lower = values.quantile(lower_quantile)
        upper = values.quantile(upper_quantile)
        clipped = values.clip(lower=lower, upper=upper, axis="columns")
        center = clipped.median()
        scale = clipped.quantile(0.75) - clipped.quantile(0.25)
        scale = scale.mask(scale <= np.finfo(float).eps, 1.0)
        return cls(
            columns=tuple(values.columns),
            lower=lower,
            upper=upper,
            center=center,
            scale=scale,
            fitted_start=values.index.min(),
            fitted_end=values.index.max(),
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if tuple(features.columns) != self.columns:
            raise ValueError("feature columns or their order differ from the fitted scaler")
        values = features.astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("features must be finite")
        clipped = values.clip(lower=self.lower, upper=self.upper, axis="columns")
        return (clipped - self.center) / self.scale

    def fit_transform(self, training_features: pd.DataFrame) -> pd.DataFrame:
        """Transform data after construction through ``fit`` for API symmetry."""
        return self.transform(training_features)

