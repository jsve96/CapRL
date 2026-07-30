"""Environment construction for the primary scientific ablations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from caprl.envs.trading_env import TradingEnv


class RLVariant(str, Enum):
    UNCONSTRAINED = "unconstrained"
    REGIME_INPUT = "regime_input"
    CAP_ONLY = "cap_only"
    FULL = "full"


@dataclass(frozen=True)
class VariantSpecification:
    include_regime_probabilities: bool
    use_hard_cap: bool


VARIANT_SPECIFICATIONS = {
    RLVariant.UNCONSTRAINED: VariantSpecification(False, False),
    RLVariant.REGIME_INPUT: VariantSpecification(True, False),
    RLVariant.CAP_ONLY: VariantSpecification(False, True),
    RLVariant.FULL: VariantSpecification(True, True),
}


def make_variant_env(
    variant: RLVariant | str,
    *,
    prices: pd.Series,
    market_features: pd.DataFrame,
    regime_data: pd.DataFrame,
    risk_data: pd.DataFrame,
    caps: pd.Series,
    **environment_kwargs: object,
) -> TradingEnv:
    variant = RLVariant(variant)
    specification = VARIANT_SPECIFICATIONS[variant]
    features = market_features.copy()
    if specification.include_regime_probabilities:
        regime_columns = ["bear_probability", "bull_probability", "posterior_entropy"]
        features = features.join(regime_data.loc[:, regime_columns])
    effective_caps = caps if specification.use_hard_cap else pd.Series(1.0, index=caps.index)
    return TradingEnv(
        prices,
        features,
        effective_caps,
        regime_data=regime_data,
        risk_data=risk_data,
        **environment_kwargs,
    )

