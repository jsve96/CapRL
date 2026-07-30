import numpy as np
import pandas as pd

from caprl.envs.wrappers import RLVariant, make_variant_env


def _inputs():
    index = pd.date_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, 101.0, 99.0, 100.0], index=index)
    market = pd.DataFrame({"market": np.arange(4)}, index=index)
    regime = pd.DataFrame(
        {
            "bear_probability": [0.8, 0.7, 0.3, 0.2],
            "bull_probability": [0.2, 0.3, 0.7, 0.8],
            "posterior_entropy": [0.5, 0.6, 0.6, 0.5],
        },
        index=index,
    )
    risk = pd.DataFrame({"conditional_cvar": [0.03] * 4}, index=index)
    caps = pd.Series([0.2, 0.3, 0.8, 0.9], index=index)
    return prices, market, regime, risk, caps


def test_four_variants_have_expected_state_and_cap_behavior():
    prices, market, regime, risk, caps = _inputs()
    environments = {
        variant: make_variant_env(
            variant,
            prices=prices,
            market_features=market,
            regime_data=regime,
            risk_data=risk,
            caps=caps,
            episode_length=1,
        )
        for variant in RLVariant
    }
    assert environments[RLVariant.UNCONSTRAINED].policy_features.shape[1] == 1
    assert environments[RLVariant.CAP_ONLY].policy_features.shape[1] == 1
    assert environments[RLVariant.REGIME_INPUT].policy_features.shape[1] == 4
    assert environments[RLVariant.FULL].policy_features.shape[1] == 4
    assert environments[RLVariant.UNCONSTRAINED].caps.iloc[0] == 1.0
    assert environments[RLVariant.REGIME_INPUT].caps.iloc[0] == 1.0
    assert environments[RLVariant.CAP_ONLY].caps.iloc[0] == caps.iloc[0]
    assert environments[RLVariant.FULL].caps.iloc[0] == caps.iloc[0]

