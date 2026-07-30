import pandas as pd

from caprl.baselines import (
    hmm_threshold_exposure,
    momentum_exposure,
    quadratic_regime_exposure,
)


def test_momentum_has_no_signal_before_full_lookback():
    prices = pd.Series([100, 101, 102, 103], index=pd.date_range("2020-01-01", periods=4))
    exposure = momentum_exposure(prices, lookback=2)
    assert exposure.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_hmm_threshold_uses_filtered_probability_directly():
    probability = pd.Series([0.49, 0.5, 0.9])
    assert hmm_threshold_exposure(probability).tolist() == [0.0, 1.0, 1.0]


def test_quadratic_regime_target_is_long_only_and_bounded():
    means = pd.Series([-0.1, 0.1, 2.0])
    second_moments = pd.Series([1.0, 1.0, 1.0])
    assert quadratic_regime_exposure(means, second_moments).tolist() == [0.0, 0.1, 1.0]

