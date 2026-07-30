import pandas as pd
import pytest

from caprl.baselines import buy_hold_exposure, cash_exposure, moving_average_exposure
from caprl.evaluation.backtest import run_backtest


def test_cash_and_buy_hold_are_deterministic():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)
    cash = run_backtest(prices, cash_exposure(prices))
    buy_hold = run_backtest(prices, buy_hold_exposure(prices))
    assert (cash["wealth"] == 1.0).all()
    assert buy_hold["wealth"].iloc[-1] == pytest.approx(prices.iloc[-1] / prices.iloc[0])


def test_buy_hold_accepts_a_constant_partial_exposure():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=dates)
    exposure = buy_hold_exposure(prices, exposure=0.52)
    assert (exposure == 0.52).all()


def test_buy_hold_rejects_invalid_exposure():
    prices = pd.Series([100.0, 101.0])
    with pytest.raises(ValueError, match="exposure must be in"):
        buy_hold_exposure(prices, exposure=1.01)


def test_moving_average_uses_current_and_past_prices_only():
    dates = pd.date_range("2020-01-01", periods=8)
    prices = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)
    changed_future = prices.copy()
    changed_future.iloc[6:] = 0.5
    original = moving_average_exposure(prices, short_window=2, long_window=3)
    changed = moving_average_exposure(changed_future, short_window=2, long_window=3)
    pd.testing.assert_series_equal(original.iloc[:6], changed.iloc[:6])
    assert (original.iloc[:2] == 0.0).all()
