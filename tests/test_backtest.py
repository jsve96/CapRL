import math

import pandas as pd
import pytest

from caprl.evaluation.backtest import run_backtest


def test_action_at_t_earns_only_next_period_return():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.Series([100.0, 110.0, 88.0], index=dates)
    exposures = pd.Series([0.5, 1.0, 0.0], index=dates)
    result = run_backtest(prices, exposures)

    assert result.index.tolist() == dates[:2].tolist()
    assert result["next_timestamp"].tolist() == dates[1:].tolist()
    assert result["asset_return"].tolist() == pytest.approx([0.10, -0.20])
    assert result["net_return"].tolist() == pytest.approx([0.05, -0.20])
    assert result["wealth"].tolist() == pytest.approx([1.05, 0.84])
    assert result["reward"].sum() == pytest.approx(math.log(0.84))


def test_initial_and_switching_turnover_are_charged():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.Series([100.0, 100.0, 100.0], index=dates)
    exposures = pd.Series([0.5, 0.25, 0.0], index=dates)
    result = run_backtest(prices, exposures, transaction_cost_rate=0.01)
    assert result["turnover"].tolist() == pytest.approx([0.5, 0.25])
    assert result["transaction_cost"].tolist() == pytest.approx([0.005, 0.0025])


def test_misaligned_exposures_are_rejected():
    prices = pd.Series([100.0, 101.0], index=pd.date_range("2020-01-01", periods=2))
    exposures = pd.Series([0.0, 1.0], index=pd.date_range("2020-01-02", periods=2))
    with pytest.raises(ValueError, match="identical indices"):
        run_backtest(prices, exposures)

