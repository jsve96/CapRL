import numpy as np
import pandas as pd

from caprl.evaluation.backtest import run_backtest
from caprl.evaluation.metrics import calculate_metrics
from caprl.evaluation.risk_metrics import calculate_risk_metrics


def test_metrics_match_simple_buy_and_hold_path():
    index = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=index)
    trajectory = run_backtest(prices, pd.Series(1.0, index=index))
    trajectory["cap"] = 1.0
    trajectory["cap_binds"] = False
    metrics = calculate_metrics(trajectory)
    np.testing.assert_allclose(metrics["cumulative_net_return"], 0.089, atol=1e-12)


def test_risk_metrics_use_discounted_additive_charge():
    trajectory = pd.DataFrame(
        {"exposure": [0.5, 1.0], "conditional_cvar": [0.02, 0.03], "net_return": [0.01, -0.02]}
    )
    metrics = calculate_risk_metrics(trajectory, target_budget=0.05, gamma=0.5)
    assert metrics["discounted_realized_risk_charge"] == 0.025
    assert metrics["realized_risk_over_beta"] == 0.5
