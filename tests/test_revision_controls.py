import numpy as np
import pandas as pd

from caprl.evaluation.revision_controls import (
    calibrate_static_cap,
    match_constant_utilization,
    matched_fixed_exposure,
    volatility_target_utilization,
)


def test_static_cap_satisfies_same_discounted_budget():
    result = calibrate_static_cap(np.array([0.1, 0.2, 0.3]), target_budget=0.25, gamma=0.9)
    assert 0 <= result.cap <= 1
    assert result.achieved_risk <= result.target_budget + 1e-12
    assert np.isclose(result.achieved_risk, result.target_budget)


def test_static_cap_handles_zero_risk_explicitly():
    result = calibrate_static_cap(np.zeros(3), target_budget=0.0)
    assert result.cap == 1.0
    assert result.achieved_risk == 0.0


def test_validation_matching_and_zero_cap_cases():
    caps = pd.Series([0.2, 0.4])
    assert np.isclose(match_constant_utilization(0.15, caps), 0.5)
    assert match_constant_utilization(0.5, pd.Series([0.0, 0.0])) == 0.0
    fixed = matched_fixed_exposure(0.3, caps.index)
    assert fixed.tolist() == [0.3, 0.3]


def test_volatility_rule_is_bounded_and_uses_same_timestamps():
    volatility = pd.Series([0.0, 0.1, 0.4], index=pd.date_range("2020-01-01", periods=3))
    result = volatility_target_utilization(volatility, target_volatility=0.2)
    assert result.index.equals(volatility.index)
    assert result.tolist() == [1.0, 1.0, 0.5]
