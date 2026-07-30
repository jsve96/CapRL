import math

import pytest

from caprl.evaluation.accounting import portfolio_step


def test_hand_calculated_transition_and_reward():
    step = portfolio_step(
        wealth=1.0,
        previous_exposure=0.0,
        target_exposure=0.5,
        next_return=0.10,
        transaction_cost_rate=0.001,
    )
    assert step.gross_return == pytest.approx(0.05)
    assert step.turnover == pytest.approx(0.5)
    assert step.transaction_cost == pytest.approx(0.0005)
    assert step.wealth == pytest.approx(1.0495)
    assert step.reward == pytest.approx(math.log(step.wealth))


def test_constant_exposure_has_no_additional_cost():
    step = portfolio_step(
        wealth=1.0,
        previous_exposure=0.4,
        target_exposure=0.4,
        next_return=0.01,
        transaction_cost_rate=0.01,
    )
    assert step.turnover == 0.0
    assert step.transaction_cost == 0.0


def test_invalid_exposure_is_rejected():
    with pytest.raises(ValueError, match="target_exposure"):
        portfolio_step(
            wealth=1.0,
            previous_exposure=0.0,
            target_exposure=1.01,
            next_return=0.0,
            transaction_cost_rate=0.0,
        )

