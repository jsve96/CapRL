"""Deterministic single-asset portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PortfolioStep:
    """Result of holding a target exposure over one subsequent return."""

    gross_return: float
    transaction_cost: float
    net_return: float
    net_factor: float
    reward: float
    wealth: float
    exposure: float
    previous_exposure: float
    turnover: float


def portfolio_step(
    *,
    wealth: float,
    previous_exposure: float,
    target_exposure: float,
    next_return: float,
    transaction_cost_rate: float,
) -> PortfolioStep:
    """Apply exposure selected at time t to the return from t to t+1."""
    if wealth <= 0 or not math.isfinite(wealth):
        raise ValueError("wealth must be finite and positive")
    if not 0.0 <= previous_exposure <= 1.0:
        raise ValueError("previous_exposure must be in [0, 1]")
    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must be in [0, 1]")
    if transaction_cost_rate < 0 or not math.isfinite(transaction_cost_rate):
        raise ValueError("transaction_cost_rate must be finite and non-negative")
    if not math.isfinite(next_return):
        raise ValueError("next_return must be finite")

    turnover = abs(target_exposure - previous_exposure)
    gross_return = target_exposure * next_return
    transaction_cost = transaction_cost_rate * turnover
    net_return = gross_return - transaction_cost
    net_factor = 1.0 + net_return
    if net_factor <= 0:
        raise ValueError("portfolio transition has a non-positive net factor")

    return PortfolioStep(
        gross_return=gross_return,
        transaction_cost=transaction_cost,
        net_return=net_return,
        net_factor=net_factor,
        reward=math.log(net_factor),
        wealth=wealth * net_factor,
        exposure=target_exposure,
        previous_exposure=previous_exposure,
        turnover=turnover,
    )

