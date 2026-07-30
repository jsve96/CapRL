"""Classical and analytical benchmark strategies."""

from caprl.baselines.buy_hold import buy_hold_exposure
from caprl.baselines.cash import cash_exposure
from caprl.baselines.full_cap import full_cap_exposure
from caprl.baselines.hmm_threshold import hmm_threshold_exposure
from caprl.baselines.momentum import momentum_exposure
from caprl.baselines.moving_average import moving_average_exposure
from caprl.baselines.quadratic_regime import quadratic_regime_exposure

__all__ = [
    "buy_hold_exposure",
    "cash_exposure",
    "full_cap_exposure",
    "hmm_threshold_exposure",
    "momentum_exposure",
    "moving_average_exposure",
    "quadratic_regime_exposure",
]
