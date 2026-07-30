import numpy as np

from caprl.risk.calibration import (
    CalibrationEpisode,
    calibrate_cap,
    empirical_cap_risk,
    verify_non_increasing_risk,
)
from caprl.risk.exposure_cap import exposure_cap


def _episodes() -> list[CalibrationEpisode]:
    probability = np.linspace(0.0, 1.0, 120)
    means = -0.001 + 0.0025 * probability
    second_moments = 0.00045 - 0.00035 * probability
    cvars = 0.04 - 0.025 * probability
    return [
        CalibrationEpisode(means, second_moments, cvars),
        CalibrationEpisode(means[::-1], second_moments[::-1], cvars[::-1]),
    ]


def test_empirical_calibration_function_is_non_increasing():
    assert verify_non_increasing_risk(np.linspace(0.0, 0.2, 50), _episodes())


def test_bisection_satisfies_fractional_budget_and_is_reproducible():
    episodes = _episodes()
    first = calibrate_cap(episodes, risk_fraction=0.5, safety_margin=0.05)
    second = calibrate_cap(episodes, risk_fraction=0.5, safety_margin=0.05)
    assert first == second
    assert first.converged
    assert first.achieved_risk <= first.target_budget + 1e-6
    assert first.eta > 0
    assert empirical_cap_risk(first.eta, episodes) == first.achieved_risk


def test_stricter_risk_fraction_produces_lower_average_caps():
    episodes = _episodes()
    strict = calibrate_cap(episodes, risk_fraction=0.25, safety_margin=0.0)
    loose = calibrate_cap(episodes, risk_fraction=0.75, safety_margin=0.0)
    episode = episodes[0]
    strict_caps = exposure_cap(
        episode.conditional_means,
        episode.second_moments,
        episode.conditional_cvars,
        eta=strict.eta,
    )
    loose_caps = exposure_cap(
        episode.conditional_means,
        episode.second_moments,
        episode.conditional_cvars,
        eta=loose.eta,
    )
    assert strict_caps.mean() <= loose_caps.mean()

