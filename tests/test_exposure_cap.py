import numpy as np

from caprl.risk.exposure_cap import cap_from_probabilities, exposure_cap
from caprl.risk.mixture_cvar import mixture_cvar_many


def test_cap_is_bounded_zero_when_risk_adjusted_mean_is_non_positive():
    caps = exposure_cap(
        np.array([-0.01, 0.01, 10.0]),
        np.array([0.02, 0.02, 0.02]),
        np.array([0.03, 0.03, 0.03]),
        eta=1.0,
    )
    np.testing.assert_allclose(caps, [0.0, 0.0, 1.0])


def test_increasing_eta_weakly_lowers_every_cap():
    means = np.linspace(-0.002, 0.002, 101)
    moments = np.linspace(0.0005, 0.0001, 101)
    cvars = np.linspace(0.04, 0.02, 101)
    low_eta = exposure_cap(means, moments, cvars, eta=0.01)
    high_eta = exposure_cap(means, moments, cvars, eta=0.05)
    assert np.all(high_eta <= low_eta)


def test_primary_two_state_cap_is_non_decreasing_in_bull_probability():
    bull = np.linspace(0.0, 1.0, 501)
    probabilities = np.column_stack((1 - bull, bull))
    means = np.array([-0.0015, 0.0010])
    variances = np.array([0.022**2, 0.008**2])
    cvars = mixture_cvar_many(probabilities, means, variances)
    caps = cap_from_probabilities(probabilities, means, variances, cvars, eta=0.025)
    assert np.all(np.diff(caps) >= -1e-12)
    assert ((0 <= caps) & (caps <= 1)).all()

