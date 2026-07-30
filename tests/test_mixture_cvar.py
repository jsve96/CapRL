from statistics import NormalDist

import numpy as np

from caprl.risk.mixture_cvar import TwoStateCVarGrid, mixture_cvar, mixture_var


def test_one_component_var_and_cvar_match_normal_formula():
    confidence = 0.95
    mean = 0.01
    standard_deviation = 0.02
    z_score = NormalDist().inv_cdf(confidence)
    expected_var = -mean + standard_deviation * z_score
    expected_cvar = -mean + standard_deviation * NormalDist().pdf(z_score) / (1 - confidence)
    probabilities = np.array([1.0])
    means = np.array([mean])
    variances = np.array([standard_deviation**2])
    np.testing.assert_allclose(
        mixture_var(probabilities, means, variances, confidence=confidence), expected_var, atol=1e-10
    )
    np.testing.assert_allclose(
        mixture_cvar(probabilities, means, variances, confidence=confidence),
        expected_cvar,
        atol=1e-10,
    )


def test_two_component_cvar_matches_seeded_monte_carlo():
    probabilities = np.array([0.35, 0.65])
    means = np.array([-0.008, 0.001])
    standard_deviations = np.array([0.025, 0.009])
    confidence = 0.95
    analytical = mixture_cvar(
        probabilities, means, standard_deviations**2, confidence=confidence
    )
    rng = np.random.default_rng(1234)
    states = rng.choice(2, size=800_000, p=probabilities)
    losses = -rng.normal(means[states], standard_deviations[states])
    empirical_var = np.quantile(losses, confidence)
    empirical = losses[losses >= empirical_var].mean()
    np.testing.assert_allclose(analytical, empirical, atol=3e-4)
    assert analytical > 0


def test_two_state_grid_interpolation_matches_direct_calculation():
    means = np.array([-0.002, 0.001])
    variances = np.array([0.02**2, 0.008**2])
    grid = TwoStateCVarGrid.build(means, variances, grid_size=501)
    bull_probabilities = np.array([0.0, 0.127, 0.5, 0.919, 1.0])
    direct = np.array(
        [
            mixture_cvar(np.array([1 - probability, probability]), means, variances)
            for probability in bull_probabilities
        ]
    )
    np.testing.assert_allclose(grid(bull_probabilities), direct, atol=2e-6)

