import numpy as np

from caprl.risk.mixture_moments import conditional_mean, conditional_second_moment


def test_mixture_moments_match_closed_form_and_monte_carlo():
    probabilities = np.array([0.3, 0.7])
    means = np.array([-0.01, 0.02])
    variances = np.array([0.04, 0.01])
    assert conditional_mean(probabilities, means, variances) == np.dot(probabilities, means)
    expected_second = np.dot(probabilities, variances + means**2)
    assert conditional_second_moment(probabilities, means, variances) == expected_second

    rng = np.random.default_rng(5)
    states = rng.choice(2, size=400_000, p=probabilities)
    samples = rng.normal(means[states], np.sqrt(variances[states]))
    np.testing.assert_allclose(
        np.mean(samples), conditional_mean(probabilities, means, variances), atol=8e-4
    )
    np.testing.assert_allclose(np.mean(samples**2), expected_second, atol=1.5e-4)


def test_vectorized_moments_preserve_leading_shape():
    probabilities = np.array([[0.2, 0.8], [0.9, 0.1]])
    means = np.array([-0.01, 0.02])
    variances = np.array([0.04, 0.01])
    assert conditional_mean(probabilities, means, variances).shape == (2,)
    assert conditional_second_moment(probabilities, means, variances).shape == (2,)
