import numpy as np

from caprl.regimes.diagnostics import HMMFitDiagnostics
from caprl.regimes.gaussian_hmm import GaussianHMM, GaussianHMMParameters


def _known_model() -> GaussianHMM:
    parameters = GaussianHMMParameters(
        initial_probabilities=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.95, 0.05], [0.04, 0.96]]),
        means=np.array([-0.01, 0.01]),
        variances=np.array([0.0004, 0.0001]),
    )
    diagnostics = HMMFitDiagnostics(0, 42, 0.0, 0, (0.5, 0.5), (20.0, 25.0), ())
    return GaussianHMM(parameters, diagnostics)


def test_filtered_probabilities_are_normalized():
    probabilities = _known_model().filter(np.array([0.01, -0.02, 0.005]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_future_observations_cannot_change_past_filtered_probabilities():
    observations = np.array([0.01, 0.005, -0.01, 0.02, -0.03, 0.01])
    changed = observations.copy()
    changed[4:] = [0.5, -0.5]
    original_probabilities = _known_model().filter(observations)
    changed_probabilities = _known_model().filter(changed)
    np.testing.assert_allclose(original_probabilities[:4], changed_probabilities[:4], atol=0.0)

