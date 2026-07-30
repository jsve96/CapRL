import numpy as np

from caprl.regimes.gaussian_hmm import GaussianHMMParameters, fit_gaussian_hmm
from caprl.synthetic.simulator import simulate_markov_returns


def test_multistart_fit_is_reproducible_and_valid():
    sample = simulate_markov_returns(
        1800,
        transition_matrix=np.array([[0.96, 0.04], [0.06, 0.94]]),
        means=np.array([0.0015, -0.0020]),
        standard_deviations=np.array([0.007, 0.020]),
        seed=123,
    )
    first = fit_gaussian_hmm(
        sample["return"], n_initializations=5, max_iterations=150, seed=99
    )
    second = fit_gaussian_hmm(
        sample["return"], n_initializations=5, max_iterations=150, seed=99
    )
    np.testing.assert_allclose(first.parameters.means, second.parameters.means)
    np.testing.assert_allclose(first.parameters.transition_matrix, second.parameters.transition_matrix)
    assert first.diagnostics.selected_initialization == second.diagnostics.selected_initialization
    assert first.parameters.means[0] < first.parameters.means[1]
    np.testing.assert_allclose(first.parameters.transition_matrix.sum(axis=1), 1.0)
    assert (first.parameters.variances > 0).all()
    assert min(first.diagnostics.state_occupancies) >= 0.03


def test_synthetic_parameters_are_recovered_with_reasonable_tolerance():
    sample = simulate_markov_returns(
        2500,
        transition_matrix=np.array([[0.97, 0.03], [0.05, 0.95]]),
        means=np.array([0.0020, -0.0030]),
        standard_deviations=np.array([0.006, 0.022]),
        seed=321,
    )
    model = fit_gaussian_hmm(
        sample["return"], n_initializations=6, max_iterations=200, seed=11
    )
    np.testing.assert_allclose(model.parameters.means, [-0.0030, 0.0020], atol=0.0025)
    np.testing.assert_allclose(np.sqrt(model.parameters.variances), [0.022, 0.006], atol=0.004)
    assert np.all(np.diag(model.parameters.transition_matrix) > 0.85)


def test_parameter_dictionary_round_trip():
    parameters = GaussianHMMParameters(
        np.array([0.4, 0.6]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([-0.01, 0.02]),
        np.array([0.04, 0.01]),
    )
    restored = GaussianHMMParameters.from_dict(parameters.to_dict())
    np.testing.assert_allclose(restored.initial_probabilities, parameters.initial_probabilities)
    np.testing.assert_allclose(restored.transition_matrix, parameters.transition_matrix)
    np.testing.assert_allclose(restored.means, parameters.means)
    np.testing.assert_allclose(restored.variances, parameters.variances)


def test_serialized_model_reproduces_predictions(tmp_path):
    sample = simulate_markov_returns(800, seed=17)
    model = fit_gaussian_hmm(sample["return"], n_initializations=3, seed=8)
    path = tmp_path / "hmm.json"
    model.save(path)
    restored = type(model).load(path)
    np.testing.assert_allclose(restored.filter(sample["return"]), model.filter(sample["return"]))
    assert restored.diagnostics.to_dict() == model.diagnostics.to_dict()
