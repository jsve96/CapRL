import numpy as np

from caprl.regimes.state_ordering import reorder_parameters


def test_relabeling_orders_bear_before_bull_and_preserves_transitions():
    initial = np.array([0.8, 0.2])
    transition = np.array([[0.9, 0.1], [0.3, 0.7]])
    means = np.array([0.02, -0.01])
    variances = np.array([0.01, 0.04])
    ordered_initial, ordered_transition, ordered_means, ordered_variances = reorder_parameters(
        initial, transition, means, variances
    )
    np.testing.assert_allclose(ordered_initial, [0.2, 0.8])
    np.testing.assert_allclose(ordered_transition, [[0.7, 0.3], [0.1, 0.9]])
    np.testing.assert_allclose(ordered_means, [-0.01, 0.02])
    np.testing.assert_allclose(ordered_variances, [0.04, 0.01])

