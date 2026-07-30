import numpy as np
import pandas.testing as pdt
import pytest

from caprl.synthetic.simulator import simulate_markov_returns


def test_simulation_is_reproducible_by_seed():
    first = simulate_markov_returns(100, seed=7)
    second = simulate_markov_returns(100, seed=7)
    pdt.assert_frame_equal(first, second)


def test_simulator_validates_transition_probabilities():
    with pytest.raises(ValueError, match="probability vectors"):
        simulate_markov_returns(10, transition_matrix=np.array([[0.9, 0.2], [0.1, 0.9]]))

