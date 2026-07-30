import numpy as np
import pandas as pd

from caprl.regimes.tuning import select_stickiness, tune_transition_stickiness
from caprl.synthetic.simulator import simulate_markov_returns


def test_stickiness_tuning_uses_ordered_training_and_validation_data():
    sample = simulate_markov_returns(700, seed=91)
    result = tune_transition_stickiness(
        sample["return"],
        train_index=sample.index[:500],
        validation_index=sample.index[600:],
        candidates=(0.0, 10.0),
        n_initializations=2,
        max_iterations=80,
        min_state_occupancy=0.01,
        seed=3,
    )
    assert result.diagnostics.index.tolist() == [0.0, 10.0]
    assert set(result.models) == {0.0, 10.0}
    assert np.isfinite(result.diagnostics.to_numpy()).all()


def test_selector_prefers_fewer_switches_within_log_score_tolerance():
    diagnostics = pd.DataFrame(
        {
            "validation_log_score_per_observation": [3.00, 2.995, 2.90],
            "annualized_switches": [20.0, 10.0, 2.0],
            "short_spell_fraction": [0.4, 0.2, 0.1],
            "bear_expected_duration": [10.0, 30.0, 200.0],
            "bull_expected_duration": [30.0, 80.0, 300.0],
        },
        index=pd.Index([0.0, 10.0, 100.0], name="transition_stickiness"),
    )
    assert (
        select_stickiness(
            diagnostics,
            max_log_score_loss_per_observation=0.01,
            max_bear_expected_duration=50,
            max_bull_expected_duration=100,
        )
        == 10.0
    )
