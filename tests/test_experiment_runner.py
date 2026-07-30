import numpy as np

from caprl.agents.ppo import PPOConfig
from caprl.evaluation.runner import ExperimentConfig, run_experiment
from caprl.synthetic.simulator import simulate_markov_returns


def test_one_command_pipeline_runs_fold_and_all_primary_variants(tmp_path):
    sample = simulate_markov_returns(
        1900,
        means=np.array([0.0015, -0.002]),
        standard_deviations=np.array([0.007, 0.02]),
        seed=88,
    )
    prices = 100.0 * (1.0 + sample["return"]).cumprod()
    config = ExperimentConfig(
        train_years=2,
        calibration_years=1,
        validation_years=1,
        test_years=1,
        step_years=5,
        hmm_initializations=2,
        hmm_max_iterations=80,
        min_state_occupancy=0.01,
        cvar_grid_size=101,
        episode_length=32,
        rl_seeds=(0,),
        bootstrap_samples=20,
        bootstrap_block_length=10,
        ppo=PPOConfig(
            hidden_layers=(8,),
            rollout_steps=8,
            training_steps=16,
            batch_size=8,
            epochs_per_update=1,
        ),
    )
    results = run_experiment(prices, tmp_path / "experiment", config=config, run_rl=True)
    assert len(results) == 1
    strategies = set(results[0].metrics["strategy"])
    assert {
            "cash",
            "buy_hold",
            "moving_average",
        "momentum",
        "hmm_threshold",
        "quadratic_regime",
        "full_cap",
        "unconstrained",
        "regime_input",
        "cap_only",
        "full",
    } == strategies
    fold_directory = results[0].output_directory
    assert (fold_directory / "cap_parameters.json").is_file()
    assert (fold_directory / "hmm_parameters.json").is_file()
    assert (fold_directory / "models" / "full_seed_0.pt").is_file()
    assert (fold_directory / "validation_summary_metrics.parquet").is_file()
    assert any(fold_directory.glob("bootstrap_results.*"))
    assert any((tmp_path / "experiment").glob("summary_metrics.*"))
    assert (tmp_path / "experiment" / "validation_seed_metrics.parquet").is_file()
