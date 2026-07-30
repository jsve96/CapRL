import pandas as pd

from caprl.evaluation.aggregate import aggregate_metrics


def test_seed_metrics_are_aggregated_with_median_and_iqr():
    metrics = pd.DataFrame(
        {
            "strategy": ["ppo", "ppo", "cash"],
            "seed": [0, 1, None],
            "fold": [0, 0, 0],
            "sharpe_ratio": [1.0, 3.0, 0.0],
        }
    )
    result = aggregate_metrics(metrics)
    ppo = result[(result["strategy"] == "ppo") & (result["metric"] == "sharpe_ratio")].iloc[0]
    assert ppo["median"] == 2.0
    assert ppo["q25"] == 1.5
    assert ppo["q75"] == 2.5

