from pathlib import Path

import yaml

from caprl.evaluation.config import load_experiment_config, load_frozen_experiment_config
from caprl.evaluation.runner import _config_dictionary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_composed_primary_configuration_loads():
    config = load_experiment_config(PROJECT_ROOT / "configs" / "experiments" / "primary.yaml")
    assert config.train_years == 10
    assert config.hmm_initializations == 20
    assert 0 < config.risk_fraction <= 1
    assert len(config.rl_seeds) == 10
    assert len(config.variants) == 4


def test_frozen_configuration_coerces_quoted_numeric_values(tmp_path):
    original = load_experiment_config(
        PROJECT_ROOT / "configs" / "experiments" / "primary.yaml"
    )
    values = _config_dictionary(original)
    values["episode_length"] = str(values["episode_length"])
    values["risk_fraction"] = str(values["risk_fraction"])
    values["rl_seeds"] = [str(seed) for seed in values["rl_seeds"]]
    values["ppo"]["learning_rate"] = str(values["ppo"]["learning_rate"])
    values["ppo"]["training_steps"] = str(values["ppo"]["training_steps"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    loaded = load_frozen_experiment_config(path)

    assert loaded == original
