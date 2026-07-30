from pathlib import Path

from caprl.agents.config import load_ppo_config, load_variants
from caprl.envs.wrappers import RLVariant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_primary_ppo_configuration_loads_strictly():
    config = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.yaml")
    assert config.hidden_layers == (64, 64)
    assert config.gamma == 1.0
    assert config.device == "cpu"


def test_all_primary_variants_are_configured():
    variants = load_variants(PROJECT_ROOT / "configs" / "experiments" / "ablations.yaml")
    assert variants == (
        RLVariant.UNCONSTRAINED,
        RLVariant.REGIME_INPUT,
        RLVariant.CAP_ONLY,
        RLVariant.FULL,
    )

