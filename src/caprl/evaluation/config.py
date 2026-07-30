"""Composition of the repository's structured experiment configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from caprl.agents.config import load_ppo_config, load_variants
from caprl.agents.ppo import PPOConfig
from caprl.evaluation.runner import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    primary = _mapping(path)
    data = _mapping((path.parent / primary["data_config"]).resolve())
    hmm = _mapping((path.parent / primary["hmm_config"]).resolve())
    cap = _mapping((path.parent / primary["cap_config"]).resolve())
    ppo_path = (path.parent / primary["agent_config"]).resolve()
    ppo = load_ppo_config(ppo_path)
    splits = data["splits"]
    environment = primary["environment"]
    evaluation = primary["evaluation"]
    ablations_path = path.parent / "ablations.yaml"
    variants = load_variants(ablations_path) if ablations_path.exists() else ()
    return ExperimentConfig(
        seed=int(primary["seed"]),
        train_years=int(splits["train_years"]),
        calibration_years=int(splits["calibration_years"]),
        validation_years=int(splits["validation_years"]),
        test_years=int(splits["test_years"]),
        step_years=int(splits["step_years"]),
        hmm_initializations=int(hmm["n_initializations"]),
        variance_floor=float(hmm["variance_floor"]),
        min_state_occupancy=float(hmm["min_state_occupancy"]),
        hmm_transition_stickiness=float(hmm.get("transition_stickiness", 0.0)),
        cvar_confidence=float(cap["cvar_confidence"]),
        risk_fraction=float(cap["risk_fraction"]),
        safety_margin=float(cap["safety_margin"]),
        risk_gamma=float(cap["gamma"]),
        transaction_cost_bps=float(environment["transaction_cost_bps"]),
        episode_length=int(environment["episode_length"]),
        rl_seeds=tuple(range(int(evaluation["seeds"]))),
        variants=tuple(variant.value for variant in variants),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_block_length=int(evaluation["bootstrap_block_length"]),
        ppo=ppo,
    )


def load_frozen_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the fully resolved configuration stored in an experiment directory."""
    values = _mapping(Path(path))
    ppo_values = dict(values.pop("ppo"))
    ppo_integer_fields = {
        "batch_size",
        "epochs_per_update",
        "rollout_steps",
        "training_steps",
    }
    ppo_float_fields = {
        "learning_rate",
        "ppo_clip",
        "gae_lambda",
        "gamma",
        "entropy_coefficient",
        "value_coefficient",
        "max_gradient_norm",
    }
    for field in ppo_integer_fields:
        ppo_values[field] = int(ppo_values[field])
    for field in ppo_float_fields:
        ppo_values[field] = float(ppo_values[field])
    ppo_values["hidden_layers"] = tuple(int(width) for width in ppo_values["hidden_layers"])

    integer_fields = {
        "seed",
        "train_years",
        "calibration_years",
        "validation_years",
        "test_years",
        "step_years",
        "hmm_initializations",
        "hmm_max_iterations",
        "cvar_grid_size",
        "episode_length",
        "bootstrap_samples",
        "bootstrap_block_length",
    }
    float_fields = {
        "variance_floor",
        "min_state_occupancy",
        "hmm_transition_stickiness",
        "cvar_confidence",
        "risk_fraction",
        "safety_margin",
        "risk_gamma",
        "transaction_cost_bps",
    }
    for field in integer_fields:
        values[field] = int(values[field])
    for field in float_fields:
        values[field] = float(values[field])
    values["rl_seeds"] = tuple(int(seed) for seed in values["rl_seeds"])
    values["variants"] = tuple(str(variant) for variant in values["variants"])
    return ExperimentConfig(ppo=PPOConfig(**ppo_values), **values)


def _mapping(path: Path) -> dict[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"configuration {path} must be a mapping")
    return values
