"""Strict structured configuration loading for PPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from caprl.agents.ppo import PPOConfig
from caprl.envs.wrappers import RLVariant


def load_ppo_config(path: str | Path) -> PPOConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("PPO configuration must be a mapping")
    if values.get("algorithm") != "ppo":
        raise ValueError("algorithm must be 'ppo'")
    if values.get("action_distribution") != "beta":
        raise ValueError("action_distribution must be 'beta'")
    if values.get("activation") != "tanh":
        raise ValueError("activation must be 'tanh'")
    fields: dict[str, Any] = {
        key: values[key]
        for key in PPOConfig.__dataclass_fields__
        if key in values
    }
    if "hidden_layers" in fields:
        fields["hidden_layers"] = tuple(fields["hidden_layers"])
    return PPOConfig(**fields)


def load_variants(path: str | Path) -> tuple[RLVariant, ...]:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not isinstance(values.get("variants"), list):
        raise ValueError("ablation configuration must contain a variant list")
    variants = tuple(RLVariant(value) for value in values["variants"])
    if len(set(variants)) != len(variants):
        raise ValueError("variant list must not contain duplicates")
    return variants

