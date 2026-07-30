"""PPO checkpoint persistence and validation-only model selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from caprl.agents.ppo import PPOAgent


@dataclass(frozen=True)
class ValidationCheckpoint:
    path: Path
    net_log_growth: float
    turnover: float
    mandate_satisfied: bool = True


def save_agent(agent: PPOAgent, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), path)


def load_agent(path: str | Path, *, map_location: str = "cpu") -> PPOAgent:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    payload["config"]["device"] = map_location
    return PPOAgent.from_state_dict(payload)


def select_validation_checkpoint(
    checkpoints: list[ValidationCheckpoint],
) -> ValidationCheckpoint:
    eligible = [checkpoint for checkpoint in checkpoints if checkpoint.mandate_satisfied]
    if not eligible:
        raise ValueError("no checkpoint satisfies the validation mandate")
    return max(eligible, key=lambda checkpoint: (checkpoint.net_log_growth, -checkpoint.turnover))

