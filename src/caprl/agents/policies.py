"""Bounded Beta actor and value critic networks."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta


class BetaActorCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        *,
        hidden_layers: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_layers:
            layers.extend((nn.Linear(input_size, hidden_size), nn.Tanh()))
            input_size = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.concentrations = nn.Linear(input_size, 2)
        self.value_head = nn.Linear(input_size, 1)

    def distribution_and_value(self, observations: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        hidden = self.backbone(observations)
        concentrations = torch.nn.functional.softplus(self.concentrations(hidden)) + 1.0
        distribution = Beta(concentrations[..., 0], concentrations[..., 1])
        return distribution, self.value_head(hidden).squeeze(-1)

    def deterministic_action(self, observations: torch.Tensor) -> torch.Tensor:
        distribution, _ = self.distribution_and_value(observations)
        return distribution.mean

