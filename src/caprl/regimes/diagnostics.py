"""Diagnostics and persistence-friendly summaries for fitted HMMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HMMFitAttempt:
    initialization: int
    seed: int
    converged: bool
    valid: bool
    iterations: int
    log_likelihood: float
    minimum_occupancy: float
    reason: str | None = None


@dataclass(frozen=True)
class HMMFitDiagnostics:
    selected_initialization: int
    selected_seed: int
    log_likelihood: float
    iterations: int
    state_occupancies: tuple[float, ...]
    expected_durations: tuple[float, ...]
    attempts: tuple[HMMFitAttempt, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

