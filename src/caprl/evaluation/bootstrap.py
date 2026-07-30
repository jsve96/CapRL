"""Paired moving-block bootstrap preserving short-range time dependence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    samples: np.ndarray


def paired_block_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    block_length: int = 40,
    n_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape or len(first) == 0:
        raise ValueError("paired samples must be non-empty matching one-dimensional arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired samples must be finite")
    if not 1 <= block_length <= len(first) or n_samples <= 0:
        raise ValueError("invalid bootstrap size or block length")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    statistic = statistic or (lambda left, right: float(np.mean(left - right)))
    rng = np.random.default_rng(seed)
    number_of_blocks = int(np.ceil(len(first) / block_length))
    samples = np.empty(n_samples)
    for sample in range(n_samples):
        starts = rng.integers(0, len(first), size=number_of_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_length)) % len(first) for start in starts]
        )[: len(first)]
        samples[sample] = statistic(first[indices], second[indices])
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=statistic(first, second),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
        samples=samples,
    )

