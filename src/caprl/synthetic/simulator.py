"""Synthetic Gaussian Markov-switching return data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_markov_returns(
    n_steps: int,
    *,
    transition_matrix: np.ndarray | None = None,
    means: np.ndarray | None = None,
    standard_deviations: np.ndarray | None = None,
    initial_probabilities: np.ndarray | None = None,
    seed: int = 42,
    start_date: str = "2000-01-03",
) -> pd.DataFrame:
    """Simulate persistent bull/bear states and conditionally Gaussian returns."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    transition = np.asarray(
        transition_matrix if transition_matrix is not None else [[0.97, 0.03], [0.05, 0.95]],
        dtype=float,
    )
    means_array = np.asarray(means if means is not None else [0.0006, -0.0010], dtype=float)
    std_array = np.asarray(
        standard_deviations if standard_deviations is not None else [0.008, 0.018], dtype=float
    )
    states = len(means_array)
    initial = np.asarray(
        initial_probabilities if initial_probabilities is not None else np.full(states, 1 / states),
        dtype=float,
    )
    if transition.shape != (states, states):
        raise ValueError("transition_matrix shape must match the number of states")
    if std_array.shape != (states,) or (std_array <= 0).any():
        raise ValueError("standard_deviations must be positive and match means")
    if initial.shape != (states,):
        raise ValueError("initial_probabilities shape must match means")
    if (transition < 0).any() or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition_matrix rows must be probability vectors")
    if (initial < 0).any() or not np.isclose(initial.sum(), 1.0):
        raise ValueError("initial_probabilities must be a probability vector")

    rng = np.random.default_rng(seed)
    latent_states = np.empty(n_steps, dtype=int)
    latent_states[0] = rng.choice(states, p=initial)
    for time in range(1, n_steps):
        latent_states[time] = rng.choice(states, p=transition[latent_states[time - 1]])
    returns = rng.normal(means_array[latent_states], std_array[latent_states])
    return pd.DataFrame(
        {"return": returns, "true_state": latent_states},
        index=pd.bdate_range(start_date, periods=n_steps, name="date"),
    )

