"""Numerically stable forward filtering for Gaussian hidden Markov models."""

from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Small dependency-free log-sum-exp implementation."""
    maximum = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    result = maximum + np.log(summed)
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


def gaussian_log_emissions(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Return log p(x_t | state=k) with shape (time, state)."""
    observations = np.asarray(observations, dtype=float).reshape(-1, 1)
    return -0.5 * (
        np.log(2.0 * np.pi * variances) + (observations - means) ** 2 / variances
    )


def forward_log_probabilities(
    log_emissions: np.ndarray,
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run the causal forward recursion and return normalized filtered probabilities."""
    time_steps, states = log_emissions.shape
    if time_steps == 0:
        raise ValueError("at least one observation is required")
    log_initial = np.log(initial_probabilities)
    log_transition = np.log(transition_matrix)
    log_alpha = np.empty((time_steps, states), dtype=float)
    log_alpha[0] = log_initial + log_emissions[0]
    for time in range(1, time_steps):
        log_alpha[time] = log_emissions[time] + logsumexp(
            log_alpha[time - 1][:, None] + log_transition, axis=0
        )
    log_likelihood = float(logsumexp(log_alpha[-1]))
    normalizers = logsumexp(log_alpha, axis=1)
    filtered = np.exp(log_alpha - normalizers[:, None])
    return filtered, log_likelihood


def forward_log_messages(
    log_emissions: np.ndarray,
    initial_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return unnormalized forward messages for internal EM calculations."""
    time_steps, states = log_emissions.shape
    log_transition = np.log(transition_matrix)
    messages = np.empty((time_steps, states), dtype=float)
    messages[0] = np.log(initial_probabilities) + log_emissions[0]
    for time in range(1, time_steps):
        messages[time] = log_emissions[time] + logsumexp(
            messages[time - 1][:, None] + log_transition, axis=0
        )
    return messages, float(logsumexp(messages[-1]))

