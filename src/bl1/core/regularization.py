"""Firing rate regularization for spiking networks.

Provides differentiable penalty functions that discourage pathological
firing rates (seizure-like activity or silence).

Inspired by Spyx's sparsity_reg and silence_reg functions.

Usage:
    from bl1.core.regularization import firing_rate_penalty, sparsity_penalty

    # After collecting spike_history (T, N) from simulation:
    loss = firing_rate_penalty(spike_history, target_rate_hz=5.0)
    total_loss = task_loss + 0.01 * loss
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def firing_rate_penalty(
    spike_history: Array,
    target_rate_hz: float = 5.0,
    dt_ms: float = 0.5,
    penalty_type: str = "l2",
) -> Array:
    """Penalty for deviation from target firing rate.

    Computes the per-neuron mean firing rate and penalizes any
    deviation from the target rate.

    Args:
        spike_history: (T, N) float spike array (1.0 = spike, 0.0 = no spike)
        target_rate_hz: desired mean firing rate in Hz
        dt_ms: timestep in ms
        penalty_type: "l2" (quadratic) or "huber" (robust to outliers)

    Returns:
        Scalar penalty value (differentiable). Zero when all neurons
        fire at exactly the target rate.
    """
    # Per-neuron mean rate in Hz
    rates = spike_history.mean(axis=0) * (1000.0 / dt_ms)
    deviation = rates - target_rate_hz

    if penalty_type == "l2":
        return jnp.mean(deviation ** 2)
    else:  # huber
        delta = 2.0
        abs_dev = jnp.abs(deviation)
        return jnp.mean(
            jnp.where(
                abs_dev <= delta,
                0.5 * deviation ** 2,
                delta * (abs_dev - 0.5 * delta),
            )
        )


def sparsity_penalty(
    spike_history: Array,
    max_rate_hz: float = 20.0,
    dt_ms: float = 0.5,
) -> Array:
    """Penalty for neurons firing above a maximum rate.

    Only penalizes neurons exceeding max_rate_hz; silent neurons
    are not penalized (use silence_penalty for that).

    Args:
        spike_history: (T, N) float spike array
        max_rate_hz: maximum acceptable firing rate in Hz
        dt_ms: timestep in ms

    Returns:
        Scalar penalty (differentiable). Zero when all neurons fire
        at or below max_rate_hz.
    """
    rates = spike_history.mean(axis=0) * (1000.0 / dt_ms)
    excess = jnp.maximum(0.0, rates - max_rate_hz)
    return jnp.mean(excess ** 2)


def silence_penalty(
    spike_history: Array,
    min_rate_hz: float = 0.5,
    dt_ms: float = 0.5,
) -> Array:
    """Penalty for neurons firing below a minimum rate.

    Only penalizes neurons below min_rate_hz; normally active neurons
    are not penalized.

    Args:
        spike_history: (T, N) float spike array
        min_rate_hz: minimum acceptable firing rate in Hz
        dt_ms: timestep in ms

    Returns:
        Scalar penalty (differentiable). Zero when all neurons fire
        at or above min_rate_hz.
    """
    rates = spike_history.mean(axis=0) * (1000.0 / dt_ms)
    deficit = jnp.maximum(0.0, min_rate_hz - rates)
    return jnp.mean(deficit ** 2)
