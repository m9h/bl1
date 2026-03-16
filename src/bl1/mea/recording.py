"""Spike detection and firing-rate estimation from MEA recordings.

Functions in this module convert neuron-level spike data into
electrode-level observables using the precomputed neuron-electrode
mapping produced by :func:`bl1.mea.electrode.build_neuron_electrode_map`.

All operations are vectorised over electrodes and neurons so they can
be JIT-compiled with JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Per-timestep spike detection
# ---------------------------------------------------------------------------

@jax.jit
def detect_spikes(
    spikes: Array,
    neuron_electrode_map: Array,
) -> Array:
    """Aggregate neuron spikes to per-electrode spike counts.

    For each electrode, sums the spike indicators of all neurons that
    fall within that electrode's detection radius.

    Args:
        spikes: Per-neuron spike indicator for the current timestep,
            shape (N,).  Boolean or 0/1 integer.
        neuron_electrode_map: Boolean mask of shape (E, N) from
            :func:`build_neuron_electrode_map`.

    Returns:
        Per-electrode spike count, shape (E,), dtype int32.
    """
    # (E, N) @ (N,) -> (E,)
    return (neuron_electrode_map.astype(jnp.float32) @ spikes.astype(jnp.float32)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Firing-rate estimation
# ---------------------------------------------------------------------------

def compute_electrode_rates(
    spike_history: Array,
    neuron_electrode_map: Array,
    window_ms: float,
    dt: float,
) -> Array:
    """Compute mean firing rate per electrode over a sliding window.

    Averages the spike counts over the most recent ``window_ms``
    milliseconds of the spike history and converts to Hz.

    Args:
        spike_history: Spike indicators over time, shape (T, N).
            The last row corresponds to the most recent timestep.
        neuron_electrode_map: Boolean mask, shape (E, N).
        window_ms: Width of the averaging window in ms.
        dt: Simulation timestep in ms.

    Returns:
        Per-electrode firing rate in Hz, shape (E,).
    """
    T = spike_history.shape[0]
    window_steps = min(max(int(round(window_ms / dt)), 1), T)

    # Slice the most recent window_steps rows
    windowed = spike_history[T - window_steps:]

    # Sum spikes over time: (T_win, N) -> (N,)
    total_spikes_per_neuron = windowed.astype(jnp.float32).sum(axis=0)

    # Project onto electrodes: (E, N) @ (N,) -> (E,)
    electrode_spike_counts = neuron_electrode_map.astype(jnp.float32) @ total_spikes_per_neuron

    # Number of neurons per electrode (avoid division by zero)
    neurons_per_electrode = neuron_electrode_map.astype(jnp.float32).sum(axis=1)
    neurons_per_electrode = jnp.maximum(neurons_per_electrode, 1.0)

    # Mean spikes per neuron per electrode
    mean_spikes = electrode_spike_counts / neurons_per_electrode

    # Convert to Hz: spikes / (window duration in seconds)
    window_sec = (window_steps * dt) / 1000.0
    rates = mean_spikes / window_sec

    return rates
