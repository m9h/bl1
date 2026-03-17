"""Electrical stimulation via MEA electrodes.

Provides functions to compute per-neuron current injection vectors
when a subset of electrodes deliver stimulation, and to generate
feedback stimulation patterns for closed-loop game experiments
(e.g. "predictable" vs "unpredictable" feedback after DishBrain).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Current injection
# ---------------------------------------------------------------------------

@jax.jit
def apply_stimulation(
    neuron_positions: Array,
    electrode_positions: Array,
    stim_electrodes: Array,
    stim_amplitude: float,
    activation_radius_um: float = 75.0,
) -> Array:
    """Compute per-neuron stimulation current from active electrodes.

    Each stimulating electrode injects current into nearby neurons with
    a linear distance falloff: neurons at the electrode centre receive
    the full ``stim_amplitude``; neurons at ``activation_radius_um``
    receive zero.  Currents from multiple electrodes are summed.

    Args:
        neuron_positions: Neuron (x, y) positions in um, shape (N, 2).
        electrode_positions: Electrode (x, y) positions in um,
            shape (E, 2).
        stim_electrodes: Boolean mask indicating which electrodes are
            stimulating, shape (E,).
        stim_amplitude: Peak current amplitude (arbitrary units matching
            the Izhikevich ``I_ext`` convention).
        activation_radius_um: Radius within which a neuron receives
            current from an electrode, in um.

    Returns:
        Per-neuron stimulation current, shape (N,).
    """
    # Pairwise distances: (E, N)
    diff = electrode_positions[:, None, :] - neuron_positions[None, :, :]  # (E, N, 2)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # (E, N)

    # Linear falloff: I = amplitude * max(0, 1 - d / radius)
    attenuation = jnp.clip(1.0 - dist / activation_radius_um, 0.0, 1.0)  # (E, N)

    # Mask by which electrodes are active
    active_mask = stim_electrodes.astype(jnp.float32)[:, None]  # (E, 1)
    current_per_electrode = stim_amplitude * attenuation * active_mask  # (E, N)

    # Sum contributions from all electrodes -> (N,)
    I_stim = current_per_electrode.sum(axis=0)
    return I_stim


# ---------------------------------------------------------------------------
# Feedback stimulation patterns
# ---------------------------------------------------------------------------

def generate_feedback_stim(
    feedback_type: Literal["predictable", "unpredictable", "none"],
    sensory_channels: Array,
    electrode_positions: Array,
    key: Array,
) -> tuple[Array, Array]:
    """Generate stimulation pattern for closed-loop feedback.

    Produces an electrode activation mask and per-electrode timing
    offsets based on whether the feedback should be "predictable"
    (synchronous, all channels) or "unpredictable" (random subset,
    random timing).

    This mirrors the DishBrain protocol where "predictable" feedback
    follows a hit (culture had agency) and "unpredictable" feedback
    follows a miss (no structured information).

    Args:
        feedback_type: One of ``"predictable"``, ``"unpredictable"``,
            or ``"none"``.
        sensory_channels: Boolean mask of which electrodes serve as
            sensory inputs, shape (E,).
        electrode_positions: Electrode positions, shape (E, 2).
            (Currently unused but reserved for spatially-structured
            stimulation in later phases.)
        key: JAX PRNG key.

    Returns:
        stim_electrodes: Boolean mask of electrodes to stimulate,
            shape (E,).
        timing_offsets_ms: Per-electrode timing offset in ms,
            shape (E,).  For "predictable" feedback all offsets are
            zero; for "unpredictable" they are drawn uniformly from
            [0, 10) ms.
    """
    n_electrodes = sensory_channels.shape[0]

    if feedback_type == "none":
        return (
            jnp.zeros(n_electrodes, dtype=jnp.bool_),
            jnp.zeros(n_electrodes),
        )

    if feedback_type == "predictable":
        # Synchronous stimulation on all sensory channels
        stim_electrodes = sensory_channels
        timing_offsets = jnp.zeros(n_electrodes)
        return stim_electrodes, timing_offsets

    if feedback_type == "unpredictable":
        # Random subset of sensory channels with random timing
        key_subset, key_timing = jax.random.split(key)

        # Each sensory channel included independently with p=0.5
        include = jax.random.bernoulli(key_subset, p=0.5, shape=(n_electrodes,))
        stim_electrodes = sensory_channels & include

        # Random timing offsets in [0, 10) ms
        timing_offsets = jax.random.uniform(
            key_timing, shape=(n_electrodes,), minval=0.0, maxval=10.0
        )
        # Non-stimulating electrodes get zero offset
        timing_offsets = jnp.where(stim_electrodes, timing_offsets, 0.0)
        return stim_electrodes, timing_offsets

    raise ValueError(
        f"Unknown feedback_type: {feedback_type!r}. "
        "Expected 'predictable', 'unpredictable', or 'none'."
    )
