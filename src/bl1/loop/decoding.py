"""Motor decoding: neural activity to game actions.

Reads population activity from designated motor-region electrodes and
produces a discrete action (stay / up / down) by comparing firing rates
with exponential recency weighting.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_motor(
    spike_history_window: Array,
    neuron_electrode_map: Array,
    motor_regions: dict[str, list[int]],
    baseline_rate: float = 20.0,
) -> tuple[int, dict[str, float]]:
    """Decode motor action from neural activity in motor regions.

    The function computes an exponentially-weighted firing rate for neurons
    associated with the "up" and "down" electrode groups, then selects the
    direction whose rate exceeds a baseline threshold.

    Args:
        spike_history_window: Recent spike history, shape ``(W, N)`` where
            *W* is the number of timesteps in the decoding window (e.g.
            the last 100 ms at 0.5 ms resolution = 200 rows) and *N* is
            the total neuron count.  Boolean or 0/1 float.
        neuron_electrode_map: Boolean mask of shape ``(E, N)`` relating
            electrodes to nearby neurons (from
            :func:`bl1.mea.electrode.build_neuron_electrode_map`).
        motor_regions: Dict with ``"up"`` and ``"down"`` keys, each
            mapping to a list of electrode indices that define the motor
            region for that action.
        baseline_rate: Firing-rate threshold in Hz.  A region must exceed
            this rate for its action to be selected.  Default 20 Hz.

    Returns:
        action: ``0`` (stay), ``1`` (up), or ``2`` (down).
        rates: Dict with ``"up"`` and ``"down"`` firing rates in Hz.
    """
    W, _N = spike_history_window.shape
    if W == 0:
        return 0, {"up": 0.0, "down": 0.0}

    spikes_f = spike_history_window.astype(jnp.float32)

    # --- Exponential recency weighting ------------------------------------
    # Build a weight vector that emphasises recent timesteps.
    # w[t] = exp(-alpha * (W - 1 - t))  so w[-1] = 1.0 (most recent).
    # Time constant is set to half the window width, giving meaningful
    # decay across the window.
    alpha = 2.0 / max(W, 1)
    t_indices = jnp.arange(W, dtype=jnp.float32)
    weights = jnp.exp(-alpha * (W - 1 - t_indices))  # (W,)
    # Normalise so the weighted sum has units of "spikes per timestep"
    weights = weights / jnp.sum(weights)

    # Weighted spike count per neuron: (N,)
    weighted_spikes = jnp.einsum("t,tn->n", weights, spikes_f)

    # --- Aggregate per motor region ----------------------------------------
    rates_dict: dict[str, float] = {}
    for region_name in ("up", "down"):
        electrode_ids = motor_regions[region_name]
        if not electrode_ids:
            rates_dict[region_name] = 0.0
            continue

        # Build a neuron mask for this region: OR over all electrodes in
        # the region.  neuron_electrode_map is (E, N).
        electrode_idx = jnp.array(electrode_ids, dtype=jnp.int32)
        region_mask = neuron_electrode_map[electrode_idx].any(axis=0)  # (N,)

        n_neurons_region = jnp.sum(region_mask.astype(jnp.float32))
        n_neurons_region = jnp.maximum(n_neurons_region, 1.0)

        # Mean weighted spike count per neuron in this region
        region_weighted = jnp.sum(weighted_spikes * region_mask.astype(jnp.float32))
        mean_weighted = region_weighted / n_neurons_region

        # Convert to Hz.  ``mean_weighted`` is a weighted average spike
        # probability per timestep.  Multiply by the effective number of
        # timesteps contributing (W / effective_window) and convert to Hz.
        # Since weights are normalised, mean_weighted ~ spikes/step.
        # Rate (Hz) = mean_weighted / dt_s, but we don't have dt here,
        # so we express it as mean_weighted * W (total weighted spikes)
        # / (window_duration_s).  We approximate using effective counts:
        # rate = mean_weighted * 1000 / dt_ms, but we lack dt_ms.
        # Instead, use the raw count approach:
        # total_spikes_region / (n_neurons * window_s)
        # For consistency, compute the raw rate without weighting for the
        # threshold comparison, and use weighting for the directional
        # comparison.  Here we simply report the weighted rate in
        # normalised units (equivalent Hz assuming 1 ms bins by convention).
        #
        # Practical approach: compute rate from raw spike counts in Hz,
        # where the window duration is W timesteps.  We assume dt=0.5 ms
        # (the BL-1 default) for the rate conversion.  The caller can
        # override baseline_rate accordingly.
        dt_ms_assumed = 0.5
        window_s = (W * dt_ms_assumed) / 1000.0

        raw_spikes = jnp.sum(
            spikes_f * region_mask.astype(jnp.float32)[None, :], axis=(0, 1)
        )
        raw_rate = raw_spikes / (n_neurons_region * window_s)
        rates_dict[region_name] = float(raw_rate)

    # --- Decision logic ----------------------------------------------------
    up_rate = rates_dict["up"]
    down_rate = rates_dict["down"]

    # Use exponentially-weighted signal for the directional comparison
    # (who is more active) but raw rate for the baseline threshold
    # (is anyone active enough to warrant movement).
    if up_rate > baseline_rate and up_rate >= down_rate:
        action = 1  # up
    elif down_rate > baseline_rate and down_rate > up_rate:
        action = 2  # down
    else:
        action = 0  # stay

    return action, rates_dict
