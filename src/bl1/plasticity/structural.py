"""Structural plasticity — activity-dependent synapse creation and elimination.

Operates on much slower timescales than STDP (hours, not ms). In simulation,
typically called every few seconds of simulated time.

Rules:
- Synapses with very low weight (below pruning threshold) are candidates for elimination
- High-activity neuron pairs that are not connected are candidates for new synapse formation
- Total synapse count is maintained near a homeostatic target

This implementation works with dense weight matrices only (MVP). BCOO support
would require dynamic sparsity pattern changes which JAX doesn't natively support
efficiently; for large-scale networks, structural plasticity should be done in
numpy and converted back to BCOO.

Weight convention
-----------------
``W[j, i]`` is the synaptic weight *from* pre-synaptic neuron **i** *to*
post-synaptic neuron **j**.  This matches the convention used in the rest of
the BL-1 codebase so that ``I_syn = W @ spikes`` yields the total synaptic
current received by each neuron.

Reference: Van Ooyen et al. (2003), Van Ooyen (2011)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class StructuralPlasticityParams(NamedTuple):
    """Parameters for structural plasticity."""
    prune_threshold: float = 0.001     # Weights below this are candidates for pruning
    prune_prob: float = 0.1            # Probability of pruning an eligible synapse per update
    growth_prob: float = 0.01          # Probability of forming a new synapse between active pair
    activity_threshold: float = 0.1    # Minimum activity level (spikes/s) for growth candidate
    max_distance_um: float = 500.0     # Maximum distance for new synapse formation
    target_synapse_fraction: float = 0.05  # Target fraction of possible synapses that exist
    w_new: float = 0.01               # Weight of newly formed synapses


# ---------------------------------------------------------------------------
# Structural plasticity update
# ---------------------------------------------------------------------------

def structural_update(
    key: Array,
    W_exc: Array,
    positions: Array,
    is_excitatory: Array,
    rate_estimates: Array,
    params: StructuralPlasticityParams = StructuralPlasticityParams(),
) -> Array:
    """Apply one round of structural plasticity to excitatory weights.

    This is a numpy-based operation (not JIT-compiled) since it changes
    the sparsity pattern of the weight matrix.

    Args:
        key: JAX PRNG key
        W_exc: (N, N) dense excitatory weight matrix
        positions: (N, 2) neuron positions in micrometres
        is_excitatory: (N,) boolean mask
        rate_estimates: (N,) firing rate estimates in Hz
        params: Structural plasticity parameters

    Returns:
        Updated W_exc with pruned and newly formed synapses
    """
    # Move everything to numpy for sparsity-pattern manipulation.
    W = np.array(W_exc, dtype=np.float64)
    pos = np.array(positions, dtype=np.float64)
    is_exc = np.array(is_excitatory, dtype=bool)
    rates = np.array(rate_estimates, dtype=np.float64)

    N = W.shape[0]

    # Consume JAX key to seed a numpy RNG.
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 0. Compute homeostatic pressure to adjust prune/growth rates
    # ------------------------------------------------------------------
    n_existing = int(np.count_nonzero(W))
    # Maximum possible excitatory synapses: excitatory pre * all post, minus
    # self-connections on excitatory neurons.
    n_exc_pre = int(np.sum(is_exc))
    n_possible = n_exc_pre * N - int(np.sum(is_exc))  # subtract self-connections
    if n_possible <= 0:
        # Nothing to do — no excitatory neurons or degenerate case.
        return jnp.array(W, dtype=jnp.float32)

    target_count = params.target_synapse_fraction * n_possible
    # Ratio > 1 means we have too many synapses, < 1 means too few.
    if target_count > 0:
        ratio = n_existing / target_count
    else:
        ratio = 1.0

    # Scale prune and growth probabilities based on homeostatic pressure.
    # If we have too many synapses (ratio > 1), increase pruning and
    # decrease growth, and vice versa.
    effective_prune_prob = np.clip(params.prune_prob * ratio, 0.0, 1.0)
    effective_growth_prob = np.clip(params.growth_prob / max(ratio, 1e-8), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 1. Pruning: remove weak excitatory synapses
    # ------------------------------------------------------------------
    # Find synapses below threshold.  Only prune excitatory synapses
    # (pre-synaptic neuron is excitatory).
    weak_mask = (W > 0.0) & (W < params.prune_threshold)
    # Ensure pre-synaptic neuron is excitatory: W[j, i] — i is column.
    exc_pre_mask = is_exc[np.newaxis, :]  # broadcast along rows
    prune_candidates = weak_mask & exc_pre_mask

    if np.any(prune_candidates):
        candidate_indices = np.argwhere(prune_candidates)  # (K, 2)
        prune_draws = rng.random(candidate_indices.shape[0])
        to_prune = prune_draws < effective_prune_prob
        for idx in candidate_indices[to_prune]:
            W[idx[0], idx[1]] = 0.0

    # ------------------------------------------------------------------
    # 2. Growth: form new excitatory synapses between active pairs
    # ------------------------------------------------------------------
    # Find pairs where:
    #   - No existing connection (W[j, i] == 0)
    #   - Pre-synaptic neuron i is excitatory
    #   - Both neurons have activity above threshold
    #   - Distance is within max_distance_um
    #   - Not a self-connection

    # Distance matrix
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, N)

    # Build eligibility mask for growth.
    no_conn = W == 0.0
    pre_is_exc = is_exc[np.newaxis, :]           # (1, N) — column = pre
    pre_active = (rates > params.activity_threshold)[np.newaxis, :]  # (1, N)
    post_active = (rates > params.activity_threshold)[:, np.newaxis]  # (N, 1)
    within_dist = dist <= params.max_distance_um
    not_self = ~np.eye(N, dtype=bool)

    growth_eligible = (
        no_conn & pre_is_exc & pre_active & post_active
        & within_dist & not_self
    )

    if np.any(growth_eligible):
        eligible_indices = np.argwhere(growth_eligible)  # (M, 2)
        growth_draws = rng.random(eligible_indices.shape[0])
        to_grow = growth_draws < effective_growth_prob
        for idx in eligible_indices[to_grow]:
            W[idx[0], idx[1]] = params.w_new

    # ------------------------------------------------------------------
    # 3. Ensure no self-connections (defensive)
    # ------------------------------------------------------------------
    np.fill_diagonal(W, 0.0)

    return jnp.array(W, dtype=jnp.float32)
