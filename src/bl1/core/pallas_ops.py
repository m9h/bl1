"""Custom event-driven sparse matmul kernels for spike propagation.

At typical cortical firing rates (1-10 Hz) with dt=0.5ms, fewer than 0.25%
of neurons spike per timestep.  The standard ``W @ spikes`` (whether BCOO
matmul or segment_sum) always touches all nonzeros.  For 100K neurons with
~330 synapses/neuron that means ~33M nonzero reads every 0.5ms step.

This module provides an **event-driven** sparse matmul that processes ONLY
the outgoing synapses of neurons that actually spiked.  At 5 Hz with
100K neurons, ~250 spike per step x ~330 synapses = ~82K active synapses --
a 400x reduction in work.

The key data structure is CSC (Compressed Sparse Column) format, which
groups synapses by their *presynaptic* (column) neuron.  Given a spiking
neuron ``i``, ``col_ptr[i]:col_ptr[i+1]`` indexes all outgoing synapses.

Approach
--------
1. **CSC event-driven kernel** (primary): Uses standard JAX operations
   (``jnp.where``, gather, ``segment_sum``) with CSC-format weights.
   Works on any JAX backend.  Requires JAX >= 0.4.1 for
   ``jnp.where(size=...)`` static-shape support inside ``jit``.

2. **Pallas kernel** (optional enhancement): If ``jax.experimental.pallas``
   is available, provides a fused GPU kernel.  Gracefully falls back to
   approach 1 if Pallas is not installed.

Requires JAX >= 0.4.1 for ``jnp.where(size=...)``.
Pallas enhancement requires JAX >= 0.4.30.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO

# ---------------------------------------------------------------------------
# Pallas availability check
# ---------------------------------------------------------------------------

_PALLAS_AVAILABLE = False
try:
    from jax.experimental import pallas as pl  # noqa: F401

    _PALLAS_AVAILABLE = True
except ImportError:
    pass


def is_pallas_available() -> bool:
    """Return True if JAX Pallas (custom kernel API) is importable."""
    return _PALLAS_AVAILABLE


# ---------------------------------------------------------------------------
# CSC weight container
# ---------------------------------------------------------------------------


class CSCWeights(NamedTuple):
    """Compressed Sparse Column format for event-driven synaptic access.

    For a weight matrix W of shape ``(N_post, N_pre)`` stored in CSC format:

    - ``col_ptr[i]`` to ``col_ptr[i+1]`` gives the range of outgoing
      synapses from presynaptic neuron ``i``.
    - ``row_indices[col_ptr[i]:col_ptr[i+1]]`` gives the postsynaptic
      targets of neuron ``i``.
    - ``data[col_ptr[i]:col_ptr[i+1]]`` gives the corresponding weights.

    Attributes
    ----------
    col_ptr : Array, shape ``(N_pre + 1,)``
        Column pointer array.  ``col_ptr[i]`` is the start index in
        ``row_indices`` / ``data`` for presynaptic neuron ``i``.
    row_indices : Array, shape ``(nnz,)``
        Postsynaptic (row) target indices for each nonzero.
    data : Array, shape ``(nnz,)``
        Weight values for each nonzero.
    n_post : int
        Number of postsynaptic neurons (rows in the original matrix).
    n_pre : int
        Number of presynaptic neurons (columns in the original matrix).
    max_synapses_per_neuron : int
        Maximum number of outgoing synapses for any single presynaptic
        neuron.  Used for padding in the event-driven kernel.
    """

    col_ptr: Array
    row_indices: Array
    data: Array
    n_post: int
    n_pre: int
    max_synapses_per_neuron: int


# ---------------------------------------------------------------------------
# BCOO -> CSC conversion
# ---------------------------------------------------------------------------


def bcoo_to_csc(W: BCOO) -> CSCWeights:
    """Convert a BCOO sparse matrix to CSC format for event-driven access.

    Parameters
    ----------
    W : BCOO
        A 2-D BCOO sparse matrix of shape ``(N_post, N_pre)``.

    Returns
    -------
    CSCWeights
        CSC representation suitable for :func:`csc_event_driven_input`.

    Notes
    -----
    This conversion runs on the host (uses NumPy for pointer construction)
    and should be called once during setup, not inside a JIT-traced function.
    """
    import numpy as np

    n_post, n_pre = W.shape

    # Extract COO data from BCOO
    rows_jax = W.indices[:, 0]
    cols_jax = W.indices[:, 1]
    data_jax = W.data

    # Move to NumPy for pointer construction
    rows_np = np.asarray(rows_jax)
    cols_np = np.asarray(cols_jax)
    data_np = np.asarray(data_jax)

    # Sort by column (presynaptic neuron) for CSC construction
    sort_order = np.argsort(cols_np, kind="stable")
    cols_sorted = cols_np[sort_order]
    rows_sorted = rows_np[sort_order]
    data_sorted = data_np[sort_order]

    # Build column pointer array
    col_ptr = np.zeros(n_pre + 1, dtype=np.int32)
    for c in cols_sorted:
        col_ptr[c + 1] += 1
    np.cumsum(col_ptr, out=col_ptr)

    # Compute max synapses per neuron
    counts = col_ptr[1:] - col_ptr[:-1]
    max_syn = int(np.max(counts)) if len(counts) > 0 else 0

    return CSCWeights(
        col_ptr=jnp.array(col_ptr, dtype=jnp.int32),
        row_indices=jnp.array(rows_sorted, dtype=jnp.int32),
        data=jnp.array(data_sorted, dtype=jnp.float32),
        n_post=n_post,
        n_pre=n_pre,
        max_synapses_per_neuron=max_syn,
    )


# ---------------------------------------------------------------------------
# Event-driven sparse matmul (standard JAX -- no Pallas required)
# ---------------------------------------------------------------------------


def csc_event_driven_input(
    csc: CSCWeights,
    spikes: Array,
    max_active: int = 5000,
) -> Array:
    """Event-driven synaptic input using CSC format.

    Only processes outgoing synapses from neurons that actually spiked,
    achieving O(active_synapses) work instead of O(total_synapses).

    Algorithm:
    1. Find indices of spiking neurons (padded to ``max_active`` for JIT).
    2. For each active neuron, gather its outgoing synapse targets/weights
       from the CSC column pointer ranges.
    3. Scatter-add weighted contributions into a postsynaptic accumulator.

    Parameters
    ----------
    csc : CSCWeights
        Pre-converted CSC weight matrix.
    spikes : Array, shape ``(N_pre,)``
        Presynaptic spike vector (float; 0.0 for no spike, positive for
        spike amplitude).
    max_active : int
        Maximum number of simultaneously active (spiking) neurons to
        process.  Must be a compile-time constant for JIT.  If more
        neurons spike than this limit, excess spikes are silently dropped.
        Default 5000.

    Returns
    -------
    Array, shape ``(n_post,)``
        Synaptic input per postsynaptic neuron.
    """
    max_syn = csc.max_synapses_per_neuron
    n_post = csc.n_post

    # Handle edge case: no synapses at all
    if max_syn == 0:
        return jnp.zeros(n_post, dtype=jnp.float32)

    # 1. Find spiking neuron indices (padded to fixed size for JIT)
    spike_mask = spikes > 0
    active_indices = jnp.where(spike_mask, size=max_active, fill_value=0)[0]

    # Create a mask for valid active entries (vs padding)
    # Count of true spikes; entries beyond n_active are padding
    n_active = jnp.sum(spike_mask.astype(jnp.int32))
    active_valid = jnp.arange(max_active) < n_active  # (max_active,)

    # 2. For each active neuron, gather its CSC column range
    starts = csc.col_ptr[active_indices]  # (max_active,)
    ends = csc.col_ptr[active_indices + 1]  # (max_active,)

    # 3. Build padded index array: (max_active, max_syn)
    offsets = jnp.arange(max_syn)  # (max_syn,)
    flat_indices = starts[:, None] + offsets[None, :]  # (max_active, max_syn)

    # Mask: valid if (a) index < end of this neuron's range AND
    #                 (b) this active slot is not padding
    syn_valid = (flat_indices < ends[:, None]) & active_valid[:, None]

    # Clip indices to valid range (padding reads data[0], masked out below)
    nnz = csc.data.shape[0]
    safe_indices = jnp.clip(flat_indices, 0, nnz - 1)

    # 4. Gather weights and targets
    weights = csc.data[safe_indices]  # (max_active, max_syn)
    targets = csc.row_indices[safe_indices]  # (max_active, max_syn)

    # Apply validity mask to weights (zero out padding contributions)
    weights = jnp.where(syn_valid, weights, 0.0)

    # Scale by spike amplitude
    spike_amps = spikes[active_indices]  # (max_active,)
    weighted = weights * spike_amps[:, None]  # (max_active, max_syn)

    # 5. Scatter-add into postsynaptic neurons
    flat_targets = targets.ravel()  # (max_active * max_syn,)
    flat_weighted = weighted.ravel()  # (max_active * max_syn,)

    return jax.ops.segment_sum(flat_weighted, flat_targets, num_segments=n_post)


# ---------------------------------------------------------------------------
# Event-driven sparse matmul v2: flat gather (no 2D waste)
# ---------------------------------------------------------------------------


def csc_event_driven_input_v2(
    csc: CSCWeights,
    spikes: Array,
    max_active: int = 5000,
    max_synapses_total: int = 200_000,
) -> Array:
    """Optimised event-driven synaptic input using flat gather.

    Avoids the (max_active, max_syn) 2D index matrix of v1 by computing
    a flat array of active synapse indices directly.  This is significantly
    more memory-efficient when max_syn varies widely across neurons.

    Parameters
    ----------
    csc : CSCWeights
        Pre-converted CSC weight matrix.
    spikes : Array, shape ``(N_pre,)``
        Presynaptic spike vector (float).
    max_active : int
        Maximum simultaneously active neurons (compile-time constant).
    max_synapses_total : int
        Maximum total active synapses across all spiking neurons
        (compile-time constant).  Set to max_active * mean_synapses.
        Default 200,000 handles 500 spikes * 400 synapses/neuron.
    """
    n_post = csc.n_post
    nnz = csc.data.shape[0]

    if csc.max_synapses_per_neuron == 0:
        return jnp.zeros(n_post, dtype=jnp.float32)

    # 1. Find spiking neuron indices
    spike_mask = spikes > 0
    active_indices = jnp.where(spike_mask, size=max_active, fill_value=0)[0]
    n_active = jnp.sum(spike_mask.astype(jnp.int32))
    active_valid = jnp.arange(max_active) < n_active

    # 2. Compute synapse counts and offsets for each active neuron
    starts = csc.col_ptr[active_indices]       # (max_active,)
    ends = csc.col_ptr[active_indices + 1]     # (max_active,)
    counts = jnp.where(active_valid, ends - starts, 0)  # (max_active,)

    # Exclusive prefix sum to get flat output offsets
    offsets = jnp.cumsum(counts) - counts  # (max_active,)

    # 3. Build flat synapse index array using repeat-based expansion
    # For each active neuron i, generate indices [starts[i], starts[i]+1, ..., ends[i]-1]
    # Expand to flat array of size max_synapses_total
    neuron_ids = jnp.repeat(jnp.arange(max_active), counts, total_repeat_length=max_synapses_total)
    within_offsets = jnp.arange(max_synapses_total) - jnp.repeat(offsets, counts, total_repeat_length=max_synapses_total)

    flat_syn_indices = jnp.repeat(starts, counts, total_repeat_length=max_synapses_total) + within_offsets

    # Validity: the synapse index must be within [starts[i], ends[i]) and within nnz
    flat_ends = jnp.repeat(ends, counts, total_repeat_length=max_synapses_total)
    total_active_synapses = jnp.sum(counts)
    flat_valid = (jnp.arange(max_synapses_total) < total_active_synapses) & (flat_syn_indices < flat_ends) & (flat_syn_indices < nnz)

    # Safe indices for gather
    safe_indices = jnp.where(flat_valid, flat_syn_indices, 0)

    # 4. Gather weights and targets
    weights = csc.data[safe_indices]
    targets = csc.row_indices[safe_indices]
    spike_amps = spikes[active_indices]
    flat_amps = jnp.repeat(spike_amps, counts, total_repeat_length=max_synapses_total)

    # Mask invalid contributions
    weighted = jnp.where(flat_valid, weights * flat_amps, 0.0)

    # 5. Scatter-add
    return jax.ops.segment_sum(weighted, targets, num_segments=n_post)


# ---------------------------------------------------------------------------
# Pallas GPU kernel for event-driven CSC synaptic input
# ---------------------------------------------------------------------------


def _make_pallas_csc_kernel(max_syn: int, nnz: int):
    """Create a Pallas kernel function with closed-over constants."""

    def kernel(
        col_ptr_ref,
        row_indices_ref,
        data_ref,
        spikes_ref,
        active_indices_ref,
        active_valid_ref,
        output_ref,
    ):
        block_idx = pl.program_id(axis=0)
        is_valid = active_valid_ref[block_idx]
        neuron_idx = active_indices_ref[block_idx]
        spike_amp = spikes_ref[neuron_idx]
        start = col_ptr_ref[neuron_idx]
        end = col_ptr_ref[neuron_idx + 1]

        def _body(syn_offset, _):
            syn_idx = start + syn_offset
            in_range = (syn_idx < end) & (syn_idx < nnz) & (is_valid > 0)
            safe_idx = jnp.where(in_range, syn_idx, 0)
            weight = data_ref[safe_idx]
            target = row_indices_ref[safe_idx]
            contrib = jnp.where(in_range, weight * spike_amp, 0.0)
            pl.atomic_add(output_ref, (target,), contrib)
            return None

        jax.lax.fori_loop(0, max_syn, _body, None)

    return kernel


def pallas_event_driven_input(
    csc: CSCWeights,
    spikes: Array,
    max_active: int = 5000,
) -> Array:
    """Event-driven synaptic input using a Pallas GPU kernel.

    Each Pallas grid block processes one spiking neuron, iterating over
    its CSC column range and using atomic_add to accumulate contributions.
    Falls back to CSC v2 if Pallas is not available or fails.
    """
    if not _PALLAS_AVAILABLE:
        return csc_event_driven_input_v2(csc, spikes, max_active)

    max_syn = csc.max_synapses_per_neuron
    n_post = csc.n_post
    nnz = csc.data.shape[0]

    if max_syn == 0:
        return jnp.zeros(n_post, dtype=jnp.float32)

    spike_mask = spikes > 0
    active_indices = jnp.where(spike_mask, size=max_active, fill_value=0)[0].astype(jnp.int32)
    n_active = jnp.sum(spike_mask.astype(jnp.int32))
    active_valid = (jnp.arange(max_active) < n_active).astype(jnp.int32)

    try:
        kernel_fn = _make_pallas_csc_kernel(max_syn, nnz)
        _no = pl.no_block_spec
        output = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((n_post,), jnp.float32),
            grid=(max_active,),
            in_specs=[_no, _no, _no, _no, _no, _no],
            out_specs=_no,
        )(
            csc.col_ptr, csc.row_indices, csc.data,
            spikes, active_indices, active_valid,
        )
        return output
    except (TypeError, AttributeError, NotImplementedError) as e:
        # Pallas API mismatch — fall back gracefully
        return csc_event_driven_input_v2(csc, spikes, max_active)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def event_driven_input(
    csc: CSCWeights,
    spikes: Array,
    max_active: int = 5000,
) -> Array:
    """Compute synaptic input ``W @ spikes`` using event-driven CSC access.

    Dispatches to the best available implementation:
    1. Pallas GPU kernel (if available and on GPU)
    2. CSC v2 flat gather (default)
    3. CSC v1 2D padded gather (legacy fallback)
    """
    if _PALLAS_AVAILABLE and jax.default_backend() == "gpu":
        return pallas_event_driven_input(csc, spikes, max_active)
    return csc_event_driven_input_v2(csc, spikes, max_active)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def benchmark_event_driven(
    n_neurons: int = 50000,
    nnz_per_neuron: int = 330,
    spike_rate: float = 0.001,
    n_repeats: int = 50,
) -> dict:
    """Benchmark all sparse matmul implementations.

    Returns dict with timing results for BCOO, CSC v1, CSC v2, and Pallas.
    """
    import time as _time

    import numpy as np

    rng = np.random.default_rng(42)
    N = n_neurons
    nnz = N * nnz_per_neuron

    # Random sparse matrix in COO
    rows = rng.integers(0, N, size=nnz).astype(np.int32)
    cols = rng.integers(0, N, size=nnz).astype(np.int32)
    data = rng.uniform(0.01, 0.1, size=nnz).astype(np.float32)

    # Build BCOO
    indices = jnp.stack([jnp.array(rows), jnp.array(cols)], axis=1)
    W_bcoo = BCOO((jnp.array(data), indices), shape=(N, N))

    # Build CSC
    csc = bcoo_to_csc(W_bcoo)

    # Spike vector
    spikes = (jax.random.uniform(jax.random.PRNGKey(0), (N,)) < spike_rate).astype(jnp.float32)
    n_spikes = int(jnp.sum(spikes))
    print(f"  N={N}, NNZ={nnz:,}, spikes={n_spikes}")

    results = {}

    # BCOO matmul
    _ = (W_bcoo @ spikes).block_until_ready()
    t0 = _time.perf_counter()
    for _ in range(n_repeats):
        (W_bcoo @ spikes).block_until_ready()
    results["bcoo_ms"] = (_time.perf_counter() - t0) / n_repeats * 1000

    # CSC v1
    _ = csc_event_driven_input(csc, spikes, max_active=max(n_spikes * 2, 100)).block_until_ready()
    ma = max(n_spikes * 2, 100)
    t0 = _time.perf_counter()
    for _ in range(n_repeats):
        csc_event_driven_input(csc, spikes, max_active=ma).block_until_ready()
    results["csc_v1_ms"] = (_time.perf_counter() - t0) / n_repeats * 1000

    # CSC v2
    mst = max(n_spikes * nnz_per_neuron * 2, 1000)
    _ = csc_event_driven_input_v2(csc, spikes, max_active=ma, max_synapses_total=mst).block_until_ready()
    t0 = _time.perf_counter()
    for _ in range(n_repeats):
        csc_event_driven_input_v2(csc, spikes, max_active=ma, max_synapses_total=mst).block_until_ready()
    results["csc_v2_ms"] = (_time.perf_counter() - t0) / n_repeats * 1000

    # Pallas (if available)
    if _PALLAS_AVAILABLE and jax.default_backend() == "gpu":
        try:
            _ = pallas_event_driven_input(csc, spikes, max_active=ma).block_until_ready()
            t0 = _time.perf_counter()
            for _ in range(n_repeats):
                pallas_event_driven_input(csc, spikes, max_active=ma).block_until_ready()
            results["pallas_ms"] = (_time.perf_counter() - t0) / n_repeats * 1000
        except Exception as e:
            results["pallas_ms"] = f"FAILED: {e}"

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BL-1 Event-Driven Sparse Matmul Benchmark")
    print("=" * 60)

    for n in [10_000, 50_000, 100_000]:
        print(f"\n--- {n:,} neurons ---")
        r = benchmark_event_driven(n_neurons=n)
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k:<15s}: {v:.3f} ms")
            else:
                print(f"  {k:<15s}: {v}")
