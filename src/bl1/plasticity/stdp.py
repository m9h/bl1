"""Eligibility-trace spike-timing-dependent plasticity (STDP).

Implements the trace-based formulation of STDP that avoids explicit spike-pair
bookkeeping.  Each neuron maintains a pre-synaptic and post-synaptic
eligibility trace that decays exponentially.  When a spike arrives, the trace
is incremented; the trace value at the time of a partner spike exactly
reproduces the exponentially-weighted STDP window.

Both dense (jnp.ndarray) and sparse (jax.experimental.sparse.BCOO) weight
matrices are supported.  The implementation auto-detects the matrix type and
dispatches to the appropriate update path.

Weight convention
-----------------
``W[j, i]`` is the synaptic weight *from* pre-synaptic neuron **i** *to*
post-synaptic neuron **j**.  This matches the convention used in the rest of
the BL-1 codebase so that ``I_syn = W @ spikes`` yields the total synaptic
current received by each neuron.

Reference
---------
Morrison, A., Diesmann, M., & Gerstner, W. (2008).  Phenomenological models
of synaptic plasticity based on spike timing.  *Biological Cybernetics*,
98(6), 459-478.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO


# ---------------------------------------------------------------------------
# Parameter and state containers
# ---------------------------------------------------------------------------

class STDPParams(NamedTuple):
    """Static parameters governing the STDP learning rule."""
    A_plus: float = 0.005       # LTP amplitude
    A_minus: float = 0.00525    # LTD amplitude (slight dominance for stability)
    tau_plus: float = 20.0      # LTP time constant (ms)
    tau_minus: float = 50.0     # LTD time constant (ms)
    w_max: float = 0.1          # Maximum excitatory weight
    w_min: float = 0.0          # Minimum weight


class STDPState(NamedTuple):
    """Per-neuron eligibility traces updated every timestep."""
    pre_trace: Array   # (N,) pre-synaptic eligibility trace
    post_trace: Array  # (N,) post-synaptic eligibility trace


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def init_stdp_state(n_neurons: int) -> STDPState:
    """Create a zeroed STDP state for *n_neurons* neurons.

    Args:
        n_neurons: Number of neurons in the population.

    Returns:
        An ``STDPState`` with zero-initialised traces.
    """
    return STDPState(
        pre_trace=jnp.zeros(n_neurons, dtype=jnp.float32),
        post_trace=jnp.zeros(n_neurons, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_bcoo(x: Array | BCOO) -> bool:
    """Return True if *x* is a JAX BCOO sparse matrix."""
    return isinstance(x, BCOO)


# ---------------------------------------------------------------------------
# Dense weight update
# ---------------------------------------------------------------------------

@jax.jit
def _stdp_update_dense(
    stdp_state: STDPState,
    stdp_params: STDPParams,
    spikes: Array,
    W_exc: Array,
    is_excitatory: Array,
    dt: float = 0.5,
) -> tuple[STDPState, Array]:
    """STDP update for a dense (N, N) weight matrix."""
    A_plus, A_minus, tau_plus, tau_minus, w_max, w_min = stdp_params

    # ------------------------------------------------------------------
    # 1. Decay traces
    # ------------------------------------------------------------------
    decay_pre = jnp.exp(-dt / tau_plus)
    decay_post = jnp.exp(-dt / tau_minus)

    pre_trace = stdp_state.pre_trace * decay_pre
    post_trace = stdp_state.post_trace * decay_post

    # ------------------------------------------------------------------
    # 2. Increment traces at spike times
    # ------------------------------------------------------------------
    spikes_f = spikes.astype(jnp.float32)
    pre_trace = pre_trace + spikes_f * A_plus
    post_trace = post_trace + spikes_f * A_minus

    # ------------------------------------------------------------------
    # 3. Compute weight changes
    # ------------------------------------------------------------------
    # spikes_pre  (i fired this step) → cols of W  → W[j, i]
    # spikes_post (j fired this step) → rows of W  → W[j, i]
    #
    # LTP: post j spiked, use pre_trace[i]
    #   dW_ltp[j, i] = spikes[j] * pre_trace[i]
    # LTD: pre i spiked, use post_trace[j]
    #   dW_ltd[j, i] = spikes[i] * post_trace[j]

    dW = (
        spikes_f[:, None] * pre_trace[None, :]      # LTP  (N_post, N_pre)
        - post_trace[:, None] * spikes_f[None, :]    # LTD  (N_post, N_pre)
    )

    # Only update existing excitatory connections.  Inhibitory weights and
    # absent connections (zero entries) are left untouched.
    exc_mask = is_excitatory[None, :]   # pre-synaptic neuron must be excitatory
    conn_mask = W_exc > 0.0             # only where a connection exists

    W_new = W_exc + dW * conn_mask * exc_mask
    W_new = jnp.clip(W_new, w_min, w_max)

    # Preserve exact zeros for absent connections (clip may have turned -0→0
    # but we explicitly zero them to keep the sparsity pattern).
    W_new = jnp.where(conn_mask, W_new, 0.0)

    new_state = STDPState(pre_trace=pre_trace, post_trace=post_trace)
    return new_state, W_new


# ---------------------------------------------------------------------------
# Sparse (BCOO) weight update
# ---------------------------------------------------------------------------

@jax.jit
def _stdp_update_sparse(
    stdp_state: STDPState,
    stdp_params: STDPParams,
    spikes: Array,
    W_exc_data: Array,
    W_exc_rows: Array,
    W_exc_cols: Array,
    is_excitatory: Array,
    dt: float = 0.5,
) -> tuple[STDPState, Array]:
    """STDP update operating directly on BCOO sparse storage arrays.

    Rather than accepting a BCOO object (which complicates JIT tracing),
    this function takes the raw data and index arrays.  The caller is
    responsible for reconstructing the BCOO matrix if needed.

    Args:
        stdp_state: Current eligibility traces.
        stdp_params: STDP hyperparameters.
        spikes: (N,) boolean spike vector.
        W_exc_data: (nnz,) weight values from BCOO.data.
        W_exc_rows: (nnz,) row (post-synaptic) indices from BCOO.indices[:, 0].
        W_exc_cols: (nnz,) col (pre-synaptic) indices from BCOO.indices[:, 1].
        is_excitatory: (N,) boolean mask.
        dt: Timestep in ms.

    Returns:
        (new_stdp_state, new_W_exc_data) — updated traces and weight values.
    """
    A_plus, A_minus, tau_plus, tau_minus, w_max, w_min = stdp_params

    # ------------------------------------------------------------------
    # 1. Decay traces
    # ------------------------------------------------------------------
    decay_pre = jnp.exp(-dt / tau_plus)
    decay_post = jnp.exp(-dt / tau_minus)

    pre_trace = stdp_state.pre_trace * decay_pre
    post_trace = stdp_state.post_trace * decay_post

    # ------------------------------------------------------------------
    # 2. Increment traces at spike times
    # ------------------------------------------------------------------
    spikes_f = spikes.astype(jnp.float32)
    pre_trace = pre_trace + spikes_f * A_plus
    post_trace = post_trace + spikes_f * A_minus

    # ------------------------------------------------------------------
    # 3. Compute weight changes at non-zero positions only
    # ------------------------------------------------------------------
    # rows = post-synaptic indices (j), cols = pre-synaptic indices (i)
    # LTP: post j spiked → use pre_trace of i
    ltp = spikes_f[W_exc_rows] * pre_trace[W_exc_cols]     # (nnz,)
    # LTD: pre i spiked → use post_trace of j
    ltd = spikes_f[W_exc_cols] * post_trace[W_exc_rows]    # (nnz,)

    # Only update excitatory pre-synaptic connections
    exc_mask = is_excitatory[W_exc_cols].astype(jnp.float32)  # (nnz,)

    new_data = W_exc_data + (ltp - ltd) * exc_mask
    new_data = jnp.clip(new_data, w_min, w_max)

    new_state = STDPState(pre_trace=pre_trace, post_trace=post_trace)
    return new_state, new_data


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def stdp_update(
    stdp_state: STDPState,
    stdp_params: STDPParams,
    spikes: Array,
    W_exc: Array | BCOO,
    is_excitatory: Array,
    dt: float = 0.5,
) -> tuple[STDPState, Array | BCOO]:
    """Update STDP traces and excitatory weights based on current spikes.

    Supports both dense ``jnp.ndarray`` and sparse ``BCOO`` weight matrices.
    For BCOO inputs the update is performed directly on the non-zero data
    array, avoiding any costly dense materialisation.

    Args:
        stdp_state: Current pre / post eligibility traces.
        stdp_params: STDP hyperparameters.
        spikes: (N,) boolean spike vector for this timestep.
        W_exc: (N, N) excitatory weight matrix — dense or BCOO.
        is_excitatory: (N,) boolean mask identifying excitatory neurons.
        dt: Timestep in ms (default 0.5, matching the Izhikevich integrator).

    Returns:
        ``(new_stdp_state, new_W_exc)`` with updated traces and weights.
        The returned weight matrix has the same type (dense or BCOO) as the
        input.
    """
    if _is_bcoo(W_exc):
        rows = W_exc.indices[:, 0]
        cols = W_exc.indices[:, 1]
        new_state, new_data = _stdp_update_sparse(
            stdp_state,
            stdp_params,
            spikes,
            W_exc.data,
            rows,
            cols,
            is_excitatory,
            dt,
        )
        new_W_exc = BCOO(
            (new_data, W_exc.indices),
            shape=W_exc.shape,
            indices_sorted=W_exc.indices_sorted,
            unique_indices=W_exc.unique_indices,
        )
        return new_state, new_W_exc

    return _stdp_update_dense(
        stdp_state, stdp_params, spikes, W_exc, is_excitatory, dt,
    )
