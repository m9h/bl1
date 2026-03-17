"""Homeostatic synaptic scaling (Turrigiano 2008).

A slow negative-feedback mechanism that multiplicatively adjusts excitatory
synaptic weights to maintain a target firing rate.  The update rule is:

    dW_ij/dt = eta * (r_target - r_i) * W_ij

where *r_target* is the desired firing rate (~5 Hz for cortical cultures),
*r_i* is the exponential-moving-average estimate of the post-synaptic
neuron's firing rate, and *eta* is a small learning rate.

Only excitatory-to-excitatory (E->E) connections are scaled; inhibitory
synapses and synapses onto interneurons are left untouched.

Both dense (jnp.ndarray) and sparse (jax.experimental.sparse.BCOO) weight
matrices are supported.  The implementation auto-detects the matrix type and
dispatches to the appropriate update path.

Weight convention
-----------------
``W[j, i]`` is the synaptic weight *from* pre-synaptic neuron **i** *to*
post-synaptic neuron **j**.  This matches the rest of the BL-1 codebase so
that ``I_syn = W @ spikes`` yields the total synaptic current received by
each neuron.

Reference
---------
Turrigiano, G. G. (2008).  The self-tuning neuron: synaptic scaling of
excitatory synapses.  *Cell*, 135(3), 422-435.
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

class HomeostaticParams(NamedTuple):
    """Parameters for homeostatic synaptic scaling."""
    r_target: float = 5.0    # Target firing rate (Hz)
    eta: float = 1e-5         # Learning rate per ms
    w_min: float = 0.001      # Minimum weight (don't scale to zero)
    w_max: float = 0.2        # Maximum weight


class HomeostaticState(NamedTuple):
    """State for homeostatic scaling -- tracks firing rate estimates."""
    rate_estimate: Array  # (N,) exponential moving average of firing rate (Hz)


# ---------------------------------------------------------------------------
# Initialisation helper
# ---------------------------------------------------------------------------

def init_homeostatic_state(
    n_neurons: int,
    initial_rate: float = 5.0,
) -> HomeostaticState:
    """Create a homeostatic state for *n_neurons* neurons.

    The rate estimate is initialised to *initial_rate* so that the first
    scaling step produces no change when the target rate equals the initial
    rate.

    Args:
        n_neurons: Number of neurons in the population.
        initial_rate: Starting rate estimate (Hz).

    Returns:
        A ``HomeostaticState`` with the rate estimate filled to *initial_rate*.
    """
    return HomeostaticState(
        rate_estimate=jnp.full(n_neurons, initial_rate, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_bcoo(x: Array | BCOO) -> bool:
    """Return True if *x* is a JAX BCOO sparse matrix."""
    return isinstance(x, BCOO)


# ---------------------------------------------------------------------------
# Rate estimation
# ---------------------------------------------------------------------------

@jax.jit
def update_rate_estimate(
    state: HomeostaticState,
    spikes: Array,
    dt_ms: float = 0.5,
    tau_rate_ms: float = 10000.0,
) -> HomeostaticState:
    """Update the exponential moving average firing rate estimate.

    Called every simulation timestep.  The update is:

        rate_new = rate_old * decay + spike * (1000/dt) * (1 - decay)

    where ``decay = exp(-dt / tau_rate)`` and the ``1000/dt`` factor converts
    spikes-per-step into Hz.

    Args:
        state: Current homeostatic state.
        spikes: (N,) boolean spike vector for this timestep.
        dt_ms: Timestep size in milliseconds (default 0.5).
        tau_rate_ms: Time constant for the rate EMA in ms (default 10000,
            i.e. 10 s).

    Returns:
        Updated ``HomeostaticState`` with new rate estimates.
    """
    decay = jnp.exp(-dt_ms / tau_rate_ms)
    instantaneous_rate = spikes.astype(jnp.float32) * (1000.0 / dt_ms)
    new_rate = state.rate_estimate * decay + instantaneous_rate * (1.0 - decay)
    return HomeostaticState(rate_estimate=new_rate)


# ---------------------------------------------------------------------------
# Dense weight update
# ---------------------------------------------------------------------------

@jax.jit
def _homeostatic_scaling_dense(
    state: HomeostaticState,
    params: HomeostaticParams,
    W_exc: Array,
    is_excitatory: Array,
    dt_ms: float = 0.5,
) -> tuple[HomeostaticState, Array]:
    """Homeostatic scaling for a dense (N, N) weight matrix.

    For each E->E synapse W[j, i] the multiplicative update is:

        W_new[j, i] = W[j, i] * (1 + eta * dt * (r_target - rate[j]))

    where j is the post-synaptic neuron and i is the pre-synaptic neuron.
    """
    r_target, eta, w_min, w_max = params

    rate_error = r_target - state.rate_estimate  # (N,)

    # Scale factor per post-synaptic neuron
    scale = 1.0 + eta * dt_ms * rate_error[:, None]  # (N, 1)

    # Masks for E->E connections
    post_exc = is_excitatory[:, None]  # (N, 1) post is excitatory
    pre_exc = is_excitatory[None, :]   # (1, N) pre is excitatory
    ee_mask = post_exc & pre_exc

    # Only scale existing E->E connections (W > 0)
    conn_mask = W_exc > 0.0
    W_new = jnp.where(ee_mask & conn_mask, W_exc * scale, W_exc)
    W_new = jnp.clip(W_new, w_min, w_max)

    # Preserve zeros (absent connections)
    W_new = jnp.where(conn_mask, W_new, 0.0)

    return state, W_new


# ---------------------------------------------------------------------------
# Sparse (BCOO) weight update
# ---------------------------------------------------------------------------

@jax.jit
def _homeostatic_scaling_sparse(
    state: HomeostaticState,
    params: HomeostaticParams,
    W_exc_data: Array,
    W_exc_rows: Array,
    W_exc_cols: Array,
    is_excitatory: Array,
    dt_ms: float = 0.5,
) -> tuple[HomeostaticState, Array]:
    """Homeostatic scaling operating directly on BCOO sparse storage arrays.

    Rather than accepting a BCOO object (which complicates JIT tracing), this
    function takes the raw data and index arrays.  The caller is responsible
    for reconstructing the BCOO matrix if needed.

    Args:
        state: Current homeostatic state (rate estimates).
        params: Homeostatic hyperparameters.
        W_exc_data: (nnz,) weight values from BCOO.data.
        W_exc_rows: (nnz,) row (post-synaptic) indices from BCOO.indices[:, 0].
        W_exc_cols: (nnz,) col (pre-synaptic) indices from BCOO.indices[:, 1].
        is_excitatory: (N,) boolean mask.
        dt_ms: Timestep in ms.

    Returns:
        ``(state, new_W_exc_data)`` -- the (unchanged) state and updated
        weight values.
    """
    r_target, eta, w_min, w_max = params

    rate_error = r_target - state.rate_estimate  # (N,)

    # Per-synapse scale factor based on the post-synaptic neuron's rate error
    scale = 1.0 + eta * dt_ms * rate_error[W_exc_rows]  # (nnz,)

    # E->E mask: both pre and post must be excitatory
    ee_mask = (
        is_excitatory[W_exc_rows] & is_excitatory[W_exc_cols]
    ).astype(jnp.float32)  # (nnz,)

    # Apply scaling only to E->E synapses; leave others unchanged
    new_data = jnp.where(ee_mask > 0.5, W_exc_data * scale, W_exc_data)
    new_data = jnp.clip(new_data, w_min, w_max)

    return state, new_data


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def homeostatic_scaling(
    state: HomeostaticState,
    params: HomeostaticParams,
    W_exc: Array | BCOO,
    is_excitatory: Array,
    dt_ms: float = 0.5,
) -> tuple[HomeostaticState, Array | BCOO]:
    """Apply homeostatic scaling to excitatory weights.

    This should be called periodically (e.g., every 1000 ms) rather than
    every timestep, since homeostatic plasticity operates on slow timescales.

    Supports both dense ``jnp.ndarray`` and sparse ``BCOO`` weight matrices.
    For BCOO inputs the update is performed directly on the non-zero data
    array, avoiding any costly dense materialisation.

    Args:
        state: Current homeostatic state (rate estimates).
        params: Homeostatic hyperparameters.
        W_exc: (N, N) excitatory weight matrix -- dense or BCOO.
        is_excitatory: (N,) boolean mask identifying excitatory neurons.
        dt_ms: Timestep in ms (default 0.5, matching the Izhikevich integrator).

    Returns:
        ``(state, new_W_exc)`` with the (unchanged) state and updated
        weights.  The returned weight matrix has the same type (dense or
        BCOO) as the input.
    """
    if _is_bcoo(W_exc):
        rows = W_exc.indices[:, 0]
        cols = W_exc.indices[:, 1]
        new_state, new_data = _homeostatic_scaling_sparse(
            state,
            params,
            W_exc.data,
            rows,
            cols,
            is_excitatory,
            dt_ms,
        )
        new_W_exc = BCOO(
            (new_data, W_exc.indices),
            shape=W_exc.shape,
            indices_sorted=W_exc.indices_sorted,
            unique_indices=W_exc.unique_indices,
        )
        return new_state, new_W_exc

    return _homeostatic_scaling_dense(
        state, params, W_exc, is_excitatory, dt_ms,
    )
