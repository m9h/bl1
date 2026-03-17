"""Axonal conduction delay buffer for spike propagation.

Implements a ring buffer that stores recent spikes and delivers them
to post-synaptic targets after a distance-dependent delay.

For a network with max delay D timesteps, the buffer stores D frames
of spike vectors. At each timestep, new spikes enter the buffer and
delayed spikes are read out for synaptic transmission.

Two implementations:
1. Dense buffer: (D, N) ring buffer -- simple, works for small networks
2. Sparse buffer: stores only non-zero entries -- memory efficient for large N

The dense buffer is JIT-compatible via jax.lax.scan.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class DelayBufferState(NamedTuple):
    """State of the spike delay buffer."""

    buffer: Array  # (max_delay, N) ring buffer of spike vectors
    head: Array  # scalar int, current write position


def init_delay_buffer(n_neurons: int, max_delay_steps: int) -> DelayBufferState:
    """Initialize an empty delay buffer.

    Args:
        n_neurons: Number of neurons.
        max_delay_steps: Maximum delay in timesteps (determines buffer size).

    Returns:
        Zeroed DelayBufferState.
    """
    buffer = jnp.zeros((max_delay_steps, n_neurons), dtype=jnp.float32)
    head = jnp.array(0, dtype=jnp.int32)
    return DelayBufferState(buffer=buffer, head=head)


@jax.jit
def delay_buffer_step(
    state: DelayBufferState,
    spikes: Array,
) -> DelayBufferState:
    """Write new spikes into the delay buffer.

    Args:
        state: Current buffer state.
        spikes: (N,) boolean or float spike vector for this timestep.

    Returns:
        Updated buffer state with new spikes written at head position.
    """
    max_delay = state.buffer.shape[0]
    new_buffer = state.buffer.at[state.head % max_delay].set(
        spikes.astype(jnp.float32)
    )
    new_head = state.head + 1
    return DelayBufferState(buffer=new_buffer, head=new_head)


@jax.jit
def read_delayed_spikes(
    state: DelayBufferState,
    delay_matrix: Array,
    weights: Array,
) -> Array:
    """Read delayed spikes and compute weighted synaptic input.

    For each synapse (i->j) with delay d, reads the spike from neuron i
    that occurred d timesteps ago and multiplies by the weight.

    This replaces the simple ``weights @ spikes`` with
    ``sum_i(weights[j,i] * spikes_delayed[i, delay[j,i]])``.

    For the dense implementation, for each possible delay value d in
    [1, max_delay]:
    - Find which spikes occurred d steps ago
    - Multiply by weights that have delay == d
    - Accumulate into postsynaptic current

    Args:
        state: Current delay buffer.
        delay_matrix: (N, N) delay values in timesteps (dense int or float
            matrix).  Entry ``[j, i]`` is the delay from pre-neuron *i* to
            post-neuron *j*, matching the weight matrix convention.
        weights: (N, N) synaptic weight matrix (dense).

    Returns:
        (N,) postsynaptic input current (same semantics as
        ``weights @ spikes`` but with per-synapse delays applied).
    """
    max_delay = state.buffer.shape[0]
    N = weights.shape[0]

    I_post = jnp.zeros(N)

    # For each delay value, gather the appropriate past spikes.
    # The loop extent is max_delay which is typically small (~40 for
    # 20 ms delays at dt=0.5 ms), so this is efficient inside XLA.
    def accumulate_delay(carry, d):
        I_acc = carry
        # Spikes that occurred d timesteps ago
        buf_idx = (state.head - d) % max_delay
        past_spikes = state.buffer[buf_idx]  # (N,)

        # Mask for synapses with this specific delay
        delay_mask = (delay_matrix == d).astype(jnp.float32)  # (N, N)

        # Weighted contribution: for each post neuron j,
        # sum weights[j,i] * past_spikes[i] where delay[j,i] == d
        contribution = (weights * delay_mask) @ past_spikes  # (N,)
        I_acc = I_acc + contribution

        return I_acc, None

    delay_values = jnp.arange(1, max_delay + 1)
    I_post, _ = jax.lax.scan(accumulate_delay, I_post, delay_values)

    return I_post


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_max_delay(delays) -> int:
    """Get maximum delay from a delay matrix (dense or BCOO).

    Args:
        delays: Delay matrix -- either a dense ``jnp.ndarray`` or a
            ``jax.experimental.sparse.BCOO`` array.

    Returns:
        Maximum delay as a Python int.
    """
    if hasattr(delays, "data"):  # BCOO
        return int(jnp.max(delays.data))
    return int(jnp.max(delays))


def delays_to_dense(delays, shape=None) -> Array:
    """Convert BCOO delay matrix to dense.  Helper for delay buffer.

    Args:
        delays: Delay matrix -- BCOO sparse or already dense.
        shape: Ignored when *delays* is already dense or carries its own
            shape (BCOO).

    Returns:
        Dense ``jnp.ndarray`` delay matrix.
    """
    if hasattr(delays, "todense"):
        return delays.todense()
    return jnp.asarray(delays)
