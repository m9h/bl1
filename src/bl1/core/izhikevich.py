"""Vectorized Izhikevich neuron model for cortical culture simulation.

Implements the Izhikevich (2003) spiking neuron model with five cortical
cell types: Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH),
Fast Spiking (FS), and Low-Threshold Spiking (LTS).

All operations are fully vectorized over the neuron population using JAX.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Izhikevich cell-type parameter table
# Columns: a, b, c, d, fraction
# Excitatory types (80 % of population)
#   RS  – 64 %
#   IB  –  8 %
#   CH  –  8 %
# Inhibitory types (20 % of population)
#   FS  – 16 %
#   LTS –  4 %
# ---------------------------------------------------------------------------

_CELL_TYPES = {
    "RS":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0, "frac": 0.64},
    "IB":  {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0, "frac": 0.08},
    "CH":  {"a": 0.02, "b": 0.2,  "c": -50.0, "d": 2.0, "frac": 0.08},
    "FS":  {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0, "frac": 0.16},
    "LTS": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0, "frac": 0.04},
}

# Ordered list so we can build contiguous blocks
_TYPE_ORDER = ["RS", "IB", "CH", "FS", "LTS"]

# Threshold potential (mV)
V_PEAK = 30.0

# Resting potential used for initialisation (mV)
V_REST = -65.0


# ---------------------------------------------------------------------------
# State containers (plain NamedTuples — JAX pytree compatible out of the box)
# ---------------------------------------------------------------------------

class IzhikevichParams(NamedTuple):
    """Per-neuron Izhikevich model parameters."""
    a: Array  # (N,) recovery time scale
    b: Array  # (N,) recovery sensitivity
    c: Array  # (N,) post-spike reset voltage (mV)
    d: Array  # (N,) post-spike recovery increment


class NeuronState(NamedTuple):
    """Per-neuron dynamic state variables."""
    v: Array      # (N,) membrane potential (mV)
    u: Array      # (N,) recovery variable
    spikes: Array # (N,) boolean spike indicator for the current timestep


# ---------------------------------------------------------------------------
# Population factory
# ---------------------------------------------------------------------------

def create_population(
    key: Array,
    n_neurons: int,
    ei_ratio: float = 0.8,
) -> tuple[IzhikevichParams, NeuronState, Array]:
    """Create a mixed cortical population of Izhikevich neurons.

    Args:
        key: JAX PRNG key (used for future stochastic extensions;
             currently the population is deterministic).
        n_neurons: Total number of neurons.
        ei_ratio: Fraction of neurons that are excitatory (default 0.8).

    Returns:
        params: IzhikevichParams with per-neuron (a, b, c, d).
        state: NeuronState initialised at rest (v = -65, u = b*v, no spikes).
        is_excitatory: Boolean array of shape (N,). True for excitatory neurons.
    """
    # Compute per-type neuron counts.  The fractions already encode the
    # excitatory / inhibitory split (RS+IB+CH = 80 %, FS+LTS = 20 %).
    # We honour ei_ratio by rescaling excitatory and inhibitory fractions.
    exc_types = ["RS", "IB", "CH"]
    inh_types = ["FS", "LTS"]

    exc_fracs_raw = jnp.array([_CELL_TYPES[t]["frac"] for t in exc_types])
    inh_fracs_raw = jnp.array([_CELL_TYPES[t]["frac"] for t in inh_types])

    # Normalise within E/I groups then scale by ei_ratio
    exc_fracs = exc_fracs_raw / exc_fracs_raw.sum() * ei_ratio
    inh_fracs = inh_fracs_raw / inh_fracs_raw.sum() * (1.0 - ei_ratio)

    ordered_fracs = jnp.concatenate([exc_fracs, inh_fracs])  # len 5
    counts = jnp.round(ordered_fracs * n_neurons).astype(jnp.int32)

    # Fix any rounding error by adjusting the largest group (RS)
    counts = counts.at[0].set(n_neurons - counts[1:].sum())

    ordered_types = exc_types + inh_types  # RS, IB, CH, FS, LTS

    # Build flat parameter arrays
    a_list, b_list, c_list, d_list = [], [], [], []
    for i, t in enumerate(ordered_types):
        n_t = int(counts[i])
        p = _CELL_TYPES[t]
        a_list.append(jnp.full(n_t, p["a"]))
        b_list.append(jnp.full(n_t, p["b"]))
        c_list.append(jnp.full(n_t, p["c"]))
        d_list.append(jnp.full(n_t, p["d"]))

    a = jnp.concatenate(a_list)
    b = jnp.concatenate(b_list)
    c = jnp.concatenate(c_list)
    d = jnp.concatenate(d_list)

    params = IzhikevichParams(a=a, b=b, c=c, d=d)

    # Initial state
    v = jnp.full(n_neurons, V_REST)
    u = b * v
    spikes = jnp.zeros(n_neurons, dtype=jnp.bool_)
    state = NeuronState(v=v, u=u, spikes=spikes)

    # Excitatory mask — first n_exc neurons are excitatory
    n_exc = int(jnp.sum(counts[:3]))
    is_excitatory = jnp.arange(n_neurons) < n_exc

    return params, state, is_excitatory


# ---------------------------------------------------------------------------
# Simulation step (JIT-compiled)
# ---------------------------------------------------------------------------

@jax.jit
def izhikevich_step(
    state: NeuronState,
    params: IzhikevichParams,
    I_ext: Array,
    dt: float = 0.5,
) -> NeuronState:
    """Advance the Izhikevich neuron population by one timestep.

    Uses semi-implicit Euler integration.  Izhikevich recommends two 0.5 ms
    half-steps per millisecond for numerical stability; when *dt* = 0.5 ms
    this corresponds to a single evaluation of the update equations.

    Args:
        state: Current NeuronState (v, u, spikes).
        params: IzhikevichParams (a, b, c, d).
        I_ext: External input current, shape (N,).
        dt: Integration timestep in ms (default 0.5).

    Returns:
        Updated NeuronState with new v, u, and spike indicators.
    """
    v, u = state.v, state.u
    a, b, c, d = params.a, params.b, params.c, params.d

    # Voltage update (semi-implicit Euler)
    v_new = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I_ext)

    # Recovery variable update
    u_new = u + dt * a * (b * v - u)

    # Spike detection
    spiked = v_new >= V_PEAK

    # Reset spiked neurons
    v_new = jnp.where(spiked, c, v_new)
    u_new = jnp.where(spiked, u_new + d, u_new)

    return NeuronState(v=v_new, u=u_new, spikes=spiked)


# ---------------------------------------------------------------------------
# Differentiable surrogate-gradient variant
# ---------------------------------------------------------------------------

def izhikevich_step_surrogate(
    state: NeuronState,
    params: IzhikevichParams,
    I_ext: Array,
    dt: float = 0.5,
    surrogate_fn=None,
    beta: float = 10.0,
) -> NeuronState:
    """Advance Izhikevich neurons by one timestep with surrogate gradients.

    The forward pass produces identical binary spikes to :func:`izhikevich_step`.
    The backward pass uses a smooth surrogate gradient through the spike
    threshold, enabling ``jax.grad`` of spike-count-based metrics.

    The reset is also made differentiable: instead of ``jnp.where(spiked, c, v)``
    (zero gradient through the boolean condition), it uses the soft form
    ``v * (1 - spike_f) + c * spike_f`` where ``spike_f`` is a float
    produced by the surrogate threshold.

    Args:
        state: Current NeuronState (v, u, spikes).
        params: IzhikevichParams (a, b, c, d).
        I_ext: External input current, shape (N,).
        dt: Integration timestep in ms (default 0.5).
        surrogate_fn: A surrogate threshold function from ``bl1.core.surrogate``
            (e.g. ``superspike_threshold``).  If None, falls back to a hard
            threshold (non-differentiable).
        beta: Sharpness parameter passed to the surrogate function.

    Returns:
        Updated NeuronState with new v, u, and spike indicators.
    """
    v, u = state.v, state.u
    a, b, c, d = params.a, params.b, params.c, params.d

    # Voltage update (semi-implicit Euler)
    v_new = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I_ext)

    # Recovery variable update
    u_new = u + dt * a * (b * v - u)

    # Spike detection via surrogate
    if surrogate_fn is not None:
        spiked_f = surrogate_fn(v_new, V_PEAK, beta)
    else:
        spiked_f = (v_new >= V_PEAK).astype(jnp.float32)

    # Soft reset (differentiable)
    v_new = v_new * (1.0 - spiked_f) + c * spiked_f
    u_new = u_new + d * spiked_f

    # Store float spikes (0.0/1.0) to preserve gradient chain.
    # The forward values are still binary, but the surrogate gradient
    # flows through spiked_f when differentiated.
    return NeuronState(v=v_new, u=u_new, spikes=spiked_f)
