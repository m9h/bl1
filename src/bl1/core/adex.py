"""Vectorized Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.

Implements the AdEx model (Brette & Gerstner 2005) with five cortical cell
types from Naud et al. (2008): Regular Spiking (RS), Bursting, Fast Spiking
(FS), Adapting, and Irregular.

The AdEx model is more biophysically grounded than Izhikevich while remaining
computationally efficient.  It uses an exponential voltage term for spike
initiation and a linear adaptation current.

Equations:
    C * dV/dt = -g_L * (V - E_L) + g_L * delta_T * exp((V - V_T) / delta_T)
                - w + I
    tau_w * dw/dt = a * (V - E_L) - w

Reset rule (when V >= V_peak):
    V -> V_reset
    w -> w + b

All operations are fully vectorized over the neuron population using JAX.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# AdEx cell-type parameter table (Naud et al. 2008)
# Excitatory types (80 % of population)
#   RS       – 64 %
#   Bursting –  8 %
#   Adapting –  8 %
# Inhibitory types (20 % of population)
#   FS        – 16 %
#   Irregular –  4 %
# ---------------------------------------------------------------------------

_CELL_TYPES = {
    "RS": {
        "C": 281.0, "g_L": 30.0, "E_L": -70.6, "delta_T": 2.0,
        "V_T": -50.4, "V_reset": -70.6, "V_peak": 20.0,
        "a": 4.0, "b": 80.5, "tau_w": 144.0,
        "frac": 0.64,
    },
    "Bursting": {
        "C": 281.0, "g_L": 30.0, "E_L": -70.6, "delta_T": 2.0,
        "V_T": -50.4, "V_reset": -47.2, "V_peak": 20.0,
        "a": 4.0, "b": 80.5, "tau_w": 20.0,
        "frac": 0.08,
    },
    "FS": {
        "C": 281.0, "g_L": 30.0, "E_L": -70.6, "delta_T": 2.0,
        "V_T": -50.4, "V_reset": -70.6, "V_peak": 20.0,
        "a": 4.0, "b": 0.5, "tau_w": 144.0,
        "frac": 0.16,
    },
    "Adapting": {
        "C": 281.0, "g_L": 30.0, "E_L": -70.6, "delta_T": 2.0,
        "V_T": -50.4, "V_reset": -70.6, "V_peak": 20.0,
        "a": 28.0, "b": 0.0, "tau_w": 144.0,
        "frac": 0.08,
    },
    "Irregular": {
        "C": 281.0, "g_L": 30.0, "E_L": -70.6, "delta_T": 2.0,
        "V_T": -50.4, "V_reset": -70.6, "V_peak": 20.0,
        "a": -10.0, "b": 30.0, "tau_w": 300.0,
        "frac": 0.04,
    },
}

# Ordered list so we can build contiguous blocks
_TYPE_ORDER = ["RS", "Bursting", "Adapting", "FS", "Irregular"]


# ---------------------------------------------------------------------------
# State containers (plain NamedTuples — JAX pytree compatible out of the box)
# ---------------------------------------------------------------------------

class AdExParams(NamedTuple):
    """Per-neuron AdEx model parameters."""
    C: Array       # (N,) membrane capacitance (pF)
    g_L: Array     # (N,) leak conductance (nS)
    E_L: Array     # (N,) leak reversal potential (mV)
    delta_T: Array # (N,) slope factor (mV)
    V_T: Array     # (N,) threshold voltage (mV)
    V_reset: Array # (N,) reset voltage (mV)
    V_peak: Array  # (N,) spike detection threshold (mV)
    a: Array       # (N,) subthreshold adaptation coupling (nS)
    b: Array       # (N,) spike-triggered adaptation increment (pA)
    tau_w: Array   # (N,) adaptation time constant (ms)


class AdExState(NamedTuple):
    """Per-neuron dynamic state variables."""
    v: Array      # (N,) membrane potential (mV)
    w: Array      # (N,) adaptation current (pA)
    spikes: Array # (N,) boolean spike indicator for the current timestep


# ---------------------------------------------------------------------------
# Population factory
# ---------------------------------------------------------------------------

def create_adex_population(
    key: Array,
    n_neurons: int,
    ei_ratio: float = 0.8,
) -> tuple[AdExParams, AdExState, Array]:
    """Create a mixed cortical population of AdEx neurons.

    Args:
        key: JAX PRNG key (used for future stochastic extensions;
             currently the population is deterministic).
        n_neurons: Total number of neurons.
        ei_ratio: Fraction of neurons that are excitatory (default 0.8).

    Returns:
        params: AdExParams with per-neuron parameters.
        state: AdExState initialised at rest (v = E_L, w = 0, no spikes).
        is_excitatory: Boolean array of shape (N,). True for excitatory neurons.
    """
    exc_types = ["RS", "Bursting", "Adapting"]
    inh_types = ["FS", "Irregular"]

    exc_fracs_raw = jnp.array([_CELL_TYPES[t]["frac"] for t in exc_types])
    inh_fracs_raw = jnp.array([_CELL_TYPES[t]["frac"] for t in inh_types])

    # Normalise within E/I groups then scale by ei_ratio
    exc_fracs = exc_fracs_raw / exc_fracs_raw.sum() * ei_ratio
    inh_fracs = inh_fracs_raw / inh_fracs_raw.sum() * (1.0 - ei_ratio)

    ordered_fracs = jnp.concatenate([exc_fracs, inh_fracs])  # len 5
    counts = jnp.round(ordered_fracs * n_neurons).astype(jnp.int32)

    # Fix any rounding error by adjusting the largest group (RS)
    counts = counts.at[0].set(n_neurons - counts[1:].sum())

    ordered_types = exc_types + inh_types  # RS, Bursting, Adapting, FS, Irregular

    # Build flat parameter arrays
    param_names = ["C", "g_L", "E_L", "delta_T", "V_T", "V_reset", "V_peak",
                   "a", "b", "tau_w"]
    param_lists = {name: [] for name in param_names}

    for i, t in enumerate(ordered_types):
        n_t = int(counts[i])
        p = _CELL_TYPES[t]
        for name in param_names:
            param_lists[name].append(jnp.full(n_t, p[name]))

    param_arrays = {name: jnp.concatenate(param_lists[name])
                    for name in param_names}

    params = AdExParams(**param_arrays)

    # Initial state: v at leak reversal, w = 0, no spikes
    v = jnp.full(n_neurons, _CELL_TYPES["RS"]["E_L"])
    w = jnp.zeros(n_neurons)
    spikes = jnp.zeros(n_neurons, dtype=jnp.bool_)
    state = AdExState(v=v, w=w, spikes=spikes)

    # Excitatory mask — first n_exc neurons are excitatory
    n_exc = int(jnp.sum(counts[:3]))
    is_excitatory = jnp.arange(n_neurons) < n_exc

    return params, state, is_excitatory


# ---------------------------------------------------------------------------
# Simulation step (JIT-compiled)
# ---------------------------------------------------------------------------

@jax.jit
def adex_step(
    state: AdExState,
    params: AdExParams,
    I_ext: Array,
    dt: float = 0.5,
) -> AdExState:
    """Advance the AdEx neuron population by one timestep.

    Uses forward Euler integration.

    Args:
        state: Current AdExState (v, w, spikes).
        params: AdExParams (C, g_L, E_L, delta_T, V_T, V_reset, V_peak,
                a, b, tau_w).
        I_ext: External input current in pA, shape (N,).  Synaptic currents
               from conductance-based synapses are in nS*mV = pA, so this is
               consistent.
        dt: Integration timestep in ms (default 0.5).

    Returns:
        Updated AdExState with new v, w, and spike indicators.
    """
    v, w = state.v, state.w
    C, g_L, E_L, delta_T, V_T, V_reset, V_peak, a, b, tau_w = params

    # Exponential term (clipped for numerical stability)
    exp_arg = jnp.clip((v - V_T) / delta_T, -20.0, 20.0)
    exp_term = g_L * delta_T * jnp.exp(exp_arg)

    # Voltage update
    dv = (-g_L * (v - E_L) + exp_term - w + I_ext) / C
    v_new = v + dt * dv

    # Adaptation current update
    dw = (a * (v - E_L) - w) / tau_w
    w_new = w + dt * dw

    # Spike detection
    spiked = v_new >= V_peak

    # Reset spiked neurons
    v_new = jnp.where(spiked, V_reset, v_new)
    w_new = jnp.where(spiked, w_new + b, w_new)

    return AdExState(v=v_new, w=w_new, spikes=spiked)


# ---------------------------------------------------------------------------
# Differentiable surrogate-gradient variant
# ---------------------------------------------------------------------------

def adex_step_surrogate(
    state: AdExState,
    params: AdExParams,
    I_ext: Array,
    dt: float = 0.5,
    surrogate_fn=None,
    beta: float = 10.0,
) -> AdExState:
    """Advance AdEx neurons by one timestep with surrogate gradients.

    The forward pass produces identical binary spikes to :func:`adex_step`.
    The backward pass uses a smooth surrogate gradient through the spike
    threshold, enabling ``jax.grad`` of spike-count-based metrics.

    The reset is also made differentiable: instead of ``jnp.where(spiked, ...)``
    (zero gradient through the boolean condition), it uses the soft form
    ``v * (1 - spike_f) + V_reset * spike_f``.

    Args:
        state: Current AdExState (v, w, spikes).
        params: AdExParams (C, g_L, E_L, delta_T, V_T, V_reset, V_peak,
                a, b, tau_w).
        I_ext: External input current in pA, shape (N,).
        dt: Integration timestep in ms (default 0.5).
        surrogate_fn: A surrogate threshold function from ``bl1.core.surrogate``
            (e.g. ``superspike_threshold``).  If None, falls back to a hard
            threshold (non-differentiable).
        beta: Sharpness parameter passed to the surrogate function.

    Returns:
        Updated AdExState with new v, w, and spike indicators.
    """
    v, w = state.v, state.w
    C, g_L, E_L, delta_T, V_T, V_reset, V_peak, a, b, tau_w = params

    # Exponential term (clipped for numerical stability)
    exp_arg = jnp.clip((v - V_T) / delta_T, -20.0, 20.0)
    exp_term = g_L * delta_T * jnp.exp(exp_arg)

    # Voltage update
    dv = (-g_L * (v - E_L) + exp_term - w + I_ext) / C
    v_new = v + dt * dv

    # Adaptation current update
    dw = (a * (v - E_L) - w) / tau_w
    w_new = w + dt * dw

    # Spike detection via surrogate
    if surrogate_fn is not None:
        spiked_f = surrogate_fn(v_new, V_peak, beta)
    else:
        spiked_f = (v_new >= V_peak).astype(jnp.float32)

    # Soft reset (differentiable)
    v_new = v_new * (1.0 - spiked_f) + V_reset * spiked_f
    w_new = w_new + b * spiked_f

    # Store float spikes (0.0/1.0) to preserve gradient chain.
    # The forward values are still binary, but the surrogate gradient
    # flows through spiked_f when differentiated.
    return AdExState(v=v_new, w=w_new, spikes=spiked_f)
