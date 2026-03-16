"""JAX time-stepper for the BL-1 cortical culture simulator.

Provides a ``simulate`` function that advances the full neuron + synapse +
(optional) plasticity system through time using ``jax.lax.scan``, yielding
a spike raster as output.  The entire simulation loop is JIT-compiled.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    gaba_a_step,
)


# Type alias for the optional plasticity callback.
# Signature: (stdp_state, spikes, W_exc) -> (stdp_state, W_exc)
PlasticityFn = Callable[[Any, Array, Array], tuple[Any, Array]]


class SimulationResult(NamedTuple):
    """Outputs returned by :func:`simulate`."""
    final_neuron_state: NeuronState
    final_syn_state: SynapseState
    final_stdp_state: Any
    final_W_exc: Array
    spike_history: Array  # (T, N) boolean


def simulate(
    params: IzhikevichParams,
    init_state: NeuronState,
    syn_state: SynapseState,
    stdp_state: Any,
    W_exc: Array,
    W_inh: Array,
    I_external: Array,
    dt: float = 0.5,
    plasticity_fn: PlasticityFn | None = None,
) -> SimulationResult:
    """Run the full simulation for T timesteps.

    The function uses ``jax.lax.scan`` so the entire time loop is
    compiled into a single XLA program — no Python-level loop overhead.

    Args:
        params: IzhikevichParams for the population.
        init_state: Initial NeuronState.
        syn_state: Initial SynapseState.
        stdp_state: Initial STDP / plasticity state (arbitrary pytree).
            Pass ``None`` when plasticity is disabled.
        W_exc: Excitatory weight matrix, shape (N, N).  Dense or BCOO sparse.
        W_inh: Inhibitory weight matrix, shape (N, N).  Dense or BCOO sparse.
        I_external: External drive current, shape (T, N).  One row per
            timestep.
        dt: Integration timestep in ms (default 0.5).
        plasticity_fn: Optional callable with signature
            ``(stdp_state, spikes, W_exc) -> (stdp_state, W_exc)``.
            Called once per timestep when provided.

    Returns:
        A ``SimulationResult`` namedtuple containing final states and the
        full (T, N) boolean spike history.
    """

    # ----- inner scan body (closed over constants) -------------------------
    def _step_fn(carry, I_t):
        neuron_state, s_state, s_stdp, w_exc = carry

        # 1. Synaptic current from current conductances
        I_syn = compute_synaptic_current(s_state, neuron_state.v)

        # 2. Total input current
        I_total = I_syn + I_t

        # 3. Izhikevich neuron update
        neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

        # 4. Synapse conductance updates driven by new spikes
        spikes_f = neuron_state.spikes.astype(jnp.float32)
        new_g_ampa = ampa_step(s_state.g_ampa, spikes_f, w_exc, dt)
        new_g_gaba_a = gaba_a_step(s_state.g_gaba_a, spikes_f, W_inh, dt)
        s_state = SynapseState(g_ampa=new_g_ampa, g_gaba_a=new_g_gaba_a)

        # 5. Optional plasticity update
        if plasticity_fn is not None:
            s_stdp, w_exc = plasticity_fn(s_stdp, neuron_state.spikes, w_exc)

        new_carry = (neuron_state, s_state, s_stdp, w_exc)
        return new_carry, neuron_state.spikes

    # ----- run scan --------------------------------------------------------
    init_carry = (init_state, syn_state, stdp_state, W_exc)

    # Use jax.lax.scan; we need to strip the @jit from inner calls since
    # the outer jit will compile everything together.
    final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)

    final_neuron_state, final_syn_state, final_stdp_state, final_W_exc = final_carry
    return SimulationResult(
        final_neuron_state=final_neuron_state,
        final_syn_state=final_syn_state,
        final_stdp_state=final_stdp_state,
        final_W_exc=final_W_exc,
        spike_history=spike_history,
    )


# Provide a top-level jitted version for convenience.  Users who need to
# pass ``plasticity_fn=None`` (the default) can call this directly; those
# who supply a plasticity callback should jit the outer call themselves
# (or rely on tracing through ``simulate`` inside their own jitted code).
simulate_jit = jax.jit(simulate, static_argnames=("dt", "plasticity_fn"))
