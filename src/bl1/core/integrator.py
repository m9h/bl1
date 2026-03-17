"""JAX time-stepper for the BL-1 cortical culture simulator.

Provides a ``simulate`` function that advances the full neuron + synapse +
(optional) plasticity system through time using ``jax.lax.scan``, yielding
a spike raster as output.  The entire simulation loop is JIT-compiled.

When axonal conduction delays are provided (``W_exc_delays`` and/or
``W_inh_delays``), the integrator inserts a :class:`DelayBufferState` into
the scan carry so that pre-synaptic spikes reach their post-synaptic targets
after the appropriate number of timesteps.  Without delay matrices the
behaviour is identical to the original instantaneous-transmission path.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bl1.core.delays import (
    DelayBufferState,
    compute_max_delay,
    delay_buffer_step,
    init_delay_buffer,
    read_delayed_spikes,
)
from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    TAU_AMPA,
    TAU_GABA_A,
    TAU_GABA_B_DECAY,
    TAU_GABA_B_RISE,
    TAU_NMDA_DECAY,
    TAU_NMDA_RISE,
    _GABA_B_NORM,
    _NMDA_NORM,
    ampa_step,
    compute_synaptic_current,
    gaba_a_step,
    gaba_b_step,
    nmda_step,
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
    W_exc_delays: Array | None = None,
    W_inh_delays: Array | None = None,
) -> SimulationResult:
    """Run the full simulation for T timesteps.

    The function uses ``jax.lax.scan`` so the entire time loop is
    compiled into a single XLA program -- no Python-level loop overhead.

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
        W_exc_delays: Dense (N, N) integer delay matrix for excitatory
            synapses, in timesteps.  Entry ``[j, i]`` is the delay from
            pre-neuron *i* to post-neuron *j*.  Pass ``None`` for
            instantaneous (zero-delay) excitatory transmission.
        W_inh_delays: Dense (N, N) integer delay matrix for inhibitory
            synapses.  Same convention as *W_exc_delays*.  Pass ``None``
            for instantaneous inhibitory transmission.

    Returns:
        A ``SimulationResult`` namedtuple containing final states and the
        full (T, N) boolean spike history.
    """
    # Decide at Python level whether to use the delay path.
    # This is a compile-time constant so it does not cause tracing issues.
    use_delays = (W_exc_delays is not None) or (W_inh_delays is not None)

    if use_delays:
        # Compute max_delay BEFORE tracing (must be a concrete Python int)
        max_delay = 1
        if W_exc_delays is not None:
            max_delay = max(max_delay, compute_max_delay(W_exc_delays))
        if W_inh_delays is not None:
            max_delay = max(max_delay, compute_max_delay(W_inh_delays))
        return _simulate_with_delays(
            params, init_state, syn_state, stdp_state,
            W_exc, W_inh, I_external, dt, plasticity_fn,
            W_exc_delays, W_inh_delays, max_delay,
        )
    else:
        return _simulate_no_delays(
            params, init_state, syn_state, stdp_state,
            W_exc, W_inh, I_external, dt, plasticity_fn,
        )


# =========================================================================
# Path 1: original instantaneous transmission (no delays)
# =========================================================================

def _simulate_no_delays(
    params, init_state, syn_state, stdp_state,
    W_exc, W_inh, I_external, dt, plasticity_fn,
) -> SimulationResult:
    """Simulation loop without axonal delays (original behaviour)."""

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

        # Phase 1: single-exponential receptors
        new_g_ampa = ampa_step(s_state.g_ampa, spikes_f, w_exc, dt)
        new_g_gaba_a = gaba_a_step(s_state.g_gaba_a, spikes_f, W_inh, dt)

        # Phase 2: dual-exponential receptors
        new_nmda_rise, new_nmda_decay, _ = nmda_step(
            s_state.g_nmda_rise, s_state.g_nmda_decay,
            spikes_f, w_exc, dt,
        )
        new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step(
            s_state.g_gaba_b_rise, s_state.g_gaba_b_decay,
            spikes_f, W_inh, dt,
        )

        s_state = SynapseState(
            g_ampa=new_g_ampa,
            g_gaba_a=new_g_gaba_a,
            g_nmda_rise=new_nmda_rise,
            g_nmda_decay=new_nmda_decay,
            g_gaba_b_rise=new_gaba_b_rise,
            g_gaba_b_decay=new_gaba_b_decay,
        )

        # 5. Optional plasticity update
        if plasticity_fn is not None:
            s_stdp, w_exc = plasticity_fn(s_stdp, neuron_state.spikes, w_exc)

        new_carry = (neuron_state, s_state, s_stdp, w_exc)
        return new_carry, neuron_state.spikes

    init_carry = (init_state, syn_state, stdp_state, W_exc)
    final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)

    final_neuron_state, final_syn_state, final_stdp_state, final_W_exc = final_carry
    return SimulationResult(
        final_neuron_state=final_neuron_state,
        final_syn_state=final_syn_state,
        final_stdp_state=final_stdp_state,
        final_W_exc=final_W_exc,
        spike_history=spike_history,
    )


# =========================================================================
# Path 2: delayed spike transmission via ring buffer
# =========================================================================

def _simulate_with_delays(
    params, init_state, syn_state, stdp_state,
    W_exc, W_inh, I_external, dt, plasticity_fn,
    W_exc_delays, W_inh_delays, max_delay,
) -> SimulationResult:
    """Simulation loop with axonal conduction delays.

    Maintains a :class:`DelayBufferState` in the scan carry. Each step:
    1. Write current spikes into the ring buffer.
    2. Read delayed spikes for each receptor type.
    3. Compute conductance updates using the delayed input.
    """
    N = init_state.v.shape[0]

    # If one delay matrix is None, create a unit-delay fallback so that
    # the code path is uniform (delay=1 reproduces instantaneous semantics
    # for a buffer that was just written into).
    if W_exc_delays is None:
        W_exc_delays = jnp.ones((N, N), dtype=jnp.int32)
    if W_inh_delays is None:
        W_inh_delays = jnp.ones((N, N), dtype=jnp.int32)

    delay_buf_init = init_delay_buffer(N, max_delay)

    def _step_fn(carry, I_t):
        neuron_state, s_state, s_stdp, w_exc, delay_buf = carry

        # 1. Synaptic current from current conductances
        I_syn = compute_synaptic_current(s_state, neuron_state.v)

        # 2. Total input current
        I_total = I_syn + I_t

        # 3. Izhikevich neuron update
        neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

        # 4. Write new spikes into delay buffer
        spikes_f = neuron_state.spikes.astype(jnp.float32)
        delay_buf = delay_buffer_step(delay_buf, spikes_f)

        # 5. Read delayed synaptic input for each receptor type
        #    read_delayed_spikes returns the equivalent of
        #    ``weights @ delayed_spikes`` -- i.e. postsynaptic input
        #    with per-synapse delays applied.
        g_input_exc = read_delayed_spikes(delay_buf, W_exc_delays, w_exc)
        g_input_inh = read_delayed_spikes(delay_buf, W_inh_delays, W_inh)

        # Phase 1: AMPA & GABA_A (single-exponential decay + delayed input)
        new_g_ampa = s_state.g_ampa * jnp.exp(-dt / TAU_AMPA) + g_input_exc
        new_g_gaba_a = s_state.g_gaba_a * jnp.exp(-dt / TAU_GABA_A) + g_input_inh

        # Phase 2: NMDA (dual-exponential, excitatory, with normalisation)
        new_nmda_rise = (
            s_state.g_nmda_rise * jnp.exp(-dt / TAU_NMDA_RISE)
            + g_input_exc * _NMDA_NORM
        )
        new_nmda_decay = (
            s_state.g_nmda_decay * jnp.exp(-dt / TAU_NMDA_DECAY)
            + g_input_exc * _NMDA_NORM
        )

        # Phase 2: GABA_B (dual-exponential, inhibitory, with normalisation)
        new_gaba_b_rise = (
            s_state.g_gaba_b_rise * jnp.exp(-dt / TAU_GABA_B_RISE)
            + g_input_inh * _GABA_B_NORM
        )
        new_gaba_b_decay = (
            s_state.g_gaba_b_decay * jnp.exp(-dt / TAU_GABA_B_DECAY)
            + g_input_inh * _GABA_B_NORM
        )

        s_state = SynapseState(
            g_ampa=new_g_ampa,
            g_gaba_a=new_g_gaba_a,
            g_nmda_rise=new_nmda_rise,
            g_nmda_decay=new_nmda_decay,
            g_gaba_b_rise=new_gaba_b_rise,
            g_gaba_b_decay=new_gaba_b_decay,
        )

        # 6. Optional plasticity update
        if plasticity_fn is not None:
            s_stdp, w_exc = plasticity_fn(s_stdp, neuron_state.spikes, w_exc)

        new_carry = (neuron_state, s_state, s_stdp, w_exc, delay_buf)
        return new_carry, neuron_state.spikes

    init_carry = (init_state, syn_state, stdp_state, W_exc, delay_buf_init)
    final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)

    final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _ = final_carry
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
