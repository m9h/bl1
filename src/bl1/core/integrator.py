"""JAX time-stepper for the BL-1 cortical culture simulator.

Provides a ``simulate`` function that advances the full neuron + synapse +
(optional) plasticity system through time using ``jax.lax.scan``, yielding
a spike raster as output.  The entire simulation loop is JIT-compiled.

When axonal conduction delays are provided (``W_exc_delays`` and/or
``W_inh_delays``), the integrator inserts a :class:`DelayBufferState` into
the scan carry so that pre-synaptic spikes reach their post-synaptic targets
after the appropriate number of timesteps.  Without delay matrices the
behaviour is identical to the original instantaneous-transmission path.

When short-term plasticity parameters are provided (``stp_params``), each
timestep calls :func:`stp_step` to modulate presynaptic spike amplitudes
before they drive conductance updates.  This enables synaptic depression
(excitatory) and facilitation (inhibitory) as described by Tsodyks & Markram
(1997).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bl1.core.delays import (
    compute_max_delay,
    delay_buffer_step,
    init_delay_buffer,
    read_delayed_spikes,
)
from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    izhikevich_step,
    izhikevich_step_surrogate,
)
from bl1.core.pallas_ops import bcoo_to_csc
from bl1.core.sparse_ops import bcoo_to_raw
from bl1.core.surrogate import superspike_threshold
from bl1.core.synapses import (
    _GABA_B_NORM,
    _NMDA_NORM,
    TAU_AMPA,
    TAU_GABA_A,
    TAU_GABA_B_DECAY,
    TAU_GABA_B_RISE,
    TAU_NMDA_DECAY,
    TAU_NMDA_RISE,
    SynapseState,
    ampa_step,
    ampa_step_event,
    ampa_step_fast,
    compute_synaptic_current,
    gaba_a_step,
    gaba_a_step_event,
    gaba_a_step_fast,
    gaba_b_step,
    gaba_b_step_event,
    gaba_b_step_fast,
    nmda_step,
    nmda_step_event,
    nmda_step_fast,
)
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step

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
    stp_params: STPParams | None = None,
    use_fast_sparse: bool = False,
    use_event_driven: bool = False,
    max_active: int = 5000,
    surrogate: bool = False,
    surrogate_fn=None,
    surrogate_beta: float = 10.0,
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
        stp_params: Optional :class:`STPParams` for Tsodyks-Markram
            short-term plasticity.  When provided, presynaptic spikes
            are modulated by a per-neuron scale factor (facilitation /
            depression) before driving conductance updates.  Pass
            ``None`` to disable STP (default).
        use_fast_sparse: When ``True``, replace BCOO sparse matmul
            (``weights @ spikes``) with :func:`fast_sparse_input` using
            pre-extracted COO arrays and ``segment_sum``.  This is
            significantly faster for large sparse networks (50K+ neurons).
            Requires that ``W_exc`` and ``W_inh`` are BCOO arrays.
            Currently only supported on the no-delays path.
            Default ``False``.
        use_event_driven: When ``True``, use the CSC event-driven sparse
            kernel that only processes synapses from neurons that actually
            spiked.  This achieves O(active_synapses) per timestep instead
            of O(total_synapses).  At typical cortical firing rates (5 Hz)
            with 100K neurons, this is ~400x less work than the full
            matmul.  Requires BCOO weight matrices.  Not compatible with
            delays, plasticity, or surrogate mode.  Default ``False``.
        max_active: Maximum number of simultaneously spiking neurons to
            process in the event-driven path.  Must be a compile-time
            constant.  Default 5000 (sufficient for 100K neurons at ~5 Hz).
        surrogate: When ``True``, use :func:`izhikevich_step_surrogate`
            instead of the hard-threshold :func:`izhikevich_step`.  This
            makes the full simulation differentiable via surrogate
            gradients (``jax.grad`` of spike-count metrics will produce
            non-zero gradients).  The initial state's ``spikes`` field
            is automatically cast to float32 for scan carry compatibility.

            .. note:: Surrogate mode is currently only supported on the
               no-delay, non-fast-sparse path.  Passing ``surrogate=True``
               together with delay matrices or ``use_fast_sparse=True``
               will raise :class:`ValueError`.

            Default ``False``.
        surrogate_fn: Surrogate gradient function from
            :mod:`bl1.core.surrogate` (e.g. ``superspike_threshold``).
            Only used when ``surrogate=True``.  Defaults to
            :func:`superspike_threshold` when ``None``.
        surrogate_beta: Sharpness parameter passed to the surrogate
            function.  Higher values approximate the true (hard) threshold
            more closely but produce noisier gradients.  Default 10.0.

    Returns:
        A ``SimulationResult`` namedtuple containing final states and the
        full (T, N) spike history.  When ``surrogate=True`` the spike
        history dtype is float32 (values 0.0/1.0) instead of bool.
    """
    # Decide at Python level whether to use the delay path.
    # This is a compile-time constant so it does not cause tracing issues.
    use_delays = (W_exc_delays is not None) or (W_inh_delays is not None)

    # Validate surrogate constraints
    if surrogate and use_delays:
        raise ValueError(
            "surrogate=True is not yet supported with axonal delays. "
            "Use the no-delay path for differentiable simulation."
        )
    if surrogate and use_fast_sparse:
        raise ValueError(
            "surrogate=True is not yet supported with use_fast_sparse=True. "
            "Use the standard no-delay path for differentiable simulation."
        )
    if surrogate and use_event_driven:
        raise ValueError(
            "surrogate=True is not yet supported with use_event_driven=True. "
            "Use the standard no-delay path for differentiable simulation."
        )
    if use_event_driven and use_delays:
        raise ValueError("use_event_driven=True is not yet supported with axonal delays.")
    if use_event_driven and use_fast_sparse:
        raise ValueError(
            "use_event_driven and use_fast_sparse are mutually exclusive. "
            "Choose one sparse acceleration strategy."
        )

    # Resolve default surrogate function
    if surrogate and surrogate_fn is None:
        surrogate_fn = superspike_threshold

    # When using surrogate, cast init_state.spikes to float32 so the
    # scan carry pytree is consistent (izhikevich_step_surrogate returns
    # float32 spikes).
    if surrogate:
        init_state = NeuronState(
            v=init_state.v,
            u=init_state.u,
            spikes=init_state.spikes.astype(jnp.float32),
        )

    if use_delays:
        # Compute max_delay BEFORE tracing (must be a concrete Python int)
        max_delay = 1
        if W_exc_delays is not None:
            max_delay = max(max_delay, compute_max_delay(W_exc_delays))
        if W_inh_delays is not None:
            max_delay = max(max_delay, compute_max_delay(W_inh_delays))
        return _simulate_with_delays(
            params,
            init_state,
            syn_state,
            stdp_state,
            W_exc,
            W_inh,
            I_external,
            dt,
            plasticity_fn,
            W_exc_delays,
            W_inh_delays,
            max_delay,
            stp_params,
        )
    elif use_event_driven:
        return _simulate_no_delays_event_driven(
            params,
            init_state,
            syn_state,
            stdp_state,
            W_exc,
            W_inh,
            I_external,
            dt,
            plasticity_fn,
            stp_params,
            max_active,
        )
    elif use_fast_sparse:
        return _simulate_no_delays_fast_sparse(
            params,
            init_state,
            syn_state,
            stdp_state,
            W_exc,
            W_inh,
            I_external,
            dt,
            plasticity_fn,
            stp_params,
        )
    else:
        return _simulate_no_delays(
            params,
            init_state,
            syn_state,
            stdp_state,
            W_exc,
            W_inh,
            I_external,
            dt,
            plasticity_fn,
            stp_params,
            surrogate=surrogate,
            surrogate_fn=surrogate_fn,
            surrogate_beta=surrogate_beta,
        )


# =========================================================================
# Path 1: original instantaneous transmission (no delays)
# =========================================================================


def _simulate_no_delays(
    params,
    init_state,
    syn_state,
    stdp_state,
    W_exc,
    W_inh,
    I_external,
    dt,
    plasticity_fn,
    stp_params=None,
    surrogate=False,
    surrogate_fn=None,
    surrogate_beta=10.0,
) -> SimulationResult:
    """Simulation loop without axonal delays (original behaviour).

    When *stp_params* is provided, presynaptic spikes are modulated by
    Tsodyks-Markram short-term plasticity before driving conductance
    updates.  The STP ``scale`` vector replaces raw ``spikes_f`` in
    synapse step calls -- it already incorporates the spike indicator
    (zero for non-spiking neurons) and the per-neuron STP modulation.

    When *surrogate* is True, uses :func:`izhikevich_step_surrogate`
    for the neuron update, making the simulation differentiable via
    surrogate gradients.
    """
    use_stp = stp_params is not None
    use_surrogate = surrogate

    # Helper: choose the appropriate neuron step function.
    # This is a Python-level ``if`` so it creates a separate scan body
    # closure, keeping the pytree structure consistent within each path.
    def _neuron_step(neuron_state, I_total):
        if use_surrogate:
            return izhikevich_step_surrogate(
                neuron_state,
                params,
                I_total,
                dt,
                surrogate_fn=surrogate_fn,
                beta=surrogate_beta,
            )
        else:
            return izhikevich_step(neuron_state, params, I_total, dt)

    if use_stp:
        N = init_state.v.shape[0]
        stp_state_init = init_stp_state(N, stp_params)

        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc, stp_st = carry

            # 1. Synaptic current from current conductances
            I_syn = compute_synaptic_current(s_state, neuron_state.v)

            # 2. Total input current
            I_total = I_syn + I_t

            # 3. Izhikevich neuron update
            neuron_state = _neuron_step(neuron_state, I_total)

            # 4. STP modulation: scale already includes spike indicator
            stp_st, scale = stp_step(stp_st, stp_params, neuron_state.spikes, dt)

            # 5. Synapse conductance updates driven by STP-modulated spikes
            # Phase 1: single-exponential receptors
            new_g_ampa = ampa_step(s_state.g_ampa, scale, w_exc, dt)
            new_g_gaba_a = gaba_a_step(s_state.g_gaba_a, scale, W_inh, dt)

            # Phase 2: dual-exponential receptors
            new_nmda_rise, new_nmda_decay, _ = nmda_step(
                s_state.g_nmda_rise,
                s_state.g_nmda_decay,
                scale,
                w_exc,
                dt,
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step(
                s_state.g_gaba_b_rise,
                s_state.g_gaba_b_decay,
                scale,
                W_inh,
                dt,
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

            new_carry = (neuron_state, s_state, s_stdp, w_exc, stp_st)
            return new_carry, neuron_state.spikes

        init_carry = (init_state, syn_state, stdp_state, W_exc, stp_state_init)
        final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)

        final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _ = final_carry
    else:

        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc = carry

            # 1. Synaptic current from current conductances
            I_syn = compute_synaptic_current(s_state, neuron_state.v)

            # 2. Total input current
            I_total = I_syn + I_t

            # 3. Izhikevich neuron update
            neuron_state = _neuron_step(neuron_state, I_total)

            # 4. Synapse conductance updates driven by new spikes
            spikes_f = neuron_state.spikes.astype(jnp.float32)

            # Phase 1: single-exponential receptors
            new_g_ampa = ampa_step(s_state.g_ampa, spikes_f, w_exc, dt)
            new_g_gaba_a = gaba_a_step(s_state.g_gaba_a, spikes_f, W_inh, dt)

            # Phase 2: dual-exponential receptors
            new_nmda_rise, new_nmda_decay, _ = nmda_step(
                s_state.g_nmda_rise,
                s_state.g_nmda_decay,
                spikes_f,
                w_exc,
                dt,
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step(
                s_state.g_gaba_b_rise,
                s_state.g_gaba_b_decay,
                spikes_f,
                W_inh,
                dt,
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
# Path 1b: fast sparse — segment_sum instead of BCOO matmul (no delays)
# =========================================================================


def _simulate_no_delays_fast_sparse(
    params,
    init_state,
    syn_state,
    stdp_state,
    W_exc,
    W_inh,
    I_external,
    dt,
    plasticity_fn,
    stp_params=None,
) -> SimulationResult:
    """Simulation loop using fast_sparse_input instead of BCOO matmul.

    Pre-extracts COO arrays from BCOO weight matrices before the scan
    loop, then uses :func:`fast_sparse_input` (gather-multiply + segment_sum)
    which avoids BCOO's internal dispatch overhead and is significantly
    faster for large sparse networks.

    Note: STP support is included.  Plasticity that modifies W_exc
    dynamically is supported but the raw arrays are re-extracted on each
    step from the carry (the plasticity_fn returns updated W_exc, so we
    must re-extract).  For non-plastic runs the raw arrays are captured
    in the closure as constants.
    """
    N = init_state.v.shape[0]
    use_stp = stp_params is not None
    use_plasticity = plasticity_fn is not None

    # Pre-extract COO arrays from BCOO matrices.
    # W_inh is never modified so we extract once.
    raw_inh = bcoo_to_raw(W_inh)
    inh_data, inh_rows, inh_cols = raw_inh.data, raw_inh.rows, raw_inh.cols

    if use_stp:
        stp_state_init = init_stp_state(N, stp_params)

        if not use_plasticity:
            # W_exc is constant — pre-extract once
            raw_exc = bcoo_to_raw(W_exc)
            exc_data, exc_rows, exc_cols = raw_exc.data, raw_exc.rows, raw_exc.cols

            def _step_fn(carry, I_t):
                neuron_state, s_state, s_stdp, w_exc, stp_st = carry

                I_syn = compute_synaptic_current(s_state, neuron_state.v)
                I_total = I_syn + I_t
                neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

                stp_st, scale = stp_step(stp_st, stp_params, neuron_state.spikes, dt)

                # Phase 1
                new_g_ampa = ampa_step_fast(
                    s_state.g_ampa, scale, exc_data, exc_rows, exc_cols, N, dt
                )
                new_g_gaba_a = gaba_a_step_fast(
                    s_state.g_gaba_a, scale, inh_data, inh_rows, inh_cols, N, dt
                )

                # Phase 2
                new_nmda_rise, new_nmda_decay, _ = nmda_step_fast(
                    s_state.g_nmda_rise,
                    s_state.g_nmda_decay,
                    scale,
                    exc_data,
                    exc_rows,
                    exc_cols,
                    N,
                    dt,
                )
                new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_fast(
                    s_state.g_gaba_b_rise,
                    s_state.g_gaba_b_decay,
                    scale,
                    inh_data,
                    inh_rows,
                    inh_cols,
                    N,
                    dt,
                )

                s_state = SynapseState(
                    g_ampa=new_g_ampa,
                    g_gaba_a=new_g_gaba_a,
                    g_nmda_rise=new_nmda_rise,
                    g_nmda_decay=new_nmda_decay,
                    g_gaba_b_rise=new_gaba_b_rise,
                    g_gaba_b_decay=new_gaba_b_decay,
                )

                new_carry = (neuron_state, s_state, s_stdp, w_exc, stp_st)
                return new_carry, neuron_state.spikes

            init_carry = (init_state, syn_state, stdp_state, W_exc, stp_state_init)
            final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)
            final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _ = final_carry
        else:
            # Plasticity modifies W_exc — fall back to BCOO matmul for exc
            # (since raw arrays would need re-extraction each step)
            def _step_fn(carry, I_t):
                neuron_state, s_state, s_stdp, w_exc, stp_st = carry

                I_syn = compute_synaptic_current(s_state, neuron_state.v)
                I_total = I_syn + I_t
                neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

                stp_st, scale = stp_step(stp_st, stp_params, neuron_state.spikes, dt)

                # Use BCOO for exc (plasticity may change it), fast for inh
                new_g_ampa = ampa_step(s_state.g_ampa, scale, w_exc, dt)
                new_g_gaba_a = gaba_a_step_fast(
                    s_state.g_gaba_a, scale, inh_data, inh_rows, inh_cols, N, dt
                )

                new_nmda_rise, new_nmda_decay, _ = nmda_step(
                    s_state.g_nmda_rise, s_state.g_nmda_decay, scale, w_exc, dt
                )
                new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_fast(
                    s_state.g_gaba_b_rise,
                    s_state.g_gaba_b_decay,
                    scale,
                    inh_data,
                    inh_rows,
                    inh_cols,
                    N,
                    dt,
                )

                s_state = SynapseState(
                    g_ampa=new_g_ampa,
                    g_gaba_a=new_g_gaba_a,
                    g_nmda_rise=new_nmda_rise,
                    g_nmda_decay=new_nmda_decay,
                    g_gaba_b_rise=new_gaba_b_rise,
                    g_gaba_b_decay=new_gaba_b_decay,
                )

                s_stdp, w_exc = plasticity_fn(s_stdp, neuron_state.spikes, w_exc)

                new_carry = (neuron_state, s_state, s_stdp, w_exc, stp_st)
                return new_carry, neuron_state.spikes

            init_carry = (init_state, syn_state, stdp_state, W_exc, stp_state_init)
            final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)
            final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _ = final_carry

    elif not use_plasticity:
        # No STP, no plasticity — both weight matrices are constant
        raw_exc = bcoo_to_raw(W_exc)
        exc_data, exc_rows, exc_cols = raw_exc.data, raw_exc.rows, raw_exc.cols

        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc = carry

            I_syn = compute_synaptic_current(s_state, neuron_state.v)
            I_total = I_syn + I_t
            neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

            spikes_f = neuron_state.spikes.astype(jnp.float32)

            # Phase 1
            new_g_ampa = ampa_step_fast(
                s_state.g_ampa, spikes_f, exc_data, exc_rows, exc_cols, N, dt
            )
            new_g_gaba_a = gaba_a_step_fast(
                s_state.g_gaba_a, spikes_f, inh_data, inh_rows, inh_cols, N, dt
            )

            # Phase 2
            new_nmda_rise, new_nmda_decay, _ = nmda_step_fast(
                s_state.g_nmda_rise,
                s_state.g_nmda_decay,
                spikes_f,
                exc_data,
                exc_rows,
                exc_cols,
                N,
                dt,
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_fast(
                s_state.g_gaba_b_rise,
                s_state.g_gaba_b_decay,
                spikes_f,
                inh_data,
                inh_rows,
                inh_cols,
                N,
                dt,
            )

            s_state = SynapseState(
                g_ampa=new_g_ampa,
                g_gaba_a=new_g_gaba_a,
                g_nmda_rise=new_nmda_rise,
                g_nmda_decay=new_nmda_decay,
                g_gaba_b_rise=new_gaba_b_rise,
                g_gaba_b_decay=new_gaba_b_decay,
            )

            new_carry = (neuron_state, s_state, s_stdp, w_exc)
            return new_carry, neuron_state.spikes

        init_carry = (init_state, syn_state, stdp_state, W_exc)
        final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)
        final_neuron_state, final_syn_state, final_stdp_state, final_W_exc = final_carry

    else:
        # Plasticity but no STP — use fast for inh, BCOO for exc
        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc = carry

            I_syn = compute_synaptic_current(s_state, neuron_state.v)
            I_total = I_syn + I_t
            neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

            spikes_f = neuron_state.spikes.astype(jnp.float32)

            # Use BCOO for exc (plasticity may change it), fast for inh
            new_g_ampa = ampa_step(s_state.g_ampa, spikes_f, w_exc, dt)
            new_g_gaba_a = gaba_a_step_fast(
                s_state.g_gaba_a, spikes_f, inh_data, inh_rows, inh_cols, N, dt
            )

            new_nmda_rise, new_nmda_decay, _ = nmda_step(
                s_state.g_nmda_rise, s_state.g_nmda_decay, spikes_f, w_exc, dt
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_fast(
                s_state.g_gaba_b_rise,
                s_state.g_gaba_b_decay,
                spikes_f,
                inh_data,
                inh_rows,
                inh_cols,
                N,
                dt,
            )

            s_state = SynapseState(
                g_ampa=new_g_ampa,
                g_gaba_a=new_g_gaba_a,
                g_nmda_rise=new_nmda_rise,
                g_nmda_decay=new_nmda_decay,
                g_gaba_b_rise=new_gaba_b_rise,
                g_gaba_b_decay=new_gaba_b_decay,
            )

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
# Path 1c: event-driven — CSC format, only active synapses (no delays)
# =========================================================================


def _simulate_no_delays_event_driven(
    params,
    init_state,
    syn_state,
    stdp_state,
    W_exc,
    W_inh,
    I_external,
    dt,
    plasticity_fn,
    stp_params=None,
    max_active=5000,
) -> SimulationResult:
    """Simulation loop using event-driven CSC sparse kernel.

    Pre-converts BCOO weight matrices to CSC format before the scan loop,
    then uses :func:`event_driven_input` which only processes outgoing
    synapses from neurons that actually spiked.  At typical cortical
    firing rates this is ~400x less work than processing all nonzeros.

    Note: Plasticity that modifies W_exc is NOT supported on this path
    because CSC conversion is done once before the scan.  If plasticity_fn
    is provided, a ValueError is raised.
    """
    N = init_state.v.shape[0]
    use_stp = stp_params is not None
    use_plasticity = plasticity_fn is not None

    if use_plasticity:
        raise ValueError(
            "Event-driven path does not support dynamic weight plasticity. "
            "CSC format is pre-computed and cannot be updated during the scan. "
            "Use use_fast_sparse=True with plasticity instead."
        )

    # Pre-convert BCOO matrices to CSC format (once, before scan).
    csc_exc = bcoo_to_csc(W_exc)
    csc_inh = bcoo_to_csc(W_inh)

    if use_stp:
        stp_state_init = init_stp_state(N, stp_params)

        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc, stp_st = carry

            I_syn = compute_synaptic_current(s_state, neuron_state.v)
            I_total = I_syn + I_t
            neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

            stp_st, scale = stp_step(stp_st, stp_params, neuron_state.spikes, dt)

            # Phase 1
            new_g_ampa = ampa_step_event(s_state.g_ampa, scale, csc_exc, max_active, dt)
            new_g_gaba_a = gaba_a_step_event(s_state.g_gaba_a, scale, csc_inh, max_active, dt)

            # Phase 2
            new_nmda_rise, new_nmda_decay, _ = nmda_step_event(
                s_state.g_nmda_rise, s_state.g_nmda_decay, scale, csc_exc, max_active, dt
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_event(
                s_state.g_gaba_b_rise, s_state.g_gaba_b_decay, scale, csc_inh, max_active, dt
            )

            s_state = SynapseState(
                g_ampa=new_g_ampa,
                g_gaba_a=new_g_gaba_a,
                g_nmda_rise=new_nmda_rise,
                g_nmda_decay=new_nmda_decay,
                g_gaba_b_rise=new_gaba_b_rise,
                g_gaba_b_decay=new_gaba_b_decay,
            )

            new_carry = (neuron_state, s_state, s_stdp, w_exc, stp_st)
            return new_carry, neuron_state.spikes

        init_carry = (init_state, syn_state, stdp_state, W_exc, stp_state_init)
        final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)
        final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _ = final_carry

    else:
        # No STP, no plasticity — simplest event-driven path
        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc = carry

            I_syn = compute_synaptic_current(s_state, neuron_state.v)
            I_total = I_syn + I_t
            neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

            spikes_f = neuron_state.spikes.astype(jnp.float32)

            # Phase 1
            new_g_ampa = ampa_step_event(s_state.g_ampa, spikes_f, csc_exc, max_active, dt)
            new_g_gaba_a = gaba_a_step_event(s_state.g_gaba_a, spikes_f, csc_inh, max_active, dt)

            # Phase 2
            new_nmda_rise, new_nmda_decay, _ = nmda_step_event(
                s_state.g_nmda_rise, s_state.g_nmda_decay, spikes_f, csc_exc, max_active, dt
            )
            new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step_event(
                s_state.g_gaba_b_rise, s_state.g_gaba_b_decay, spikes_f, csc_inh, max_active, dt
            )

            s_state = SynapseState(
                g_ampa=new_g_ampa,
                g_gaba_a=new_g_gaba_a,
                g_nmda_rise=new_nmda_rise,
                g_nmda_decay=new_nmda_decay,
                g_gaba_b_rise=new_gaba_b_rise,
                g_gaba_b_decay=new_gaba_b_decay,
            )

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
    params,
    init_state,
    syn_state,
    stdp_state,
    W_exc,
    W_inh,
    I_external,
    dt,
    plasticity_fn,
    W_exc_delays,
    W_inh_delays,
    max_delay,
    stp_params=None,
) -> SimulationResult:
    """Simulation loop with axonal conduction delays.

    Maintains a :class:`DelayBufferState` in the scan carry. Each step:
    1. Write current spikes into the ring buffer.
    2. Read delayed spikes for each receptor type.
    3. Compute conductance updates using the delayed input.

    When *stp_params* is provided, the STP-modulated ``scale`` vector
    (which already incorporates the spike indicator) is written into the
    delay buffer instead of raw ``spikes_f``.  Delayed reads then
    automatically carry the STP amplitude.
    """
    N = init_state.v.shape[0]
    use_stp = stp_params is not None

    # If one delay matrix is None, create a unit-delay fallback so that
    # the code path is uniform (delay=1 reproduces instantaneous semantics
    # for a buffer that was just written into).
    if W_exc_delays is None:
        W_exc_delays = jnp.ones((N, N), dtype=jnp.int32)
    if W_inh_delays is None:
        W_inh_delays = jnp.ones((N, N), dtype=jnp.int32)

    delay_buf_init = init_delay_buffer(N, max_delay)

    if use_stp:
        stp_state_init = init_stp_state(N, stp_params)

        def _step_fn(carry, I_t):
            neuron_state, s_state, s_stdp, w_exc, delay_buf, stp_st = carry

            # 1. Synaptic current from current conductances
            I_syn = compute_synaptic_current(s_state, neuron_state.v)

            # 2. Total input current
            I_total = I_syn + I_t

            # 3. Izhikevich neuron update
            neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

            # 4. STP modulation: scale includes spike indicator
            stp_st, scale = stp_step(stp_st, stp_params, neuron_state.spikes, dt)

            # 5. Write STP-modulated spikes into delay buffer
            delay_buf = delay_buffer_step(delay_buf, scale)

            # 6. Read delayed synaptic input for each receptor type
            g_input_exc = read_delayed_spikes(delay_buf, W_exc_delays, w_exc)
            g_input_inh = read_delayed_spikes(delay_buf, W_inh_delays, W_inh)

            # Phase 1: AMPA & GABA_A (single-exponential decay + delayed input)
            new_g_ampa = s_state.g_ampa * jnp.exp(-dt / TAU_AMPA) + g_input_exc
            new_g_gaba_a = s_state.g_gaba_a * jnp.exp(-dt / TAU_GABA_A) + g_input_inh

            # Phase 2: NMDA (dual-exponential, excitatory, with normalisation)
            new_nmda_rise = (
                s_state.g_nmda_rise * jnp.exp(-dt / TAU_NMDA_RISE) + g_input_exc * _NMDA_NORM
            )
            new_nmda_decay = (
                s_state.g_nmda_decay * jnp.exp(-dt / TAU_NMDA_DECAY) + g_input_exc * _NMDA_NORM
            )

            # Phase 2: GABA_B (dual-exponential, inhibitory, with normalisation)
            new_gaba_b_rise = (
                s_state.g_gaba_b_rise * jnp.exp(-dt / TAU_GABA_B_RISE) + g_input_inh * _GABA_B_NORM
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

            # 7. Optional plasticity update
            if plasticity_fn is not None:
                s_stdp, w_exc = plasticity_fn(s_stdp, neuron_state.spikes, w_exc)

            new_carry = (neuron_state, s_state, s_stdp, w_exc, delay_buf, stp_st)
            return new_carry, neuron_state.spikes

        init_carry = (init_state, syn_state, stdp_state, W_exc, delay_buf_init, stp_state_init)
        final_carry, spike_history = jax.lax.scan(_step_fn, init_carry, I_external)

        final_neuron_state, final_syn_state, final_stdp_state, final_W_exc, _, _ = final_carry
    else:

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
            g_input_exc = read_delayed_spikes(delay_buf, W_exc_delays, w_exc)
            g_input_inh = read_delayed_spikes(delay_buf, W_inh_delays, W_inh)

            # Phase 1: AMPA & GABA_A (single-exponential decay + delayed input)
            new_g_ampa = s_state.g_ampa * jnp.exp(-dt / TAU_AMPA) + g_input_exc
            new_g_gaba_a = s_state.g_gaba_a * jnp.exp(-dt / TAU_GABA_A) + g_input_inh

            # Phase 2: NMDA (dual-exponential, excitatory, with normalisation)
            new_nmda_rise = (
                s_state.g_nmda_rise * jnp.exp(-dt / TAU_NMDA_RISE) + g_input_exc * _NMDA_NORM
            )
            new_nmda_decay = (
                s_state.g_nmda_decay * jnp.exp(-dt / TAU_NMDA_DECAY) + g_input_exc * _NMDA_NORM
            )

            # Phase 2: GABA_B (dual-exponential, inhibitory, with normalisation)
            new_gaba_b_rise = (
                s_state.g_gaba_b_rise * jnp.exp(-dt / TAU_GABA_B_RISE) + g_input_inh * _GABA_B_NORM
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
simulate_jit = jax.jit(
    simulate,
    static_argnames=(
        "dt",
        "plasticity_fn",
        "use_fast_sparse",
        "use_event_driven",
        "max_active",
        "surrogate",
        "surrogate_fn",
        "surrogate_beta",
    ),
)
