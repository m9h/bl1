"""Tsodyks-Markram short-term plasticity (STP).

Models facilitation and depression at individual synapses using the
Tsodyks-Markram formulation:

    dx/dt = (1-x)/tau_rec - u*x*delta(t-t_spike)
    du/dt = (U-u)/tau_fac + U*(1-u)*delta(t-t_spike)

Where:
- x: fraction of available synaptic resources (0-1), starts at 1
- u: release probability (0-1), starts at U
- U: baseline release probability
- tau_rec: recovery time constant (ms)
- tau_fac: facilitation time constant (ms)

The effective synaptic weight at each spike is: w_eff = w_static * u * x

Default parameter regimes (per Tsodyks & Markram, 1997):
- Excitatory (depressing): U=0.5, tau_rec=800ms, tau_fac~0ms
- Inhibitory (facilitating): U=0.04, tau_rec=100ms, tau_fac=1000ms

Usage
-----
The ``stp_step`` function returns a per-neuron scale factor.  Instead of
computing ``I = W @ spikes``, the caller computes
``I = W @ (spikes * scale)`` to obtain STP-modulated synaptic currents.

Reference
---------
Tsodyks, M. V., & Markram, H. (1997).  The neural code between neocortical
pyramidal neurons depends on neurotransmitter release probability.
*Proceedings of the National Academy of Sciences*, 94(2), 719-723.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Parameter and state containers
# ---------------------------------------------------------------------------


class STPParams(NamedTuple):
    """Per-synapse STP parameters.

    For the MVP implementation these are per-neuron parameters (one value per
    pre-synaptic neuron), broadcast across all outgoing synapses.
    """

    U: Array  # (N,) baseline release probability
    tau_rec: Array  # (N,) recovery time constant (ms)
    tau_fac: Array  # (N,) facilitation time constant (ms)


class STPState(NamedTuple):
    """Per-neuron STP state variables."""

    x: Array  # (N,) available resources (fraction, 0-1)
    u: Array  # (N,) release probability (fraction, 0-1)


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def create_stp_params(n_neurons: int, is_excitatory: Array) -> STPParams:
    """Create STP parameters based on neuron type.

    Excitatory neurons use a depressing regime (high U, long recovery, no
    facilitation).  Inhibitory neurons use a facilitating regime (low U, short
    recovery, long facilitation).

    Args:
        n_neurons: Number of neurons (unused but kept for API symmetry).
        is_excitatory: (N,) boolean mask -- True for excitatory neurons.

    Returns:
        ``STPParams`` with per-neuron parameter arrays.
    """
    U = jnp.where(is_excitatory, 0.5, 0.04)
    tau_rec = jnp.where(is_excitatory, 800.0, 100.0)
    # Use 0.001 instead of 0 for excitatory tau_fac to avoid division issues
    tau_fac = jnp.where(is_excitatory, 0.001, 1000.0)
    return STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)


def init_stp_state(n_neurons: int, stp_params: STPParams) -> STPState:
    """Initialize STP state: x=1 (all resources available), u=U (baseline).

    Args:
        n_neurons: Number of neurons in the population.
        stp_params: STP parameters (used to set initial u = U).

    Returns:
        An ``STPState`` with x=1 and u=U for every neuron.
    """
    return STPState(
        x=jnp.ones(n_neurons, dtype=jnp.float32),
        u=stp_params.U.astype(jnp.float32),
    )


# ---------------------------------------------------------------------------
# Core STP update
# ---------------------------------------------------------------------------


@jax.jit
def stp_step(
    state: STPState,
    params: STPParams,
    spikes: Array,
    dt: float = 0.5,
) -> tuple[STPState, Array]:
    """Update STP state and compute effective weight scaling factor.

    Algorithm per timestep:

    1. Recover x toward 1:  ``x += dt * (1 - x) / tau_rec``
    2. Decay u toward U:    ``u += dt * (U - u) / tau_fac``
    3. On spike:
       - Facilitation: ``u_spike = u + U * (1 - u)``
       - Scale factor:  ``scale = u_spike * x``   (before depression)
       - Depression:    ``x_new  = x * (1 - u_spike)``
    4. For non-spiking neurons, scale = 0 (no transmission).

    Args:
        state: Current ``STPState`` (x, u).
        params: ``STPParams`` (U, tau_rec, tau_fac).
        spikes: (N,) boolean spike vector for pre-synaptic neurons.
        dt: Timestep in ms (default 0.5, matching the Izhikevich integrator).

    Returns:
        ``(new_state, scale)`` where scale is an (N,) multiplicative factor.
        When neuron *i* spikes, the effective weight from *i* is
        ``w * scale[i]``.  When neuron *i* does not spike, ``scale[i] = 0``.
    """
    x, u = state.x, state.u
    U, tau_rec, tau_fac = params.U, params.tau_rec, params.tau_fac
    spikes_f = spikes.astype(jnp.float32)

    # 1. Recovery of x toward 1 (exact exponential decay, unconditionally stable)
    decay_x = jnp.exp(-dt / tau_rec)
    x = 1.0 - (1.0 - x) * decay_x  # equivalent to x + (1-x)*(1-exp(-dt/tau_rec))

    # 2. Decay of u toward U (exact exponential decay, unconditionally stable)
    # For tau_fac ~ 0 (excitatory), exp(-dt/tau_fac) -> 0, so u snaps to U.
    decay_u = jnp.exp(-dt / tau_fac)
    u = U + (u - U) * decay_u

    # 3. On spike: facilitation then depression
    # u jumps up: u_spike = u + U*(1-u)
    u_spike = u + U * (1.0 - u)
    # Apply u update only for spiking neurons
    u_new = jnp.where(spikes, u_spike, u)

    # Effective transmission: u_new * x (this is the scale factor)
    scale = u_new * x * spikes_f  # zero for non-spiking neurons

    # x drops: x_new = x - u_new * x = x * (1 - u_new)
    x_new = jnp.where(spikes, x * (1.0 - u_new), x)

    new_state = STPState(x=x_new, u=u_new)
    return new_state, scale
