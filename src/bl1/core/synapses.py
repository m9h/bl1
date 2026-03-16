"""Conductance-based AMPA and GABA_A synapse models.

Each synapse type is modelled as a single-exponential conductance that
decays with a characteristic time constant and receives instantaneous
jumps on presynaptic spikes.  Synaptic current is conductance-based
(driving-force model):

    I_syn = g_ampa * (E_ampa - v) + g_gaba_a * (E_gaba_a - v)

Weight matrices may be dense ``jnp.ndarray`` or JAX BCOO sparse arrays;
the step functions use the ``@`` operator which dispatches correctly for
both representations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Biophysical constants
# ---------------------------------------------------------------------------

TAU_AMPA: float = 2.0    # AMPA decay time constant (ms)
TAU_GABA_A: float = 6.0  # GABA_A decay time constant (ms)

E_AMPA: float = 0.0      # AMPA reversal potential (mV)
E_GABA_A: float = -75.0  # GABA_A reversal potential (mV)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class SynapseState(NamedTuple):
    """Aggregate conductance state per postsynaptic neuron."""
    g_ampa: Array    # (N,) total AMPA conductance onto each neuron
    g_gaba_a: Array  # (N,) total GABA_A conductance onto each neuron


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_synapse_state(n_neurons: int) -> SynapseState:
    """Return a zeroed SynapseState for *n_neurons* postsynaptic neurons."""
    return SynapseState(
        g_ampa=jnp.zeros(n_neurons),
        g_gaba_a=jnp.zeros(n_neurons),
    )


# ---------------------------------------------------------------------------
# Per-receptor step functions
# ---------------------------------------------------------------------------

@jax.jit
def ampa_step(
    g: Array,
    spikes: Array,
    weights: Array,
    dt: float = 0.5,
) -> Array:
    """Update AMPA conductances for one timestep.

    Args:
        g: Current AMPA conductance per postsynaptic neuron, shape (N_post,).
        spikes: Presynaptic spike indicators, shape (N_pre,).  Boolean or
            0/1 float.
        weights: Excitatory weight matrix, shape (N_post, N_pre).  May be a
            dense ``jnp.ndarray`` or a ``jax.experimental.sparse.BCOO`` array.
        dt: Timestep in ms.

    Returns:
        Updated AMPA conductance array, shape (N_post,).
    """
    decay = jnp.exp(-dt / TAU_AMPA)
    # Matrix-vector product gives total weighted input per postsynaptic neuron
    g_input = weights @ spikes.astype(jnp.float32)
    return g * decay + g_input


@jax.jit
def gaba_a_step(
    g: Array,
    spikes: Array,
    weights: Array,
    dt: float = 0.5,
) -> Array:
    """Update GABA_A conductances for one timestep.

    Args:
        g: Current GABA_A conductance per postsynaptic neuron, shape (N_post,).
        spikes: Presynaptic spike indicators, shape (N_pre,).
        weights: Inhibitory weight matrix, shape (N_post, N_pre).
        dt: Timestep in ms.

    Returns:
        Updated GABA_A conductance array, shape (N_post,).
    """
    decay = jnp.exp(-dt / TAU_GABA_A)
    g_input = weights @ spikes.astype(jnp.float32)
    return g * decay + g_input


# ---------------------------------------------------------------------------
# Synaptic current computation
# ---------------------------------------------------------------------------

@jax.jit
def compute_synaptic_current(syn_state: SynapseState, v: Array) -> Array:
    """Compute total synaptic current using the conductance driving-force model.

    Args:
        syn_state: Current SynapseState (g_ampa, g_gaba_a).
        v: Membrane potential of postsynaptic neurons, shape (N,).

    Returns:
        Total synaptic current I_syn, shape (N,).  Positive current is
        depolarising (follows the convention I_ext uses).
    """
    I_ampa = syn_state.g_ampa * (E_AMPA - v)
    I_gaba_a = syn_state.g_gaba_a * (E_GABA_A - v)
    return I_ampa + I_gaba_a
