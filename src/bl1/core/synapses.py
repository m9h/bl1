"""Conductance-based synapse models: AMPA, GABA_A, NMDA, and GABA_B.

Phase 1 receptors (AMPA, GABA_A) use single-exponential conductances with
instantaneous jumps on presynaptic spikes.

Phase 2 receptors (NMDA, GABA_B) use difference-of-exponentials (dual-
exponential) kinetics.  NMDA additionally includes voltage-dependent Mg2+
block, which is critical for STDP-dependent learning.

Synaptic current is conductance-based (driving-force model):

    I_syn = g_ampa * (E_ampa - v)
          + g_nmda * B(v) * (E_nmda - v)
          + g_gaba_a * (E_gaba_a - v)
          + g_gaba_b * (E_gaba_b - v)

where B(v) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * v)) is the Mg2+ block.

Weight matrices may be dense ``jnp.ndarray`` or JAX BCOO sparse arrays;
the step functions use the ``@`` operator which dispatches correctly for
both representations.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bl1.core.pallas_ops import CSCWeights, event_driven_input
from bl1.core.sparse_ops import fast_sparse_input

# ---------------------------------------------------------------------------
# Biophysical constants -- Phase 1 (AMPA, GABA_A)
# ---------------------------------------------------------------------------

TAU_AMPA: float = 2.0    # AMPA decay time constant (ms)
TAU_GABA_A: float = 6.0  # GABA_A decay time constant (ms)

E_AMPA: float = 0.0      # AMPA reversal potential (mV)
E_GABA_A: float = -75.0  # GABA_A reversal potential (mV)

# ---------------------------------------------------------------------------
# Biophysical constants -- Phase 2 (NMDA, GABA_B)
# ---------------------------------------------------------------------------

TAU_NMDA_RISE: float = 2.0      # NMDA rise time constant (ms)
TAU_NMDA_DECAY: float = 100.0   # NMDA decay time constant (ms)
E_NMDA: float = 0.0             # NMDA reversal potential (mV)
MG_CONC: float = 1.0            # Extracellular Mg2+ concentration (mM)

TAU_GABA_B_RISE: float = 45.0   # GABA_B rise time constant (ms)
TAU_GABA_B_DECAY: float = 170.0 # GABA_B decay time constant (ms)
E_GABA_B: float = -95.0         # GABA_B reversal potential (mV)


# ---------------------------------------------------------------------------
# Dual-exponential normalization factors
# ---------------------------------------------------------------------------

def _dual_exp_norm(tau_rise: float, tau_decay: float) -> float:
    """Normalization factor so peak of (exp(-t/tau_d) - exp(-t/tau_r)) = 1.

    For a difference-of-exponentials kernel f(t) = exp(-t/td) - exp(-t/tr),
    the peak occurs at t_peak = td*tr/(td-tr) * ln(td/tr), and the
    normalization factor is 1 / f(t_peak).
    """
    t_peak = (tau_decay * tau_rise / (tau_decay - tau_rise)
              * math.log(tau_decay / tau_rise))
    peak_val = math.exp(-t_peak / tau_decay) - math.exp(-t_peak / tau_rise)
    return 1.0 / peak_val


_NMDA_NORM: float = _dual_exp_norm(TAU_NMDA_RISE, TAU_NMDA_DECAY)
_GABA_B_NORM: float = _dual_exp_norm(TAU_GABA_B_RISE, TAU_GABA_B_DECAY)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class SynapseState(NamedTuple):
    """Aggregate conductance state per postsynaptic neuron."""
    g_ampa: Array         # (N,) total AMPA conductance onto each neuron
    g_gaba_a: Array       # (N,) total GABA_A conductance onto each neuron
    g_nmda_rise: Array    # (N,) NMDA rise component (dual-exponential)
    g_nmda_decay: Array   # (N,) NMDA decay component (dual-exponential)
    g_gaba_b_rise: Array  # (N,) GABA_B rise component (dual-exponential)
    g_gaba_b_decay: Array # (N,) GABA_B decay component (dual-exponential)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_synapse_state(n_neurons: int) -> SynapseState:
    """Return a zeroed SynapseState for *n_neurons* postsynaptic neurons."""
    z = jnp.zeros(n_neurons)
    return SynapseState(
        g_ampa=z,
        g_gaba_a=z,
        g_nmda_rise=z,
        g_nmda_decay=z,
        g_gaba_b_rise=z,
        g_gaba_b_decay=z,
    )


# ---------------------------------------------------------------------------
# Per-receptor step functions -- Phase 1
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
# Per-receptor step functions -- Phase 2
# ---------------------------------------------------------------------------

def nmda_mg_block(v: Array) -> Array:
    """Mg2+ voltage-dependent block factor B(V).

    B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))

    At rest (~-65 mV) the channel is mostly blocked (~0.08); at
    depolarised potentials (~0 mV) it is mostly unblocked (~0.78).
    """
    return 1.0 / (1.0 + (MG_CONC / 3.57) * jnp.exp(-0.062 * v))


@jax.jit
def nmda_step(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    weights: Array,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Update NMDA conductance (dual-exponential) for one timestep.

    Uses the difference-of-exponentials formulation:
        g_nmda(t) = g_decay(t) - g_rise(t)

    On each presynaptic spike, both ``g_rise`` and ``g_decay`` receive an
    increment of ``weight * normalization_factor`` so that the peak
    conductance equals the synaptic weight.

    Args:
        g_rise: NMDA rise state variable, shape (N_post,).
        g_decay: NMDA decay state variable, shape (N_post,).
        spikes: Presynaptic spike indicators, shape (N_pre,).
        weights: Excitatory weight matrix, shape (N_post, N_pre).
        dt: Timestep in ms.

    Returns:
        Tuple ``(new_g_rise, new_g_decay, g_nmda)`` where
        ``g_nmda = g_decay - g_rise`` is the net NMDA conductance.
    """
    # Decay both components
    new_rise = g_rise * jnp.exp(-dt / TAU_NMDA_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_NMDA_DECAY)
    # Spike input (with normalization so peak = weight)
    spike_input = weights @ spikes.astype(jnp.float32)
    new_rise = new_rise + spike_input * _NMDA_NORM
    new_decay = new_decay + spike_input * _NMDA_NORM
    g_nmda = new_decay - new_rise
    return new_rise, new_decay, g_nmda


@jax.jit
def gaba_b_step(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    weights: Array,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """Update GABA_B conductance (dual-exponential) for one timestep.

    Same difference-of-exponentials approach as :func:`nmda_step` but with
    GABA_B time constants (much slower kinetics: 45 ms rise, 170 ms decay).

    Args:
        g_rise: GABA_B rise state variable, shape (N_post,).
        g_decay: GABA_B decay state variable, shape (N_post,).
        spikes: Presynaptic spike indicators, shape (N_pre,).
        weights: Inhibitory weight matrix, shape (N_post, N_pre).
        dt: Timestep in ms.

    Returns:
        Tuple ``(new_g_rise, new_g_decay, g_gaba_b)`` where
        ``g_gaba_b = g_decay - g_rise`` is the net GABA_B conductance.
    """
    # Decay both components
    new_rise = g_rise * jnp.exp(-dt / TAU_GABA_B_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_GABA_B_DECAY)
    # Spike input (with normalization so peak = weight)
    spike_input = weights @ spikes.astype(jnp.float32)
    new_rise = new_rise + spike_input * _GABA_B_NORM
    new_decay = new_decay + spike_input * _GABA_B_NORM
    g_gaba_b = new_decay - new_rise
    return new_rise, new_decay, g_gaba_b


# ---------------------------------------------------------------------------
# Synaptic current computation
# ---------------------------------------------------------------------------

@jax.jit
def compute_synaptic_current(syn_state: SynapseState, v: Array) -> Array:
    """Compute total synaptic current using the conductance driving-force model.

    Includes contributions from all four receptor types: AMPA, NMDA,
    GABA_A, and GABA_B.  NMDA current is attenuated by the voltage-
    dependent Mg2+ block factor.

    Args:
        syn_state: Current SynapseState.
        v: Membrane potential of postsynaptic neurons, shape (N,).

    Returns:
        Total synaptic current I_syn, shape (N,).  Positive current is
        depolarising (follows the convention I_ext uses).
    """
    # Phase 1 receptors
    I_ampa = syn_state.g_ampa * (E_AMPA - v)
    I_gaba_a = syn_state.g_gaba_a * (E_GABA_A - v)

    # Phase 2 receptors
    g_nmda = syn_state.g_nmda_decay - syn_state.g_nmda_rise
    I_nmda = g_nmda * nmda_mg_block(v) * (E_NMDA - v)

    g_gaba_b = syn_state.g_gaba_b_decay - syn_state.g_gaba_b_rise
    I_gaba_b = g_gaba_b * (E_GABA_B - v)

    return I_ampa + I_nmda + I_gaba_a + I_gaba_b


# ---------------------------------------------------------------------------
# Fast sparse path — uses segment_sum instead of BCOO matmul
# ---------------------------------------------------------------------------

def ampa_step_fast(
    g: Array,
    spikes: Array,
    W_data: Array,
    W_rows: Array,
    W_cols: Array,
    n_post: int,
    dt: float = 0.5,
) -> Array:
    """AMPA step using :func:`fast_sparse_input` instead of BCOO matmul."""
    decay = jnp.exp(-dt / TAU_AMPA)
    g_input = fast_sparse_input(W_data, W_rows, W_cols, spikes.astype(jnp.float32), n_post)
    return g * decay + g_input


def gaba_a_step_fast(
    g: Array,
    spikes: Array,
    W_data: Array,
    W_rows: Array,
    W_cols: Array,
    n_post: int,
    dt: float = 0.5,
) -> Array:
    """GABA_A step using :func:`fast_sparse_input` instead of BCOO matmul."""
    decay = jnp.exp(-dt / TAU_GABA_A)
    g_input = fast_sparse_input(W_data, W_rows, W_cols, spikes.astype(jnp.float32), n_post)
    return g * decay + g_input


def nmda_step_fast(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    W_data: Array,
    W_rows: Array,
    W_cols: Array,
    n_post: int,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """NMDA step using :func:`fast_sparse_input` instead of BCOO matmul."""
    new_rise = g_rise * jnp.exp(-dt / TAU_NMDA_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_NMDA_DECAY)
    spike_input = fast_sparse_input(W_data, W_rows, W_cols, spikes.astype(jnp.float32), n_post)
    new_rise = new_rise + spike_input * _NMDA_NORM
    new_decay = new_decay + spike_input * _NMDA_NORM
    g_nmda = new_decay - new_rise
    return new_rise, new_decay, g_nmda


def gaba_b_step_fast(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    W_data: Array,
    W_rows: Array,
    W_cols: Array,
    n_post: int,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """GABA_B step using :func:`fast_sparse_input` instead of BCOO matmul."""
    new_rise = g_rise * jnp.exp(-dt / TAU_GABA_B_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_GABA_B_DECAY)
    spike_input = fast_sparse_input(W_data, W_rows, W_cols, spikes.astype(jnp.float32), n_post)
    new_rise = new_rise + spike_input * _GABA_B_NORM
    new_decay = new_decay + spike_input * _GABA_B_NORM
    g_gaba_b = new_decay - new_rise
    return new_rise, new_decay, g_gaba_b


# ---------------------------------------------------------------------------
# Event-driven path — CSC format, processes only spiking neurons
# ---------------------------------------------------------------------------

def ampa_step_event(
    g: Array,
    spikes: Array,
    csc: CSCWeights,
    max_active: int,
    dt: float = 0.5,
) -> Array:
    """AMPA step using event-driven CSC kernel (only active synapses)."""
    decay = jnp.exp(-dt / TAU_AMPA)
    g_input = event_driven_input(csc, spikes.astype(jnp.float32), max_active)
    return g * decay + g_input


def gaba_a_step_event(
    g: Array,
    spikes: Array,
    csc: CSCWeights,
    max_active: int,
    dt: float = 0.5,
) -> Array:
    """GABA_A step using event-driven CSC kernel (only active synapses)."""
    decay = jnp.exp(-dt / TAU_GABA_A)
    g_input = event_driven_input(csc, spikes.astype(jnp.float32), max_active)
    return g * decay + g_input


def nmda_step_event(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    csc: CSCWeights,
    max_active: int,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """NMDA step using event-driven CSC kernel (only active synapses)."""
    new_rise = g_rise * jnp.exp(-dt / TAU_NMDA_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_NMDA_DECAY)
    spike_input = event_driven_input(csc, spikes.astype(jnp.float32), max_active)
    new_rise = new_rise + spike_input * _NMDA_NORM
    new_decay = new_decay + spike_input * _NMDA_NORM
    g_nmda = new_decay - new_rise
    return new_rise, new_decay, g_nmda


def gaba_b_step_event(
    g_rise: Array,
    g_decay: Array,
    spikes: Array,
    csc: CSCWeights,
    max_active: int,
    dt: float = 0.5,
) -> tuple[Array, Array, Array]:
    """GABA_B step using event-driven CSC kernel (only active synapses)."""
    new_rise = g_rise * jnp.exp(-dt / TAU_GABA_B_RISE)
    new_decay = g_decay * jnp.exp(-dt / TAU_GABA_B_DECAY)
    spike_input = event_driven_input(csc, spikes.astype(jnp.float32), max_active)
    new_rise = new_rise + spike_input * _GABA_B_NORM
    new_decay = new_decay + spike_input * _GABA_B_NORM
    g_gaba_b = new_decay - new_rise
    return new_rise, new_decay, g_gaba_b
