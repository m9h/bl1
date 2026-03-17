"""Core neuron and synapse models.

Provides two spiking neuron models (Izhikevich and AdEx), four receptor-type
conductance-based synapses (AMPA, GABA_A, NMDA, GABA_B), and a JIT-compiled
time-stepper built on ``jax.lax.scan``.

All state containers are JAX-compatible NamedTuples and all step functions
are vectorised over the neuron population.
"""

from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step, create_population
from bl1.core.adex import AdExParams, AdExState, adex_step, create_adex_population
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    gaba_a_step,
    nmda_step,
    nmda_mg_block,
    gaba_b_step,
    compute_synaptic_current,
    create_synapse_state,
)
from bl1.core.integrator import simulate, simulate_jit, SimulationResult
from bl1.core.jaxley_adapter import (
    is_jaxley_available,
    JaxleyConfig,
    JaxleyNetwork,
    JaxleyState,
)
from bl1.core.hybrid import (
    HybridPopulation,
    HybridParams,
    HybridState,
    hybrid_step,
)

__all__ = [
    "IzhikevichParams",
    "NeuronState",
    "izhikevich_step",
    "create_population",
    "AdExParams",
    "AdExState",
    "adex_step",
    "create_adex_population",
    "SynapseState",
    "ampa_step",
    "gaba_a_step",
    "nmda_step",
    "nmda_mg_block",
    "gaba_b_step",
    "compute_synaptic_current",
    "create_synapse_state",
    "simulate",
    "simulate_jit",
    "SimulationResult",
    "is_jaxley_available",
    "JaxleyConfig",
    "JaxleyNetwork",
    "JaxleyState",
    "HybridPopulation",
    "HybridParams",
    "HybridState",
    "hybrid_step",
]
