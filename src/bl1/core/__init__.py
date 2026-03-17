"""Core neuron and synapse models.

Provides two spiking neuron models (Izhikevich and AdEx), four receptor-type
conductance-based synapses (AMPA, GABA_A, NMDA, GABA_B), and a JIT-compiled
time-stepper built on ``jax.lax.scan``.

All state containers are JAX-compatible NamedTuples and all step functions
are vectorised over the neuron population.
"""

from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step, izhikevich_step_surrogate, create_population
from bl1.core.adex import AdExParams, AdExState, adex_step, adex_step_surrogate, create_adex_population
from bl1.core.surrogate import (
    superspike_threshold,
    sigmoid_threshold,
    fast_sigmoid_threshold,
    arctan_threshold,
)
from bl1.core.regularization import (
    firing_rate_penalty,
    sparsity_penalty,
    silence_penalty,
)
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
from bl1.core.sparse_ops import (
    RawSparseWeights,
    bcoo_to_raw,
    fast_sparse_input,
    fast_sparse_input_raw,
)
from bl1.core.pallas_ops import (
    CSCWeights,
    bcoo_to_csc,
    csc_event_driven_input,
    event_driven_input,
    is_pallas_available,
)
from bl1.core.delays import (
    DelayBufferState,
    init_delay_buffer,
    delay_buffer_step,
    read_delayed_spikes,
    compute_max_delay,
    delays_to_dense,
)
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
    "izhikevich_step_surrogate",
    "create_population",
    "AdExParams",
    "AdExState",
    "adex_step",
    "adex_step_surrogate",
    "create_adex_population",
    "superspike_threshold",
    "sigmoid_threshold",
    "fast_sigmoid_threshold",
    "arctan_threshold",
    "firing_rate_penalty",
    "sparsity_penalty",
    "silence_penalty",
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
    "RawSparseWeights",
    "bcoo_to_raw",
    "fast_sparse_input",
    "fast_sparse_input_raw",
    "CSCWeights",
    "bcoo_to_csc",
    "csc_event_driven_input",
    "event_driven_input",
    "is_pallas_available",
    "DelayBufferState",
    "init_delay_buffer",
    "delay_buffer_step",
    "read_delayed_spikes",
    "compute_max_delay",
    "delays_to_dense",
]
