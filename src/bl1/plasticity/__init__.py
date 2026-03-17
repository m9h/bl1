"""Synaptic plasticity rules operating on multiple timescales.

- **STDP** (ms) -- Eligibility-trace spike-timing-dependent plasticity for
  excitatory weight modification.
- **STP** (ms-s) -- Tsodyks-Markram short-term facilitation and depression.
- **Homeostatic scaling** (s-min) -- Multiplicative synaptic scaling to
  maintain a target firing rate.
- **Structural plasticity** (min-h) -- Activity-dependent synapse creation
  and elimination.

All rules support both dense ``jnp.ndarray`` and sparse ``BCOO`` weight
matrices except structural plasticity, which operates on dense matrices only.
"""

from bl1.plasticity.homeostatic import (
    HomeostaticParams,
    HomeostaticState,
    homeostatic_scaling,
    init_homeostatic_state,
    update_rate_estimate,
)
from bl1.plasticity.stdp import STDPParams, STDPState, init_stdp_state, stdp_update
from bl1.plasticity.stp import (
    STPParams,
    STPState,
    create_stp_params,
    init_stp_state,
    stp_step,
)
from bl1.plasticity.structural import (
    StructuralPlasticityParams,
    structural_update,
)

__all__ = [
    "STDPState",
    "STDPParams",
    "init_stdp_state",
    "stdp_update",
    "HomeostaticParams",
    "HomeostaticState",
    "init_homeostatic_state",
    "update_rate_estimate",
    "homeostatic_scaling",
    "STPParams",
    "STPState",
    "create_stp_params",
    "init_stp_state",
    "stp_step",
    "StructuralPlasticityParams",
    "structural_update",
]
