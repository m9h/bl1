"""Post-hoc analysis tools for cortical culture experiments.

- **criticality** -- Branching ratio and neuronal avalanche distributions
  for assessing proximity to the critical regime.
- **bursts** -- Network burst detection via threshold-crossing on population
  firing rate, with inter-burst interval and recruitment statistics.
- **metrics** -- Rally-length extraction and cross-condition performance
  comparison (Mann-Whitney U).
- **pharmacology** -- Drug effect modelling (TTX, Carbamazepine, Bicuculline,
  APV, CNQX) via conductance and weight scaling.
- **sensitivity** -- Parameter sensitivity analysis, sweeps, and fitting.
"""

from bl1.analysis.criticality import branching_ratio, avalanche_size_distribution
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.analysis.metrics import rally_length, performance_comparison
from bl1.analysis.pharmacology import (
    DrugEffect,
    TTX,
    CARBAMAZEPINE,
    BICUCULLINE,
    APV,
    CNQX,
    apply_drug,
    apply_drug_to_synapses,
    wash_out,
)
from bl1.analysis.sensitivity import (
    parameter_sensitivity,
    sweep_parameter,
    fit_parameters,
    mean_firing_rate,
    synchrony_index,
    temporal_sparseness,
)

__all__ = [
    "branching_ratio",
    "avalanche_size_distribution",
    "detect_bursts",
    "burst_statistics",
    "rally_length",
    "performance_comparison",
    "DrugEffect",
    "TTX",
    "CARBAMAZEPINE",
    "BICUCULLINE",
    "APV",
    "CNQX",
    "apply_drug",
    "apply_drug_to_synapses",
    "wash_out",
    "parameter_sensitivity",
    "sweep_parameter",
    "fit_parameters",
    "mean_firing_rate",
    "synchrony_index",
    "temporal_sparseness",
]
