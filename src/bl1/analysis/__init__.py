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
- **connectivity** -- Functional connectivity inference from spike trains
  (cross-correlation, transfer entropy, small-world/rich-club analysis).
- **information** -- Information-theoretic metrics (active information
  storage, mutual information, integration, complexity).
"""

from bl1.analysis.bursts import burst_statistics, detect_bursts
from bl1.analysis.connectivity import (
    cross_correlation_matrix,
    effective_connectivity_graph,
    rich_club_coefficient,
    small_world_coefficient,
    transfer_entropy,
)
from bl1.analysis.criticality import avalanche_size_distribution, branching_ratio
from bl1.analysis.information import (
    active_information_storage,
    complexity,
    integration,
    mutual_information_matrix,
)
from bl1.analysis.metrics import performance_comparison, rally_length
from bl1.analysis.pharmacology import (
    APV,
    BICUCULLINE,
    CARBAMAZEPINE,
    CNQX,
    TTX,
    DrugEffect,
    apply_drug,
    apply_drug_to_synapses,
    wash_out,
)
from bl1.analysis.sensitivity import (
    fit_parameters,
    mean_firing_rate,
    parameter_sensitivity,
    sweep_parameter,
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
    "cross_correlation_matrix",
    "transfer_entropy",
    "effective_connectivity_graph",
    "small_world_coefficient",
    "rich_club_coefficient",
    "active_information_storage",
    "mutual_information_matrix",
    "integration",
    "complexity",
]
