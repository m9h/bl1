"""Virtual multi-electrode array."""

from bl1.mea.electrode import (
    MEA,
    MEAConfig,
    build_neuron_electrode_map,
    build_neuron_electrode_map_sparse,
    compute_lfp,
    select_electrode_subset,
)
from bl1.mea.recording import detect_spikes, compute_electrode_rates
from bl1.mea.stimulation import apply_stimulation, generate_feedback_stim

__all__ = [
    "MEA",
    "MEAConfig",
    "build_neuron_electrode_map",
    "build_neuron_electrode_map_sparse",
    "compute_lfp",
    "select_electrode_subset",
    "detect_spikes",
    "compute_electrode_rates",
    "apply_stimulation",
    "generate_feedback_stim",
]
