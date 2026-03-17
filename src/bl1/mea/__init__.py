"""Virtual multi-electrode array (MEA) for recording and stimulation.

Provides electrode configurations (CL1 64-channel and MaxOne HD-MEA),
neuron-electrode spatial mapping, spike detection, firing-rate estimation,
LFP approximation, and electrical stimulation with linear distance falloff.
"""

from bl1.mea.electrode import (
    MEA,
    MEAConfig,
    build_neuron_electrode_map,
    build_neuron_electrode_map_sparse,
    compute_lfp,
    select_electrode_subset,
)
from bl1.mea.recording import compute_electrode_rates, detect_spikes
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
