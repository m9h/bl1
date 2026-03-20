"""Open dataset comparison framework for validating BL-1 simulations.

Catalogs publicly available cortical culture electrophysiology datasets
and provides tools to compare BL-1 simulation output against real data.

Submodules
----------
- **datasets** -- Catalog of open-access cortical culture MEA datasets
  with published statistics for validation targets.
- **comparison** -- Functions to compute culture statistics from spike
  rasters and compare them against published dataset ranges.
- **loaders** -- Load external electrophysiology data (NWB, Maxwell HDF5)
  and convert to formats compatible with BL-1 analysis functions.
"""

from bl1.validation.comparison import (
    compute_culture_statistics,
    generate_comparison_report,
)
from bl1.validation.datasets import (
    DATASETS,
    DatasetInfo,
    compare_statistics,
    list_datasets,
)
from bl1.validation.loaders import (
    compute_recording_statistics,
    load_maxwell_h5,
    load_nwb_spike_trains,
    spike_trains_to_raster,
)

__all__ = [
    "DatasetInfo",
    "DATASETS",
    "list_datasets",
    "compare_statistics",
    "compute_culture_statistics",
    "generate_comparison_report",
    "load_nwb_spike_trains",
    "load_maxwell_h5",
    "spike_trains_to_raster",
    "compute_recording_statistics",
]
