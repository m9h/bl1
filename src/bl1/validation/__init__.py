"""Open dataset comparison framework for validating BL-1 simulations.

Catalogs publicly available cortical culture electrophysiology datasets
and provides tools to compare BL-1 simulation output against real data.

Submodules
----------
- **datasets** -- Catalog of open-access cortical culture MEA datasets
  with published statistics for validation targets.
- **comparison** -- Functions to compute culture statistics from spike
  rasters and compare them against published dataset ranges.
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

__all__ = [
    "DatasetInfo",
    "DATASETS",
    "list_datasets",
    "compare_statistics",
    "compute_culture_statistics",
    "generate_comparison_report",
]
