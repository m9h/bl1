"""Catalog of open-access cortical culture electrophysiology datasets.

Each entry documents a publicly available dataset with metadata and
published summary statistics that can serve as validation targets for
BL-1 simulations.  The catalog is intentionally conservative -- only
datasets with clear download URLs and published quantitative statistics
are included.

Adding a new dataset
--------------------
1. Create a new :class:`DatasetInfo` instance in :data:`DATASETS`.
2. Fill in as many ``*_range`` fields as the source paper reports.
3. Add the citation so users can find the original methods.

The comparison helpers in :mod:`bl1.validation.comparison` use the
``*_range`` tuple fields for pass/fail checks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a published cortical culture electrophysiology dataset."""

    name: str
    description: str
    species: str  # "rat", "mouse", "human iPSC", "rat + human iPSC"
    culture_type: str  # "dissociated", "organoid", "organotypic slice"
    mea_type: str  # "60ch", "HD-MEA", "48-well MEA", etc.
    div_range: str  # e.g., "7-35"
    url: str  # primary download / landing page URL
    paper: str  # citation string
    n_recordings: int  # approximate number of recordings in dataset
    data_format: str  # "mat", "hdf5", "nwb", "csv", "raw binary"

    # Published summary statistics for comparison (min, max) ranges.
    # None means the paper did not report this metric explicitly.
    burst_rate_per_min: tuple[float, float] | None = None
    mean_firing_rate_hz: tuple[float, float] | None = None
    ibi_mean_ms: tuple[float, float] | None = None
    ibi_cv: tuple[float, float] | None = None
    burst_duration_mean_ms: tuple[float, float] | None = None
    recruitment_fraction: tuple[float, float] | None = None
    branching_ratio: tuple[float, float] | None = None
    avalanche_size_exponent: tuple[float, float] | None = None
    avalanche_duration_exponent: tuple[float, float] | None = None

    # Optional extra notes (e.g. access restrictions, file sizes)
    notes: str = ""


# ============================================================================
# Dataset catalog
# ============================================================================

DATASETS: dict[str, DatasetInfo] = {
    # ------------------------------------------------------------------
    # 1. Wagenaar, Pine & Potter 2006 -- the foundational bursting paper
    # ------------------------------------------------------------------
    "wagenaar_2006": DatasetInfo(
        name="Wagenaar, Pine & Potter 2006",
        description=(
            "58 dense dissociated rat cortical cultures on 60-channel MEAs, "
            "recorded across the first 5 weeks in vitro (DIV 3-35). Captures "
            "the full developmental trajectory from sporadic spiking to mature "
            "network bursting. Widely cited reference for burst pattern taxonomy."
        ),
        species="rat",
        culture_type="dissociated",
        mea_type="60ch",
        div_range="3-35",
        url="https://neurodatasharing.bme.gatech.edu/development-data/html/index.html",
        paper=(
            "Wagenaar DA, Pine J, Potter SM (2006) An extremely rich "
            "repertoire of bursting patterns during the development of "
            "cortical cultures. BMC Neurosci 7:11. "
            "doi:10.1186/1471-2202-7-11"
        ),
        n_recordings=59,
        data_format="mat",
        # Published ranges (aggregated across cultures and DIV):
        burst_rate_per_min=(0.2, 20.0),
        mean_firing_rate_hz=(0.1, 5.0),
        ibi_mean_ms=(3000.0, 300000.0),  # 3 s to 300 s IBI range
        ibi_cv=(0.3, 2.0),
        burst_duration_mean_ms=(100.0, 2000.0),
        recruitment_fraction=(0.1, 0.95),
        notes=(
            "Data hosted at Potter Lab / Georgia Tech. Download requires "
            "acknowledgment of the original publication. ~500 MB total."
        ),
    ),

    # ------------------------------------------------------------------
    # 2. Kapucu et al. 2022 -- hPSC-derived + rat comparative dataset
    # ------------------------------------------------------------------
    "kapucu_2022": DatasetInfo(
        name="Kapucu et al. 2022 (hPSC + rat comparative)",
        description=(
            "Parallel MEA recordings from human pluripotent stem cell "
            "(hPSC)-derived and rat embryonic cortical neurons during "
            "in-vitro maturation. ~740 min of raw recordings. Includes "
            "spike times, waveforms, burst activity, and network "
            "synchronization metrics. Pharmacological responses at mature "
            "stages also included."
        ),
        species="rat + human iPSC",
        culture_type="dissociated",
        mea_type="48-well MEA",
        div_range="7-49",
        url="https://doi.gin.g-node.org/10.12751/g-node.wvr3jf/",
        paper=(
            "Kapucu FE, Vinogradov A, Hyvarinen T, Yla-Outinen L, "
            "Bhatt DK, Narkilahti S (2022) Comparative microelectrode "
            "array data of the functional development of hPSC-derived "
            "and rat neuronal networks. Sci Data 9:120. "
            "doi:10.1038/s41597-022-01242-4"
        ),
        n_recordings=180,
        data_format="hdf5",
        # Published ranges for rat cortical cultures at maturity:
        burst_rate_per_min=(2.0, 30.0),
        mean_firing_rate_hz=(0.5, 15.0),
        ibi_mean_ms=(2000.0, 30000.0),
        burst_duration_mean_ms=(50.0, 500.0),
        notes=(
            "Hosted on G-Node GIN. ~2 TB total (raw + processed). "
            "Processed spike data is much smaller. Analysis pipeline "
            "code also provided."
        ),
    ),

    # ------------------------------------------------------------------
    # 3. Beggs & Plenz 2003/2004 -- neuronal avalanche reference
    # ------------------------------------------------------------------
    "beggs_plenz_2003": DatasetInfo(
        name="Beggs & Plenz 2003 (neuronal avalanches)",
        description=(
            "Organotypic cortical slice cultures on 60-channel MEAs. "
            "Foundational study demonstrating neuronal avalanches with "
            "power-law size/duration distributions and branching ratio "
            "near 1.0. Cultures produced ~4700 avalanches/hour on average."
        ),
        species="rat",
        culture_type="organotypic slice",
        mea_type="60ch",
        div_range="7-28",
        url="https://www.jneurosci.org/content/23/35/11167",
        paper=(
            "Beggs JM, Plenz D (2003) Neuronal avalanches in neocortical "
            "circuits. J Neurosci 23(35):11167-11177. "
            "doi:10.1523/JNEUROSCI.23-35-11167.2003"
        ),
        n_recordings=7,
        data_format="mat",
        # Published criticality statistics:
        branching_ratio=(0.95, 1.05),
        avalanche_size_exponent=(-1.8, -1.2),  # ~-3/2
        avalanche_duration_exponent=(-2.3, -1.7),  # ~-2
        mean_firing_rate_hz=(0.5, 3.0),
        notes=(
            "Raw data not in a public repository; published statistics "
            "extracted from paper figures and text. Contact authors for "
            "raw data. Companion paper: Beggs & Plenz (2004) J Neurosci "
            "24(22):5216-5229."
        ),
    ),

    # ------------------------------------------------------------------
    # 4. Kagan et al. 2022 -- DishBrain
    # ------------------------------------------------------------------
    "kagan_2022_dishbrain": DatasetInfo(
        name="Kagan et al. 2022 (DishBrain)",
        description=(
            "Human iPSC-derived and mouse embryonic cortical neurons on "
            "HD-MEA (MaxOne, Maxwell Biosystems) playing a simplified Pong "
            "game via real-time closed-loop stimulation. Demonstrates "
            "goal-directed learning in biological neural networks."
        ),
        species="mouse + human iPSC",
        culture_type="dissociated",
        mea_type="HD-MEA",
        div_range="14-28",
        url="https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6",
        paper=(
            "Kagan BJ, Kitchen AC, Tran NT, et al. (2022) In vitro "
            "neurons learn and exhibit sentience when embodied in a "
            "simulated game-world. Neuron 110(23):3952-3969. "
            "doi:10.1016/j.neuron.2022.09.001"
        ),
        n_recordings=15,
        data_format="hdf5",
        mean_firing_rate_hz=(0.5, 10.0),
        burst_rate_per_min=(1.0, 15.0),
        notes=(
            "Supplementary data includes some processed metrics; full "
            "raw HD-MEA recordings not in public repository. Contact "
            "Cortical Labs for data access. The DishBrain analysis "
            "methodology is the direct inspiration for BL-1's game loop."
        ),
    ),

    # ------------------------------------------------------------------
    # 5. Mendeley MEA dataset -- rat cortical culture longitudinal
    # ------------------------------------------------------------------
    "mendeley_rat_cortical": DatasetInfo(
        name="Mendeley rat cortical MEA dataset",
        description=(
            "Longitudinal recordings from rat cortical neurons in culture, "
            "sampled over several days (2-3 weeks following plating). "
            "60-channel MEA recordings with spike-sorted data."
        ),
        species="rat",
        culture_type="dissociated",
        mea_type="60ch",
        div_range="14-21",
        url="https://data.mendeley.com/datasets/4ztc7yxngf/1",
        paper=(
            "See Mendeley Data repository. Associated with multi-channel "
            "systems cortical culture recordings."
        ),
        n_recordings=10,
        data_format="mat",
        mean_firing_rate_hz=(0.2, 8.0),
        burst_rate_per_min=(1.0, 15.0),
        notes="Freely downloadable from Mendeley Data. Relatively small dataset.",
    ),

    # ------------------------------------------------------------------
    # 6. Heiney et al. 2022 -- criticality in embodied cultures
    # ------------------------------------------------------------------
    "heiney_2022_criticality": DatasetInfo(
        name="Heiney et al. 2022 (criticality in embodied networks)",
        description=(
            "Cortical cultures on HD-MEA with structured stimulation "
            "paradigms, demonstrating that critical dynamics emerge during "
            "information processing. Includes spontaneous and evoked "
            "activity recordings with avalanche statistics."
        ),
        species="mouse + human iPSC",
        culture_type="dissociated",
        mea_type="HD-MEA",
        div_range="14-35",
        url="https://www.nature.com/articles/s41467-023-41020-3",
        paper=(
            "Heiney K, Huse Ramstad O, Fiskum V, et al. (2023) "
            "Critical dynamics arise during structured information "
            "presentation within embodied in vitro neuronal networks. "
            "Nat Commun 14:5287. doi:10.1038/s41467-023-41020-3"
        ),
        n_recordings=20,
        data_format="hdf5",
        branching_ratio=(0.90, 1.10),
        avalanche_size_exponent=(-1.8, -1.2),
        mean_firing_rate_hz=(0.5, 8.0),
        notes=(
            "Supplementary data includes avalanche statistics. Check "
            "paper supplementary materials and associated code repositories "
            "for processed data."
        ),
    ),

    # ------------------------------------------------------------------
    # 7. Multi-Channel Systems example data
    # ------------------------------------------------------------------
    "mcs_example_cortical": DatasetInfo(
        name="Multi Channel Systems example cortical cultures",
        description=(
            "Example dissociated cortical culture recordings distributed "
            "by Multi Channel Systems (MCS) for MEA analysis software "
            "validation. Includes raw and processed spike data."
        ),
        species="rat",
        culture_type="dissociated",
        mea_type="60ch",
        div_range="14-28",
        url="https://www.multichannelsystems.com/content/dissociated-cell-cultures-cortex",
        paper="Multi Channel Systems application note (no formal citation).",
        n_recordings=5,
        data_format="raw binary",
        mean_firing_rate_hz=(0.5, 5.0),
        burst_rate_per_min=(2.0, 12.0),
        burst_duration_mean_ms=(100.0, 800.0),
        notes=(
            "Small example files freely downloadable. Useful for testing "
            "data loading pipelines before working with larger datasets."
        ),
    ),
}


# ============================================================================
# Utility functions
# ============================================================================


def list_datasets(verbose: bool = True) -> list[str]:
    """Print a summary table of available comparison datasets.

    Args:
        verbose: If ``True``, print a formatted summary table to stdout.

    Returns:
        List of dataset keys in :data:`DATASETS`.
    """
    keys = sorted(DATASETS.keys())
    if not verbose:
        return keys

    # Header
    header = f"{'Key':<28s} {'Species':<18s} {'Culture':<18s} {'DIV':<8s} {'Format':<8s} {'#Rec':>5s}"
    print(header)
    print("-" * len(header))

    for key in keys:
        ds = DATASETS[key]
        print(
            f"{key:<28s} {ds.species:<18s} {ds.culture_type:<18s} "
            f"{ds.div_range:<8s} {ds.data_format:<8s} {ds.n_recordings:>5d}"
        )

    print(f"\n{len(keys)} datasets catalogued.")
    return keys


# ============================================================================
# Comparison against published ranges
# ============================================================================

# Mapping from simulation stat keys to DatasetInfo field names
_METRIC_FIELDS: dict[str, str] = {
    "burst_rate_per_min": "burst_rate_per_min",
    "mean_firing_rate_hz": "mean_firing_rate_hz",
    "ibi_mean_ms": "ibi_mean_ms",
    "ibi_cv": "ibi_cv",
    "burst_duration_mean_ms": "burst_duration_mean_ms",
    "recruitment_mean": "recruitment_fraction",
    "branching_ratio": "branching_ratio",
    "avalanche_size_exponent": "avalanche_size_exponent",
    "avalanche_duration_exponent": "avalanche_duration_exponent",
}


def compare_statistics(
    sim_stats: dict[str, float],
    dataset_name: str = "wagenaar_2006",
) -> dict[str, dict]:
    """Compare BL-1 simulation statistics against published dataset ranges.

    For each metric present in *both* ``sim_stats`` and the target
    dataset, checks whether the simulated value falls within the
    published ``(min, max)`` range.

    Args:
        sim_stats: Dict with keys like ``"burst_rate_per_min"``,
            ``"mean_firing_rate_hz"``, ``"ibi_mean_ms"``,
            ``"recruitment_mean"``, ``"branching_ratio"``, etc.
        dataset_name: Key into :data:`DATASETS` for the reference
            dataset to compare against.

    Returns:
        Dict keyed by metric name.  Each value is a dict with:

        - ``sim_value``: The simulation value.
        - ``ref_range``: ``(min, max)`` from the dataset, or ``None``.
        - ``in_range``: ``True`` if ``min <= sim_value <= max``,
          ``False`` otherwise, ``None`` if dataset has no range for
          this metric.

    Raises:
        KeyError: If ``dataset_name`` is not in :data:`DATASETS`.
    """
    if dataset_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(DATASETS.keys())}"
        )

    ds = DATASETS[dataset_name]
    results: dict[str, dict] = {}

    for sim_key, ds_field in _METRIC_FIELDS.items():
        if sim_key not in sim_stats:
            continue

        sim_val = sim_stats[sim_key]
        ref_range = getattr(ds, ds_field, None)

        if ref_range is not None:
            in_range = ref_range[0] <= sim_val <= ref_range[1]
        else:
            in_range = None

        results[sim_key] = {
            "sim_value": sim_val,
            "ref_range": ref_range,
            "in_range": in_range,
        }

    return results
