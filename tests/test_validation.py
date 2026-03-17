"""Tests for the open dataset validation framework (bl1.validation).

All tests use synthetic data -- no real dataset downloads are performed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bl1.validation.datasets import (
    DatasetInfo,
    DATASETS,
    list_datasets,
    compare_statistics,
)
from bl1.validation.comparison import (
    compute_culture_statistics,
    generate_comparison_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bursty_raster(
    n_neurons: int = 50,
    n_steps: int = 20000,
    dt_ms: float = 0.5,
    burst_period_ms: float = 5000.0,
    burst_duration_ms: float = 200.0,
    background_prob: float = 0.002,
    burst_prob: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic spike raster with periodic bursts.

    Generates a ``(n_steps, n_neurons)`` boolean raster with:
    - Low background firing at ``background_prob`` per timestep per neuron.
    - Periodic bursts every ``burst_period_ms`` where firing probability
      jumps to ``burst_prob`` for ``burst_duration_ms``.
    """
    rng = np.random.default_rng(seed)
    raster = rng.random((n_steps, n_neurons)) < background_prob

    burst_period_steps = int(burst_period_ms / dt_ms)
    burst_dur_steps = int(burst_duration_ms / dt_ms)

    for t_start in range(0, n_steps, burst_period_steps):
        t_end = min(t_start + burst_dur_steps, n_steps)
        burst_mask = rng.random((t_end - t_start, n_neurons)) < burst_prob
        raster[t_start:t_end] |= burst_mask

    return raster.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: Dataset catalog
# ---------------------------------------------------------------------------


class TestDatasetCatalog:
    """Tests for the dataset catalog and metadata."""

    def test_dataset_catalog_not_empty(self):
        """DATASETS should contain at least one entry."""
        assert len(DATASETS) > 0, "DATASETS catalog is empty"

    def test_all_entries_are_dataset_info(self):
        """Every value in DATASETS should be a DatasetInfo instance."""
        for key, ds in DATASETS.items():
            assert isinstance(ds, DatasetInfo), (
                f"DATASETS['{key}'] is {type(ds)}, expected DatasetInfo"
            )

    def test_required_fields_populated(self):
        """Each dataset should have name, species, url, and paper."""
        for key, ds in DATASETS.items():
            assert ds.name, f"DATASETS['{key}'].name is empty"
            assert ds.species, f"DATASETS['{key}'].species is empty"
            assert ds.url, f"DATASETS['{key}'].url is empty"
            assert ds.paper, f"DATASETS['{key}'].paper is empty"

    def test_wagenaar_present(self):
        """The Wagenaar 2006 dataset should always be in the catalog."""
        assert "wagenaar_2006" in DATASETS

    def test_wagenaar_has_burst_stats(self):
        """Wagenaar 2006 should have burst rate and firing rate ranges."""
        ds = DATASETS["wagenaar_2006"]
        assert ds.burst_rate_per_min is not None
        assert ds.mean_firing_rate_hz is not None
        assert ds.burst_rate_per_min[0] < ds.burst_rate_per_min[1]
        assert ds.mean_firing_rate_hz[0] < ds.mean_firing_rate_hz[1]

    def test_list_datasets_returns_keys(self):
        """list_datasets should return all dataset keys."""
        keys = list_datasets(verbose=False)
        assert set(keys) == set(DATASETS.keys())

    def test_list_datasets_verbose(self, capsys):
        """list_datasets(verbose=True) should print a table."""
        list_datasets(verbose=True)
        captured = capsys.readouterr()
        assert "wagenaar_2006" in captured.out
        assert "datasets catalogued" in captured.out


# ---------------------------------------------------------------------------
# Tests: compare_statistics
# ---------------------------------------------------------------------------


class TestCompareStatistics:
    """Tests for the compare_statistics function."""

    def test_compare_returns_dict(self):
        """compare_statistics should return a dict."""
        sim_stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")
        assert isinstance(result, dict)

    def test_compare_structure(self):
        """Each metric in the result should have sim_value, ref_range, in_range."""
        sim_stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")

        for key, entry in result.items():
            assert "sim_value" in entry, f"Missing 'sim_value' for {key}"
            assert "ref_range" in entry, f"Missing 'ref_range' for {key}"
            assert "in_range" in entry, f"Missing 'in_range' for {key}"

    def test_in_range_true(self):
        """A value within the published range should be flagged as in_range=True."""
        sim_stats = {"mean_firing_rate_hz": 2.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")
        # Wagenaar range is (0.1, 5.0), so 2.0 should be in range
        assert result["mean_firing_rate_hz"]["in_range"] is True

    def test_in_range_false(self):
        """A value outside the published range should be flagged as in_range=False."""
        sim_stats = {"mean_firing_rate_hz": 100.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")
        assert result["mean_firing_rate_hz"]["in_range"] is False

    def test_missing_metric_skipped(self):
        """Metrics not in sim_stats should not appear in the result."""
        sim_stats = {"mean_firing_rate_hz": 2.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")
        assert "avalanche_size_exponent" not in result

    def test_no_ref_range_gives_none(self):
        """If dataset has no range for a metric, in_range should be None."""
        # Wagenaar 2006 has no branching_ratio range
        sim_stats = {"branching_ratio": 1.0}
        result = compare_statistics(sim_stats, "wagenaar_2006")
        assert result["branching_ratio"]["in_range"] is None

    def test_unknown_dataset_raises(self):
        """Requesting an unknown dataset should raise KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            compare_statistics({"mean_firing_rate_hz": 1.0}, "nonexistent")


# ---------------------------------------------------------------------------
# Tests: compute_culture_statistics
# ---------------------------------------------------------------------------


class TestComputeCultureStatistics:
    """Tests for compute_culture_statistics on synthetic data."""

    def test_all_keys_present(self):
        """Output dict should contain all expected metric keys."""
        raster = _make_bursty_raster(n_neurons=30, n_steps=10000)
        stats = compute_culture_statistics(raster, dt_ms=0.5)

        expected_keys = {
            "mean_firing_rate_hz",
            "burst_rate_per_min",
            "ibi_mean_ms",
            "ibi_cv",
            "burst_duration_mean_ms",
            "recruitment_mean",
            "branching_ratio",
            "avalanche_size_exponent",
            "avalanche_duration_exponent",
            "population_rate_cv",
            "fraction_active",
        }
        assert expected_keys.issubset(stats.keys()), (
            f"Missing keys: {expected_keys - stats.keys()}"
        )

    def test_firing_rate_positive(self):
        """Mean firing rate should be positive for bursty raster."""
        raster = _make_bursty_raster()
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert stats["mean_firing_rate_hz"] > 0

    def test_burst_rate_positive(self):
        """Burst rate should be positive for a raster with periodic bursts."""
        raster = _make_bursty_raster(n_steps=40000, burst_period_ms=5000.0)
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert stats["burst_rate_per_min"] > 0

    def test_branching_ratio_finite(self):
        """Branching ratio should be a finite number for active raster."""
        raster = _make_bursty_raster()
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert math.isfinite(stats["branching_ratio"])

    def test_fraction_active_range(self):
        """fraction_active should be in [0, 1]."""
        raster = _make_bursty_raster()
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert 0.0 <= stats["fraction_active"] <= 1.0

    def test_fraction_active_is_one_for_high_rate(self):
        """All neurons should be active if background rate is high enough."""
        raster = _make_bursty_raster(background_prob=0.05, burst_prob=0.5)
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert stats["fraction_active"] == 1.0

    def test_silent_raster(self):
        """A raster with no spikes should produce zeros / NaNs gracefully."""
        raster = np.zeros((5000, 20), dtype=np.float32)
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        assert stats["mean_firing_rate_hz"] == 0.0
        assert stats["burst_rate_per_min"] == 0.0
        assert stats["fraction_active"] == 0.0


# ---------------------------------------------------------------------------
# Tests: generate_comparison_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for the text comparison report."""

    def test_report_is_nonempty_string(self):
        """Report should be a non-empty string."""
        raster = _make_bursty_raster()
        stats = compute_culture_statistics(raster, dt_ms=0.5)
        report = generate_comparison_report(stats, "wagenaar_2006")
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_dataset_name(self):
        """Report should mention the reference dataset."""
        stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        report = generate_comparison_report(stats, "wagenaar_2006")
        assert "Wagenaar" in report

    def test_report_contains_pass_or_fail(self):
        """Report should contain PASS or FAIL indicators."""
        stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        report = generate_comparison_report(stats, "wagenaar_2006")
        assert "PASS" in report or "FAIL" in report

    def test_report_summary_line(self):
        """Report should contain a summary line with counts."""
        stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        report = generate_comparison_report(stats, "wagenaar_2006")
        assert "passed" in report
        assert "failed" in report

    def test_report_all_datasets(self):
        """Generate a report for each cataloged dataset without error."""
        stats = {"mean_firing_rate_hz": 2.0, "burst_rate_per_min": 8.0}
        for ds_name in DATASETS:
            report = generate_comparison_report(stats, ds_name)
            assert isinstance(report, str)
            assert len(report) > 0
