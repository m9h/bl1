"""Compute and compare culture-level statistics between simulation and data.

This module provides two main entry points:

- :func:`compute_culture_statistics` takes a spike raster (the standard
  BL-1 simulation output) and computes a comprehensive set of summary
  statistics that match the metrics reported in published cortical
  culture datasets.

- :func:`generate_comparison_report` produces a human-readable text
  report comparing a simulation's statistics against a named reference
  dataset, with pass/fail indicators for each metric.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from bl1.validation.datasets import DATASETS, compare_statistics

# ============================================================================
# Comprehensive culture statistics
# ============================================================================


def compute_culture_statistics(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    burst_threshold_std: float = 2.0,
    burst_min_duration_ms: float = 50.0,
    avalanche_bin_ms: float = 4.0,
) -> dict[str, float]:
    """Compute a comprehensive set of culture statistics from a spike raster.

    This function wraps the individual analysis routines in
    :mod:`bl1.analysis.bursts` and :mod:`bl1.analysis.criticality` to
    produce a single dict of metrics that can be directly compared
    against published dataset ranges via
    :func:`bl1.validation.datasets.compare_statistics`.

    Args:
        spike_raster: ``(T, N)`` boolean or 0/1 spike raster where
            ``T`` is the number of timesteps and ``N`` is the number of
            neurons/electrodes.
        dt_ms: Simulation timestep in ms (default 0.5).
        burst_threshold_std: Number of standard deviations for burst
            onset detection (default 2.0).
        burst_min_duration_ms: Minimum burst duration in ms
            (default 50.0).
        avalanche_bin_ms: Bin width in ms for avalanche detection
            (default 4.0).

    Returns:
        Dict with the following keys (all values are ``float``):

        - ``mean_firing_rate_hz`` -- Mean firing rate across all
          neurons, in Hz.
        - ``burst_rate_per_min`` -- Number of detected bursts per
          minute of simulated time.
        - ``ibi_mean_ms`` -- Mean inter-burst interval in ms.
        - ``ibi_cv`` -- Coefficient of variation of inter-burst
          intervals.
        - ``burst_duration_mean_ms`` -- Mean burst duration in ms.
        - ``recruitment_mean`` -- Mean fraction of neurons recruited
          per burst.
        - ``branching_ratio`` -- Branching ratio sigma (1.0 =
          critical).
        - ``avalanche_size_exponent`` -- Estimated power-law exponent
          for avalanche size distribution (via log-log linear
          regression).  Negative for power-law decays.
        - ``avalanche_duration_exponent`` -- Estimated power-law
          exponent for avalanche duration distribution.
        - ``population_rate_cv`` -- Coefficient of variation of the
          population spike count time series.
        - ``fraction_active`` -- Fraction of neurons that fire at
          least once during the recording.
    """
    from bl1.analysis.bursts import burst_statistics, detect_bursts
    from bl1.analysis.criticality import (
        avalanche_size_distribution,
    )
    from bl1.analysis.criticality import (
        branching_ratio as compute_branching_ratio,
    )

    raster = np.asarray(spike_raster, dtype=np.float32)
    T, N = raster.shape
    total_time_s = T * dt_ms / 1000.0

    stats: dict[str, float] = {}

    # --- Firing rate --------------------------------------------------------
    total_spikes = float(raster.sum())
    if N > 0 and total_time_s > 0:
        stats["mean_firing_rate_hz"] = total_spikes / (N * total_time_s)
    else:
        stats["mean_firing_rate_hz"] = 0.0

    # --- Burst detection and statistics ------------------------------------
    bursts = detect_bursts(
        raster,
        dt_ms=dt_ms,
        threshold_std=burst_threshold_std,
        min_duration_ms=burst_min_duration_ms,
    )
    bstats = burst_statistics(bursts)

    n_bursts = len(bursts)
    total_time_min = total_time_s / 60.0
    stats["burst_rate_per_min"] = n_bursts / total_time_min if total_time_min > 0 else 0.0
    stats["ibi_mean_ms"] = bstats["ibi_mean"]
    stats["ibi_cv"] = bstats["ibi_cv"]
    stats["burst_duration_mean_ms"] = bstats["duration_mean"]
    stats["recruitment_mean"] = bstats["recruitment_mean"]

    # --- Criticality metrics -----------------------------------------------
    sigma = compute_branching_ratio(raster, dt_ms=dt_ms, bin_ms=avalanche_bin_ms)
    stats["branching_ratio"] = sigma

    sizes, durations = avalanche_size_distribution(raster, dt_ms=dt_ms, bin_ms=avalanche_bin_ms)

    stats["avalanche_size_exponent"] = _estimate_power_law_exponent(sizes)
    stats["avalanche_duration_exponent"] = _estimate_power_law_exponent(
        durations.astype(np.float64)
    )

    # --- Population rate statistics ----------------------------------------
    pop_count = raster.sum(axis=1)  # (T,)
    pop_mean = float(np.mean(pop_count))
    pop_std = float(np.std(pop_count))
    stats["population_rate_cv"] = pop_std / pop_mean if pop_mean > 0 else 0.0

    # --- Fraction of active neurons ----------------------------------------
    neuron_spike_counts = raster.sum(axis=0)  # (N,)
    stats["fraction_active"] = float(np.mean(neuron_spike_counts > 0))

    # --- Functional connectivity & information metrics ---------------------
    # These are O(N^2) or worse, so only compute on a small subset of
    # neurons to keep runtime manageable for large networks.
    from bl1.analysis.connectivity import cross_correlation_matrix, transfer_entropy
    from bl1.analysis.information import active_information_storage
    from bl1.analysis.information import integration as compute_integration

    _FC_SUBSET = 50
    subset_raster = raster[:, : min(N, _FC_SUBSET)]

    try:
        cc_mat = cross_correlation_matrix(subset_raster, dt_ms=dt_ms)
        # Mean off-diagonal cross-correlation
        n_sub = cc_mat.shape[0]
        if n_sub > 1:
            mask = ~np.eye(n_sub, dtype=bool)
            stats["mean_cross_correlation"] = float(np.mean(cc_mat[mask]))
        else:
            stats["mean_cross_correlation"] = 0.0
    except Exception:
        stats["mean_cross_correlation"] = float("nan")

    try:
        te_mat = transfer_entropy(
            subset_raster,
            dt_ms=dt_ms,
            history_bins=3,
            subset=min(N, _FC_SUBSET),
        )
        n_sub = te_mat.shape[0]
        if n_sub > 1:
            mask = ~np.eye(n_sub, dtype=bool)
            stats["transfer_entropy_mean"] = float(np.mean(te_mat[mask]))
        else:
            stats["transfer_entropy_mean"] = 0.0
    except Exception:
        stats["transfer_entropy_mean"] = float("nan")

    try:
        ais = active_information_storage(
            subset_raster,
            dt_ms=dt_ms,
            history_length=3,
        )
        stats["active_information_storage_mean"] = float(np.mean(ais))
    except Exception:
        stats["active_information_storage_mean"] = float("nan")

    try:
        stats["integration"] = compute_integration(
            subset_raster,
            dt_ms=dt_ms,
            n_samples=50,
        )
    except Exception:
        stats["integration"] = float("nan")

    return stats


# ============================================================================
# Power-law exponent estimation (simple log-log regression)
# ============================================================================


def _estimate_power_law_exponent(values: NDArray) -> float:
    """Estimate power-law exponent via log-log linear regression.

    This is a quick-and-dirty estimate suitable for validation
    comparison.  For rigorous power-law testing, use the
    ``powerlaw`` package or maximum-likelihood methods (Clauset
    et al. 2009).

    Args:
        values: 1-D array of positive values (e.g. avalanche sizes).

    Returns:
        Slope of log-log regression (negative for power-law decay),
        or ``nan`` if insufficient data.
    """
    if len(values) < 5:
        return float("nan")

    # Only use positive values
    pos = values[values > 0]
    if len(pos) < 5:
        return float("nan")

    # Build empirical complementary CDF (survival function)
    sorted_vals = np.sort(pos)
    n = len(sorted_vals)

    # Use unique values and their frequencies for cleaner regression
    unique_vals, counts = np.unique(sorted_vals, return_counts=True)
    if len(unique_vals) < 3:
        return float("nan")

    # Cumulative probability P(X >= x)
    cum_counts = np.cumsum(counts[::-1])[::-1]
    survival = cum_counts / n

    log_x = np.log(unique_vals)
    log_y = np.log(survival)

    # Filter out any -inf or nan
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    log_x = log_x[valid]
    log_y = log_y[valid]

    if len(log_x) < 3:
        return float("nan")

    # Simple linear regression: log_y = slope * log_x + intercept
    # The power-law exponent alpha relates to slope as:
    # P(X >= x) ~ x^{-(alpha-1)}, so slope = -(alpha-1)
    # We return slope directly (negative for power-law decay)
    n_pts = len(log_x)
    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xy = np.sum(log_x * log_y)
    sum_xx = np.sum(log_x * log_x)

    denom = n_pts * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return float("nan")

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
    return float(slope)


# ============================================================================
# Comparison report
# ============================================================================


def generate_comparison_report(
    sim_stats: dict[str, float],
    dataset_name: str = "wagenaar_2006",
) -> str:
    """Generate a text report comparing simulation to published data.

    Args:
        sim_stats: Dict of simulation statistics, as returned by
            :func:`compute_culture_statistics`.
        dataset_name: Key into :data:`~bl1.validation.datasets.DATASETS`
            for the reference dataset.

    Returns:
        Formatted multi-line string with pass/fail for each metric.
    """
    ds = DATASETS[dataset_name]
    comparison = compare_statistics(sim_stats, dataset_name)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"BL-1 Validation Report vs. {ds.name}")
    lines.append(f"Paper: {ds.paper}")
    lines.append(f"Species: {ds.species}  |  Culture: {ds.culture_type}  |  DIV: {ds.div_range}")
    lines.append("=" * 72)
    lines.append("")

    n_pass = 0
    n_fail = 0
    n_no_ref = 0

    for metric_key, result in sorted(comparison.items()):
        sim_val = result["sim_value"]
        ref_range = result["ref_range"]
        in_range = result["in_range"]

        if in_range is True:
            status = "PASS"
            n_pass += 1
        elif in_range is False:
            status = "FAIL"
            n_fail += 1
        else:
            status = "N/A "
            n_no_ref += 1

        if ref_range is not None:
            ref_str = f"[{ref_range[0]:.3g}, {ref_range[1]:.3g}]"
        else:
            ref_str = "no reference"

        # Format sim value, handling NaN
        if math.isnan(sim_val):
            val_str = "NaN"
        else:
            val_str = f"{sim_val:.4g}"

        lines.append(f"  [{status}]  {metric_key:<30s}  sim={val_str:<12s}  ref={ref_str}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"Summary: {n_pass} passed, {n_fail} failed, {n_no_ref} no reference data")
    lines.append("-" * 72)

    return "\n".join(lines)
