"""Firing-rate analysis plots.

Population-level and single-neuron rate histograms for comparing
simulated cultures against experimental MEA recordings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d

from bl1.visualization._style import (
    BLUE_LIGHT,
    DPI,
    GREY,
    bl1_style,
)


def plot_population_rate(
    spike_raster: np.ndarray,
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    smoothing_ms: float = 50.0,
    figsize: tuple = (12, 4),
    title: str = "Population Firing Rate",
) -> Figure:
    """Population firing rate over time with optional Gaussian smoothing.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms.
        bin_ms: Bin width in ms for rate estimation.
        smoothing_ms: Gaussian kernel sigma in ms.  Set to ``0`` to
            disable smoothing.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    raster = np.asarray(spike_raster, dtype=bool)
    T, N = raster.shape

    bin_steps = max(int(bin_ms / dt_ms), 1)
    n_bins = T // bin_steps

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        if n_bins > 0:
            trimmed = raster[: n_bins * bin_steps].reshape(n_bins, bin_steps, N)
            pop_counts = trimmed.sum(axis=(1, 2)).astype(float)
            rate_hz = pop_counts / (N * bin_ms / 1000.0)
            bin_times = np.arange(n_bins) * bin_ms

            if smoothing_ms > 0:
                sigma_bins = smoothing_ms / bin_ms
                rate_smooth = gaussian_filter1d(rate_hz, sigma=sigma_bins)
                ax.fill_between(bin_times, rate_hz, alpha=0.3, color=BLUE_LIGHT, label="Raw")
                ax.plot(bin_times, rate_smooth, color=BLUE_LIGHT, lw=1.5, label="Smoothed")
                ax.legend(framealpha=0.8)
            else:
                ax.fill_between(bin_times, rate_hz, alpha=0.7, color=BLUE_LIGHT)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Population Rate (Hz)")
        ax.set_title(title)

        fig.tight_layout()
    return fig


def plot_firing_rate_histogram(
    spike_raster: np.ndarray,
    dt_ms: float = 0.5,
    n_bins: int = 50,
    figsize: tuple = (8, 5),
    title: str = "Firing Rate Distribution",
) -> Figure:
    """Histogram of per-neuron mean firing rates.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms.
        n_bins: Number of histogram bins.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    raster = np.asarray(spike_raster, dtype=bool)
    T, N = raster.shape
    duration_s = T * dt_ms / 1000.0

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        if duration_s > 0 and N > 0:
            spike_counts = raster.sum(axis=0)
            rates = spike_counts / duration_s

            ax.hist(rates, bins=n_bins, color=BLUE_LIGHT, edgecolor="white", alpha=0.8)

            mean_rate = rates.mean()
            ax.axvline(mean_rate, color=GREY, ls="--", lw=1.5, label=f"Mean = {mean_rate:.1f} Hz")
            ax.legend(framealpha=0.8)

        ax.set_xlabel("Firing Rate (Hz)")
        ax.set_ylabel("Number of Neurons")
        ax.set_title(title)

        fig.tight_layout()
    return fig


def plot_rate_comparison(
    rasters: dict[str, np.ndarray],
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    figsize: tuple = (12, 5),
    title: str = "Rate Comparison",
) -> Figure:
    """Overlay population-rate traces from multiple conditions.

    Args:
        rasters: Mapping from condition label to ``(T, N)`` spike raster.
        dt_ms: Simulation timestep in ms.
        bin_ms: Bin width in ms for rate estimation.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        for label, spike_raster in rasters.items():
            raster = np.asarray(spike_raster, dtype=bool)
            T, N = raster.shape

            bin_steps = max(int(bin_ms / dt_ms), 1)
            n_bins = T // bin_steps

            if n_bins > 0 and N > 0:
                trimmed = raster[: n_bins * bin_steps].reshape(n_bins, bin_steps, N)
                pop_counts = trimmed.sum(axis=(1, 2)).astype(float)
                rate_hz = pop_counts / (N * bin_ms / 1000.0)
                bin_times = np.arange(n_bins) * bin_ms

                ax.plot(bin_times, rate_hz, lw=1.2, alpha=0.8, label=label)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Population Rate (Hz)")
        ax.set_title(title)
        ax.legend(framealpha=0.8)

        fig.tight_layout()
    return fig
