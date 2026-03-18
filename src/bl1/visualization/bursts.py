"""Burst detection and ISI analysis plots.

Visualizations for network bursts (detected by
:func:`~bl1.analysis.bursts.detect_bursts`) and inter-spike interval
distributions, the standard characterisation of culture burstiness.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from bl1.visualization._style import (
    BLUE_LIGHT,
    DPI,
    GREY,
    RED_LIGHT,
    bl1_style,
)


def plot_burst_overlay(
    spike_raster: np.ndarray,
    bursts: list[tuple[float, float, int, float]],
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    figsize: tuple = (14, 6),
    title: str = "Population Rate with Detected Bursts",
) -> Figure:
    """Population rate with detected burst intervals highlighted.

    Each burst interval (from :func:`~bl1.analysis.bursts.detect_bursts`)
    is drawn as a shaded vertical span behind the rate trace.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        bursts: Output of :func:`~bl1.analysis.bursts.detect_bursts` --
            list of ``(start_ms, end_ms, n_spikes, fraction_recruited)``
            tuples.
        dt_ms: Simulation timestep in ms.
        bin_ms: Bin width in ms for the population-rate trace.
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

            ax.fill_between(bin_times, rate_hz, alpha=0.6, color=BLUE_LIGHT, label="Pop. Rate")

        # Overlay burst intervals
        for i, (start, end, _n_spk, _frac) in enumerate(bursts):
            label = "Burst" if i == 0 else None
            ax.axvspan(start, end, alpha=0.25, color=RED_LIGHT, label=label)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Population Rate (Hz)")
        ax.set_title(title)
        if bursts:
            ax.legend(framealpha=0.8)

        fig.tight_layout()
    return fig


def plot_isi_distribution(
    spike_raster: np.ndarray,
    dt_ms: float = 0.5,
    neuron_idx: int | None = None,
    n_bins: int = 60,
    figsize: tuple = (8, 5),
    title: str | None = None,
) -> Figure:
    """Inter-spike interval distribution (log-scale histogram).

    Useful for identifying bursting neurons (bimodal ISI) vs tonic
    (unimodal).  When ``neuron_idx`` is ``None`` the ISIs of all neurons
    are pooled.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms.
        neuron_idx: Index of a single neuron to analyse, or ``None``
            for the whole population.
        n_bins: Number of histogram bins on the log axis.
        figsize: Figure dimensions.
        title: Plot title (auto-generated if ``None``).

    Returns:
        matplotlib Figure.
    """
    raster = np.asarray(spike_raster, dtype=bool)

    if neuron_idx is not None:
        spike_times = np.where(raster[:, neuron_idx])[0]
        default_title = f"ISI Distribution -- Neuron {neuron_idx}"
    else:
        # Pool all neurons: collect ISIs per neuron then concatenate
        all_isis: list[np.ndarray] = []
        N = raster.shape[1]
        for n in range(N):
            times = np.where(raster[:, n])[0]
            if len(times) > 1:
                all_isis.append(np.diff(times))
        spike_times = None  # not used in pool mode
        default_title = "ISI Distribution (all neurons)"

    if title is None:
        title = default_title

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        if neuron_idx is not None:
            assert spike_times is not None
            if len(spike_times) > 1:
                isis_ms = np.diff(spike_times) * dt_ms
            else:
                isis_ms = np.array([], dtype=np.float64)
        else:
            if all_isis:
                isis_ms = np.concatenate(all_isis) * dt_ms
            else:
                isis_ms = np.array([], dtype=np.float64)

        isis_ms = np.asarray(isis_ms, dtype=np.float64)

        if len(isis_ms) > 0:
            # Log-spaced bins
            low = float(max(isis_ms.min(), dt_ms))
            high = float(isis_ms.max())
            if high > low:
                bins = np.logspace(np.log10(low), np.log10(high), n_bins)
            else:
                bins = np.linspace(low * 0.9, high * 1.1, n_bins)

            ax.hist(isis_ms, bins=bins.tolist(), color=BLUE_LIGHT, edgecolor="white", alpha=0.8)
            ax.set_xscale("log")

            median_isi = np.median(isis_ms)
            ax.axvline(
                median_isi,
                color=GREY,
                ls="--",
                lw=1.5,
                label=f"Median = {median_isi:.1f} ms",
            )
            ax.legend(framealpha=0.8)

        ax.set_xlabel("Inter-Spike Interval (ms)")
        ax.set_ylabel("Count")
        ax.set_title(title)

        fig.tight_layout()
    return fig
