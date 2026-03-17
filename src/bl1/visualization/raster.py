"""Spike raster plots.

The raster is the most fundamental visualization for spiking neural
network data: each dot marks a spike, with neurons on the y-axis and
time on the x-axis.  These functions mirror the standard raster views
used in cortical-culture MEA literature (Wagenaar et al. 2006).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from bl1.visualization._style import (
    BLACK,
    BLUE_E,
    BLUE_LIGHT,
    DPI,
    RED_I,
    bl1_style,
)


def plot_raster(
    spike_raster: np.ndarray,
    dt_ms: float = 0.5,
    neuron_subset: int = 500,
    time_range: tuple[float, float] | None = None,
    ei_boundary: int | None = None,
    figsize: tuple = (14, 6),
    title: str = "Spike Raster",
    colors: tuple = (BLUE_E, RED_I),
) -> Figure:
    """Plot a spike raster with optional E/I colouring.

    Args:
        spike_raster: ``(T, N)`` boolean array of spikes.
        dt_ms: Simulation timestep in ms.
        neuron_subset: Maximum neurons to display (subsample if N > this).
        time_range: ``(start_ms, end_ms)`` window to display, or ``None``
            for the full recording.
        ei_boundary: Neuron index separating excitatory (< boundary) from
            inhibitory (>= boundary).  When ``None`` all spikes are drawn
            in black.
        figsize: Figure dimensions ``(width, height)`` in inches.
        title: Plot title.
        colors: ``(excitatory_colour, inhibitory_colour)``.

    Returns:
        matplotlib :class:`~matplotlib.figure.Figure`.
    """
    raster = np.asarray(spike_raster, dtype=bool)
    T, N = raster.shape

    # -- time range ----------------------------------------------------------
    if time_range is not None:
        t_start = max(int(time_range[0] / dt_ms), 0)
        t_end = min(int(time_range[1] / dt_ms), T)
        raster = raster[t_start:t_end]
        T = raster.shape[0]
        time_offset = t_start * dt_ms
    else:
        time_offset = 0.0

    # -- neuron subsampling --------------------------------------------------
    if neuron_subset < N:
        idx = np.sort(np.random.choice(N, neuron_subset, replace=False))
        raster_sub = raster[:, idx]
        # Remap ei_boundary to subsampled indices
        if ei_boundary is not None:
            ei_boundary_sub = int(np.searchsorted(idx, ei_boundary))
        else:
            ei_boundary_sub = None
    else:
        raster_sub = raster
        idx = np.arange(N)
        ei_boundary_sub = ei_boundary

    # -- extract spike coordinates -------------------------------------------
    spike_t, spike_n = np.where(raster_sub)
    spike_times_ms = spike_t * dt_ms + time_offset

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        if ei_boundary_sub is not None and ei_boundary_sub > 0:
            exc_mask = spike_n < ei_boundary_sub
            inh_mask = ~exc_mask
            if exc_mask.any():
                ax.scatter(
                    spike_times_ms[exc_mask],
                    spike_n[exc_mask],
                    s=0.5,
                    c=colors[0],
                    alpha=0.5,
                    rasterized=True,
                    label="Excitatory",
                )
            if inh_mask.any():
                ax.scatter(
                    spike_times_ms[inh_mask],
                    spike_n[inh_mask],
                    s=0.5,
                    c=colors[1],
                    alpha=0.5,
                    rasterized=True,
                    label="Inhibitory",
                )
            ax.legend(loc="upper right", markerscale=8, framealpha=0.8)
        else:
            ax.scatter(
                spike_times_ms,
                spike_n,
                s=0.5,
                c=BLACK,
                alpha=0.5,
                rasterized=True,
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron")
        ax.set_title(title)
        ax.set_xlim(time_offset, time_offset + T * dt_ms)
        ax.set_ylim(-0.5, raster_sub.shape[1] - 0.5)

        fig.tight_layout()
    return fig


def plot_raster_with_rate(
    spike_raster: np.ndarray,
    dt_ms: float = 0.5,
    neuron_subset: int = 500,
    rate_bin_ms: float = 10.0,
    figsize: tuple = (14, 8),
    title: str = "Neural Activity",
) -> Figure:
    """Spike raster (top) with population firing-rate trace (bottom).

    This is the standard dual-panel visualization for cortical culture
    data.  The two subplots share the x-axis (time).

    Args:
        spike_raster: ``(T, N)`` boolean array of spikes.
        dt_ms: Simulation timestep in ms.
        neuron_subset: Maximum neurons to display in the raster.
        rate_bin_ms: Bin width in ms for the population-rate trace.
        figsize: Figure dimensions ``(width, height)`` in inches.
        title: Plot title.

    Returns:
        matplotlib :class:`~matplotlib.figure.Figure`.
    """
    raster = np.asarray(spike_raster, dtype=bool)
    T, N = raster.shape

    with bl1_style():
        fig, (ax_raster, ax_rate) = plt.subplots(
            2,
            1,
            figsize=figsize,
            dpi=DPI,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # -- raster panel ----------------------------------------------------
        if neuron_subset < N:
            idx = np.sort(np.random.choice(N, neuron_subset, replace=False))
            raster_sub = raster[:, idx]
        else:
            raster_sub = raster

        spike_t, spike_n = np.where(raster_sub)
        ax_raster.scatter(
            spike_t * dt_ms,
            spike_n,
            s=0.5,
            c=BLACK,
            alpha=0.5,
            rasterized=True,
        )
        ax_raster.set_ylabel("Neuron")
        ax_raster.set_title(title)

        # -- population-rate panel -------------------------------------------
        bin_steps = max(int(rate_bin_ms / dt_ms), 1)
        n_bins = T // bin_steps
        if n_bins > 0:
            trimmed = raster[: n_bins * bin_steps].reshape(n_bins, bin_steps, N)
            pop_counts = trimmed.sum(axis=(1, 2))
            rate_hz = pop_counts / (N * rate_bin_ms / 1000.0)
            bin_times = np.arange(n_bins) * rate_bin_ms
            ax_rate.fill_between(bin_times, rate_hz, alpha=0.7, color=BLUE_LIGHT)
            ax_rate.set_ylabel("Pop. Rate (Hz)")

        ax_rate.set_xlabel("Time (ms)")

        fig.tight_layout()
    return fig
