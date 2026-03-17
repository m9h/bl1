"""MEA spatial activity plots.

Heatmaps and scatter plots that show the spatial distribution of neural
activity across the electrode array, matching the view experimentalists
see when inspecting culture recordings on tools like AxIS Navigator or
BrainWave.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from bl1.visualization._style import DPI, bl1_style


def plot_mea_heatmap(
    spike_raster: np.ndarray,
    neuron_electrode_map: np.ndarray,
    electrode_positions: np.ndarray,
    window_ms: float = 1000.0,
    dt_ms: float = 0.5,
    grid_shape: tuple[int, int] = (8, 8),
    figsize: tuple = (8, 8),
    title: str = "MEA Firing Rate Heatmap",
) -> Figure:
    """Heatmap of firing rates on the MEA electrode grid.

    Shows the spatial distribution of activity across the electrode array.
    Uses the neuron-electrode map to aggregate neuron spikes onto
    electrodes, then displays rates on the physical grid layout.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        neuron_electrode_map: Boolean mask ``(E, N)`` from
            :func:`~bl1.mea.electrode.build_neuron_electrode_map`.
        electrode_positions: ``(E, 2)`` electrode centre positions in um.
        window_ms: Time window in ms over which to compute rates.
            Uses the last ``window_ms`` of the recording.
        dt_ms: Simulation timestep in ms.
        grid_shape: ``(rows, cols)`` of the electrode array.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    raster = np.asarray(spike_raster, dtype=np.float32)
    elec_map = np.asarray(neuron_electrode_map, dtype=np.float32)
    elec_pos = np.asarray(electrode_positions, dtype=np.float64)
    T, N = raster.shape
    E = elec_map.shape[0]

    # Use the last window_ms of the recording
    window_steps = min(max(int(window_ms / dt_ms), 1), T)
    windowed = raster[T - window_steps:]

    # Total spikes per neuron in window
    spikes_per_neuron = windowed.sum(axis=0)  # (N,)

    # Project onto electrodes
    electrode_counts = elec_map @ spikes_per_neuron  # (E,)

    # Neurons per electrode (for normalisation)
    neurons_per_electrode = elec_map.sum(axis=1)
    neurons_per_electrode = np.maximum(neurons_per_electrode, 1.0)

    # Mean rate per neuron per electrode in Hz
    window_sec = window_steps * dt_ms / 1000.0
    rates = (electrode_counts / neurons_per_electrode) / window_sec  # (E,)

    # Reshape onto the grid
    rows, cols = grid_shape
    rate_grid = np.full((rows, cols), np.nan)
    # Map electrodes to grid positions via their physical coordinates
    xs_unique = np.sort(np.unique(np.round(elec_pos[:, 0], 2)))
    ys_unique = np.sort(np.unique(np.round(elec_pos[:, 1], 2)))

    for e in range(E):
        col_idx = int(np.argmin(np.abs(xs_unique - elec_pos[e, 0])))
        row_idx = int(np.argmin(np.abs(ys_unique - elec_pos[e, 1])))
        if row_idx < rows and col_idx < cols:
            rate_grid[row_idx, col_idx] = rates[e]

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        im = ax.imshow(
            rate_grid,
            cmap="YlOrRd",
            interpolation="nearest",
            origin="lower",
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="Firing Rate (Hz)")

        ax.set_xlabel("Electrode Column")
        ax.set_ylabel("Electrode Row")
        ax.set_title(title)

        fig.tight_layout()
    return fig


def plot_mea_activity(
    spike_raster: np.ndarray,
    neuron_positions: np.ndarray,
    dt_ms: float = 0.5,
    time_point_ms: float | None = None,
    window_ms: float = 100.0,
    figsize: tuple = (8, 8),
    title: str = "Spatial Activity",
) -> Figure:
    """Scatter plot of neuron positions coloured by activity level.

    Each neuron is drawn at its physical position; colour intensity
    encodes the firing rate within a short window around the requested
    time point.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        neuron_positions: ``(N, 2)`` neuron (x, y) positions in um.
        dt_ms: Simulation timestep in ms.
        time_point_ms: Centre of the activity window.  ``None`` uses the
            last ``window_ms`` of the recording.
        window_ms: Width of the activity window in ms.
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    raster = np.asarray(spike_raster, dtype=bool)
    positions = np.asarray(neuron_positions, dtype=np.float64)
    T, N = raster.shape

    half_win = int(window_ms / dt_ms / 2)

    if time_point_ms is not None:
        centre = int(time_point_ms / dt_ms)
    else:
        centre = T - half_win

    t_start = max(centre - half_win, 0)
    t_end = min(centre + half_win, T)

    spike_counts = raster[t_start:t_end].sum(axis=0).astype(float)
    window_sec = max((t_end - t_start) * dt_ms / 1000.0, 1e-9)
    rates = spike_counts / window_sec

    with bl1_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        sc = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=rates,
            s=4,
            cmap="YlOrRd",
            alpha=0.7,
            rasterized=True,
            norm=Normalize(vmin=0, vmax=max(rates.max(), 1.0)),
        )
        fig.colorbar(sc, ax=ax, shrink=0.8, label="Firing Rate (Hz)")

        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(title)
        ax.set_aspect("equal")

        fig.tight_layout()
    return fig
