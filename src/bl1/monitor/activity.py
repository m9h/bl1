"""Streaming neural activity tracker for live monitoring.

Follows the RatInABox composable-plot pattern:

*   Every ``plot_*`` method accepts optional ``fig``/``ax`` arguments and
    returns ``(fig, ax)`` so panels can be freely composed.
*   History is appended O(1) per tick via pre-allocated circular buffers.
*   A single :class:`ActivityMonitor` instance is shared between the
    simulation thread (which calls :meth:`record_tick`) and a render
    thread (which calls :meth:`render_frame`), protected by a lock.

Panels
------
1. **Spike raster** -- last N s of per-group spike events, coloured by
   channel group.
2. **Mountain plot** -- stacked filled-area of per-group firing rates
   (inspired by RatInABox's ``Neurons.plot_rate_timeseries()``).
3. **MEA heatmap** -- 8x8 electrode grid with forbidden corners masked.
4. **Firing-rate timeseries** -- per-group lines over a longer window.
5. **Spike histogram** -- bar chart of total spikes per group.
6. **Dashboard** -- 2x2 composite of the above.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bl1.monitor.style import (
    CHANNEL_COLOR_LIST,
    MONITOR_DPI,
    TEXT_COLOR,
    apply_dark_theme,
    apply_monitor_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Use the non-interactive Agg backend for headless RGB rendering.
matplotlib.use("Agg")

# Forbidden MEA electrodes (corners of the 8x8 grid that have no pad).
_FORBIDDEN_ELECTRODES: set[int] = {0, 7, 56, 63}

# doom-neuron channel groups (must match udp_bridge.CHANNEL_GROUPS)
_DEFAULT_CHANNEL_GROUPS: list[tuple[str, list[int]]] = [
    ("encoding", [8, 9, 10, 17, 18, 25, 27, 28]),
    ("move_forward", [41, 42, 49]),
    ("move_backward", [50, 51, 58]),
    ("move_left", [13, 14, 21]),
    ("move_right", [45, 46, 53]),
    ("turn_left", [29, 30, 31, 37]),
    ("turn_right", [59, 60, 61, 62]),
    ("attack", [32, 33, 34]),
]


class ActivityMonitor:
    """Streams and plots neural activity from a running BL-1 simulation.

    Designed for real-time closed-loop experiments (e.g. playing DOOM):
    the simulation loop calls :meth:`record_tick` at each tick and a
    separate render / MJPEG thread calls :meth:`render_frame`.

    Parameters
    ----------
    n_neurons:
        Total neuron count in the simulated culture.
    n_electrodes:
        Number of electrodes on the virtual MEA (default 64 for 8x8).
    max_history:
        Maximum ticks retained in the circular buffer.  At 10 Hz this
        gives 10 minutes of history by default.
    channel_groups:
        List of ``(name, [electrode_indices])`` tuples.  Defaults to the
        eight doom-neuron functional groups if *None*.
    """

    # ------------------------------------------------------------------ #
    # Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        n_neurons: int,
        n_electrodes: int = 64,
        max_history: int = 6000,
        channel_groups: list[tuple[str, list[int]]] | None = None,
    ) -> None:
        self.n_neurons = n_neurons
        self.n_electrodes = n_electrodes
        self.max_history = max_history

        if channel_groups is not None:
            self.channel_groups = channel_groups
        else:
            self.channel_groups = _DEFAULT_CHANNEL_GROUPS

        self.n_groups = len(self.channel_groups)

        # -- Pre-allocated circular buffers --------------------------------
        self._spike_counts = np.zeros((max_history, self.n_groups), dtype=np.float32)
        self._electrode_spikes = np.zeros((max_history, n_electrodes), dtype=np.float32)
        self._firing_rates = np.zeros((max_history, self.n_groups), dtype=np.float32)
        self._timestamps = np.zeros(max_history, dtype=np.float64)

        self._head: int = 0  # next write position
        self._count: int = 0  # total ticks recorded (may exceed max_history)

        # -- Thread safety -------------------------------------------------
        self._lock = threading.Lock()

        # -- Cached figure for render_frame --------------------------------
        self._cached_fig: Figure | None = None
        self._cached_axes: dict[str, Axes] | None = None

    # ------------------------------------------------------------------ #
    # Recording                                                           #
    # ------------------------------------------------------------------ #

    def record_tick(
        self,
        spike_counts: np.ndarray,
        electrode_spikes: np.ndarray | None = None,
        firing_rates: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Append one tick of data.  O(1) via circular buffer.

        Parameters
        ----------
        spike_counts:
            Per-channel-group spike counts, shape ``(n_groups,)``.
        electrode_spikes:
            Per-electrode spike counts, shape ``(n_electrodes,)``.
            Optional -- if *None* the electrode buffer is left at zero
            for this tick.
        firing_rates:
            Pre-computed per-group firing rates (Hz).  If *None* rates
            are estimated from *spike_counts* as ``counts / dt`` where
            *dt* is derived from successive timestamps.
        timestamp:
            Monotonic time in seconds.  If *None* the tick index is used.
        """
        with self._lock:
            h = self._head

            self._spike_counts[h] = np.asarray(spike_counts, dtype=np.float32)

            if electrode_spikes is not None:
                self._electrode_spikes[h] = np.asarray(electrode_spikes, dtype=np.float32)
            else:
                self._electrode_spikes[h] = 0.0

            if timestamp is not None:
                self._timestamps[h] = timestamp
            else:
                self._timestamps[h] = float(self._count)

            if firing_rates is not None:
                self._firing_rates[h] = np.asarray(firing_rates, dtype=np.float32)
            else:
                # Estimate from spike counts using inter-tick dt.
                if self._count > 0:
                    prev = (h - 1) % self.max_history
                    dt = self._timestamps[h] - self._timestamps[prev]
                    if dt > 0:
                        self._firing_rates[h] = self._spike_counts[h] / dt
                    else:
                        self._firing_rates[h] = self._spike_counts[h]
                else:
                    self._firing_rates[h] = self._spike_counts[h]

            self._head = (h + 1) % self.max_history
            self._count += 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _window_slices(self, window_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(timestamps, indices)`` for the last *window_s* seconds.

        Returned arrays are ordered oldest-to-newest.
        """
        n = min(self._count, self.max_history)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

        # Build ordered index array (oldest first).
        if self._count <= self.max_history:
            indices = np.arange(n)
        else:
            indices = np.arange(self._head, self._head + n) % self.max_history

        ts = self._timestamps[indices]

        if window_s is not None and window_s > 0 and n > 0:
            t_min = ts[-1] - window_s
            mask = ts >= t_min
            indices = indices[mask]
            ts = ts[mask]

        return ts, indices

    def _group_names(self) -> list[str]:
        """Channel group display names."""
        return [g[0] for g in self.channel_groups]

    def _group_color(self, idx: int) -> str:
        """Colour for channel group *idx*."""
        if idx < len(CHANNEL_COLOR_LIST):
            return CHANNEL_COLOR_LIST[idx]
        # Fallback for extra groups.
        fallback = [
            "#FFFFFF",
            "#AAAAAA",
            "#66BB6A",
            "#AB47BC",
            "#EF5350",
            "#26A69A",
            "#FFA726",
            "#78909C",
        ]
        return fallback[idx % len(fallback)]

    @staticmethod
    def _ensure_axes(
        fig: Figure | None,
        ax: Axes | None,
        figsize: tuple[float, float] = (8, 4),
    ) -> tuple[Figure, Axes]:
        """Create *fig*/*ax* if not provided (RatInABox pattern)."""
        if ax is not None:
            if fig is None:
                fig = ax.get_figure()
            return fig, ax  # type: ignore[return-value]
        if fig is not None:
            ax = fig.add_subplot(111)
            return fig, ax
        fig, ax = plt.subplots(figsize=figsize, dpi=MONITOR_DPI)
        return fig, ax

    # ------------------------------------------------------------------ #
    # RatInABox-style composable plot methods                             #
    # ------------------------------------------------------------------ #

    def plot_raster(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        window_s: float = 10.0,
        colorby: str = "channel",
    ) -> tuple[Figure, Axes]:
        """Spike raster coloured by channel group.

        Parameters
        ----------
        fig, ax:
            Existing figure/axes to draw into.  Created if *None*.
        window_s:
            How many seconds of history to display.
        colorby:
            ``'channel'`` colours dots by channel group (default).

        Returns
        -------
        (fig, ax)
        """
        fig, ax = self._ensure_axes(fig, ax, figsize=(10, 4))

        with self._lock:
            ts, idx = self._window_slices(window_s)
            if len(idx) == 0:
                apply_dark_theme(ax)
                ax.set_title("Raster (no data)")
                return fig, ax
            counts = self._spike_counts[idx].copy()

        names = self._group_names()
        for g in range(self.n_groups):
            t_spikes: list[float] = []
            n_spikes: list[float] = []
            for k, t in enumerate(ts):
                c = int(counts[k, g])
                if c > 0:
                    t_spikes.extend([t] * c)
                    # Jitter neurons within the group band.
                    n_spikes.extend((g + np.random.uniform(-0.35, 0.35, size=c)).tolist())
            if t_spikes:
                ax.scatter(
                    t_spikes,
                    n_spikes,
                    s=4,
                    c=self._group_color(g),
                    alpha=0.7,
                    rasterized=True,
                    label=names[g],
                )

        ax.set_yticks(range(self.n_groups))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_ylim(-0.5, self.n_groups - 0.5)
        if len(ts) > 0:
            ax.set_xlim(ts[0], ts[-1])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("Spike Raster", fontsize=9)
        apply_dark_theme(ax)
        return fig, ax

    def plot_mountain(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        window_s: float = 10.0,
        overlap: float = 0.3,
    ) -> tuple[Figure, Axes]:
        """Mountain plot (stacked filled area) of per-channel firing rates.

        Inspired by RatInABox's ``Neurons.plot_rate_timeseries()``.
        Traces are stacked vertically with a configurable overlap so each
        channel group's rate envelope is visible.

        Parameters
        ----------
        fig, ax:
            Existing figure/axes to draw into.  Created if *None*.
        window_s:
            Seconds of history to display.
        overlap:
            Vertical overlap fraction between adjacent traces (0 = no
            overlap, 1 = fully overlapping).

        Returns
        -------
        (fig, ax)
        """
        fig, ax = self._ensure_axes(fig, ax, figsize=(10, 5))

        with self._lock:
            ts, idx = self._window_slices(window_s)
            if len(idx) == 0:
                apply_dark_theme(ax)
                ax.set_title("Mountain (no data)")
                return fig, ax
            rates = self._firing_rates[idx].copy()

        names = self._group_names()
        max_rate = max(float(rates.max()), 1.0)
        scale = 1.0 / max_rate
        spacing = 1.0 - overlap

        for g in range(self.n_groups):
            offset = g * spacing
            trace = rates[:, g] * scale
            ax.fill_between(
                ts,
                offset,
                offset + trace,
                alpha=0.7,
                color=self._group_color(g),
                linewidth=0.5,
                edgecolor=self._group_color(g),
                zorder=self.n_groups - g,
                label=names[g],
            )

        ax.set_yticks([g * spacing for g in range(self.n_groups)])
        ax.set_yticklabels(names, fontsize=7)
        if len(ts) > 0:
            ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(-0.1, (self.n_groups - 1) * spacing + 1.1)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title("Mountain Plot", fontsize=9)
        apply_dark_theme(ax)
        return fig, ax

    def plot_mea_heatmap(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        window_s: float = 1.0,
        _colorbar: bool = True,
    ) -> tuple[Figure, Axes]:
        """8x8 MEA electrode activity heatmap.

        Forbidden corner electrodes (0, 7, 56, 63) are masked with NaN
        so they appear as the background colour.

        Parameters
        ----------
        fig, ax:
            Existing figure/axes to draw into.  Created if *None*.
        window_s:
            Seconds of history to average over for the heatmap.
        _colorbar:
            Internal flag.  Set to *False* in streaming / dashboard
            contexts to prevent colourbar axes from accumulating across
            redraws.

        Returns
        -------
        (fig, ax)
        """
        fig, ax = self._ensure_axes(fig, ax, figsize=(5, 5))

        with self._lock:
            ts, idx = self._window_slices(window_s)
            if len(idx) == 0:
                grid = np.full((8, 8), np.nan)
            else:
                elec = self._electrode_spikes[idx].copy()
                mean_activity = elec.mean(axis=0)
                grid = np.full((8, 8), np.nan)
                n_elec = min(len(mean_activity), 64)
                for e in range(n_elec):
                    if e in _FORBIDDEN_ELECTRODES:
                        continue
                    row, col = divmod(e, 8)
                    grid[row, col] = mean_activity[e]

        im = ax.imshow(
            grid,
            cmap="hot",
            interpolation="nearest",
            origin="lower",
            aspect="equal",
            vmin=0,
        )
        if _colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
            cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            cbar.set_label("Spikes", fontsize=8, color=TEXT_COLOR)

        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.set_title("MEA Heatmap", fontsize=9)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        apply_dark_theme(ax)
        return fig, ax

    def plot_firing_rates(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        window_s: float = 30.0,
    ) -> tuple[Figure, Axes]:
        """Line plot of per-channel-group firing rates over time.

        Parameters
        ----------
        fig, ax:
            Existing figure/axes to draw into.  Created if *None*.
        window_s:
            Seconds of history to display.

        Returns
        -------
        (fig, ax)
        """
        fig, ax = self._ensure_axes(fig, ax, figsize=(10, 4))

        with self._lock:
            ts, idx = self._window_slices(window_s)
            if len(idx) == 0:
                apply_dark_theme(ax)
                ax.set_title("Firing Rates (no data)")
                return fig, ax
            rates = self._firing_rates[idx].copy()

        names = self._group_names()
        for g in range(self.n_groups):
            ax.plot(
                ts,
                rates[:, g],
                color=self._group_color(g),
                linewidth=1.2,
                alpha=0.85,
                label=names[g],
            )

        ax.legend(fontsize=6, loc="upper left", ncol=4, framealpha=0.5)
        if len(ts) > 0:
            ax.set_xlim(ts[0], ts[-1])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Rate (Hz)", fontsize=8)
        ax.set_title("Firing Rates", fontsize=9)
        apply_dark_theme(ax)
        return fig, ax

    def plot_spike_histogram(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        window_s: float = 60.0,
    ) -> tuple[Figure, Axes]:
        """Histogram of spike count distribution across channel groups.

        Parameters
        ----------
        fig, ax:
            Existing figure/axes to draw into.  Created if *None*.
        window_s:
            Seconds of history to include.

        Returns
        -------
        (fig, ax)
        """
        fig, ax = self._ensure_axes(fig, ax, figsize=(8, 4))

        with self._lock:
            ts, idx = self._window_slices(window_s)
            if len(idx) == 0:
                apply_dark_theme(ax)
                ax.set_title("Spike Histogram (no data)")
                return fig, ax
            counts = self._spike_counts[idx].copy()

        totals = counts.sum(axis=0)
        names = self._group_names()
        colors = [self._group_color(g) for g in range(self.n_groups)]

        bars = ax.bar(names, totals, color=colors, edgecolor="none", alpha=0.85)
        ax.set_ylabel("Total Spikes", fontsize=8)
        ax.set_title(f"Spike Histogram ({window_s:.0f} s)", fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        # Add count labels on top of bars.
        for bar, val in zip(bars, totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=TEXT_COLOR,
            )
        apply_dark_theme(ax)
        return fig, ax

    # ------------------------------------------------------------------ #
    # Dashboard                                                           #
    # ------------------------------------------------------------------ #

    def plot_dashboard(
        self,
        fig: Figure | None = None,
        window_s: float = 10.0,
    ) -> tuple[Figure, dict[str, Axes]]:
        """Multi-panel dashboard: raster + mountain + MEA heatmap + rates.

        Creates a 2x2 subplot grid.  If *fig* is provided it must have
        no existing axes (they will be created via ``add_subplot``).

        Parameters
        ----------
        fig:
            Existing figure to draw into.  Created if *None*.
        window_s:
            Seconds of history for each sub-plot.

        Returns
        -------
        (fig, axes_dict)
            *axes_dict* maps panel name to the axes object:
            ``'raster'``, ``'mountain'``, ``'mea'``, ``'rates'``.
        """
        if fig is None:
            fig = plt.figure(figsize=(14, 8), dpi=MONITOR_DPI)

        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
        ax_raster = fig.add_subplot(gs[0, 0])
        ax_mountain = fig.add_subplot(gs[0, 1])
        ax_mea = fig.add_subplot(gs[1, 0])
        ax_rates = fig.add_subplot(gs[1, 1])

        self.plot_raster(fig=fig, ax=ax_raster, window_s=window_s)
        self.plot_mountain(fig=fig, ax=ax_mountain, window_s=window_s)
        self.plot_mea_heatmap(fig=fig, ax=ax_mea, window_s=window_s, _colorbar=False)
        self.plot_firing_rates(fig=fig, ax=ax_rates, window_s=window_s)

        apply_monitor_style(fig)
        fig.suptitle(
            "BL-1 Live Monitor",
            fontsize=11,
            color=TEXT_COLOR,
            fontweight="bold",
        )
        return fig, {
            "raster": ax_raster,
            "mountain": ax_mountain,
            "mea": ax_mea,
            "rates": ax_rates,
        }

    # ------------------------------------------------------------------ #
    # MJPEG / numpy-RGB rendering                                         #
    # ------------------------------------------------------------------ #

    def render_frame(
        self,
        width: int = 640,
        height: int = 480,
        dpi: int = MONITOR_DPI,
    ) -> np.ndarray:
        """Render the dashboard to a numpy RGB array.

        Designed for <50 ms latency: the figure and axes are cached and
        reused across calls to avoid repeated figure creation.  Only the
        data and ``canvas.draw()`` are refreshed.

        Parameters
        ----------
        width, height:
            Output image size in pixels.
        dpi:
            Resolution.

        Returns
        -------
        numpy.ndarray
            ``(height, width, 3)`` uint8 RGB array.
        """
        fig_w = width / dpi
        fig_h = height / dpi

        # Re-use the cached figure when dimensions match.
        if (
            self._cached_fig is not None
            and self._cached_axes is not None
            and abs(self._cached_fig.get_size_inches()[0] - fig_w) < 0.01
            and abs(self._cached_fig.get_size_inches()[1] - fig_h) < 0.01
        ):
            fig = self._cached_fig
            axes = self._cached_axes
            # Clear axes content but keep the structure.
            for a in axes.values():
                a.clear()
        else:
            # Create a fresh figure.
            if self._cached_fig is not None:
                plt.close(self._cached_fig)
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.35)
            axes = {
                "raster": fig.add_subplot(gs[0, 0]),
                "mountain": fig.add_subplot(gs[0, 1]),
                "mea": fig.add_subplot(gs[1, 0]),
                "rates": fig.add_subplot(gs[1, 1]),
            }
            self._cached_fig = fig
            self._cached_axes = axes

        # Draw each panel.  _colorbar=False prevents colourbar axes
        # from stacking up across cached redraws.
        self.plot_raster(fig=fig, ax=axes["raster"])
        self.plot_mountain(fig=fig, ax=axes["mountain"])
        self.plot_mea_heatmap(fig=fig, ax=axes["mea"], _colorbar=False)
        self.plot_firing_rates(fig=fig, ax=axes["rates"])

        apply_monitor_style(fig)
        fig.suptitle(
            "BL-1 Live Monitor",
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="bold",
        )

        # Render to RGB numpy array.
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        rgba = np.asarray(buf, dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()  # drop alpha

    # ------------------------------------------------------------------ #
    # Convenience properties                                              #
    # ------------------------------------------------------------------ #

    @property
    def tick_count(self) -> int:
        """Total number of ticks recorded so far."""
        return self._count

    @property
    def history_length(self) -> int:
        """Number of ticks currently stored in the buffer."""
        return min(self._count, self.max_history)
