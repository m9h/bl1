"""FuncAnimation support for replaying recorded monitoring sessions.

Provides :func:`animate_session` which wraps
:class:`~matplotlib.animation.FuncAnimation` to play back the history
stored in an :class:`~bl1.monitor.activity.ActivityMonitor`, optionally
saving to MP4/GIF.

The *additional_plot_func* callback follows the RatInABox pattern for
custom overlays: ``callback(fig, axes_dict, frame_index)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from bl1.monitor.style import (
    MONITOR_DPI,
    TEXT_COLOR,
    apply_monitor_style,
)

if TYPE_CHECKING:
    from bl1.monitor.activity import ActivityMonitor

matplotlib.use("Agg")


def animate_session(
    monitor: ActivityMonitor,
    speed_up: float = 5.0,
    fps: int = 15,
    window_s: float = 10.0,
    save_path: str | None = None,
    additional_plot_func: Callable | None = None,
) -> FuncAnimation:
    """Replay a recorded session as a matplotlib animation.

    Parameters
    ----------
    monitor:
        An :class:`~bl1.monitor.activity.ActivityMonitor` that has
        already accumulated history via :meth:`record_tick`.
    speed_up:
        Playback speed multiplier relative to real time.
    fps:
        Frames per second of the output animation.
    window_s:
        Visible time-window width for each frame (seconds).
    save_path:
        If provided, save the animation to this path (e.g.
        ``"session.mp4"`` or ``"session.gif"``).
    additional_plot_func:
        Optional callback ``func(fig, axes_dict, frame_index)`` for
        custom overlays drawn on top of every frame.  Follows the
        RatInABox pattern.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.  Call ``.save(...)`` or display in a
        notebook via ``IPython.display.HTML(anim.to_jshtml())``.
    """
    n_stored = monitor.history_length
    if n_stored == 0:
        raise ValueError("Monitor has no recorded data to animate.")

    # Determine time span of the stored history.
    with monitor._lock:  # noqa: SLF001
        if monitor._count <= monitor.max_history:  # noqa: SLF001
            all_ts = monitor._timestamps[: monitor._count].copy()  # noqa: SLF001
        else:
            head = monitor._head  # noqa: SLF001
            mh = monitor.max_history
            ordered = np.arange(head, head + mh) % mh
            all_ts = monitor._timestamps[ordered].copy()  # noqa: SLF001

    t_start = float(all_ts[0])
    t_end = float(all_ts[-1])
    duration_s = t_end - t_start
    if duration_s <= 0:
        raise ValueError("Recorded session has zero duration.")

    # Number of animation frames.
    playback_duration_s = duration_s / speed_up
    n_frames = max(int(playback_duration_s * fps), 1)

    # The virtual time corresponding to each frame.
    frame_times = np.linspace(t_start + window_s, t_end, n_frames)

    # Build the figure once.
    fig = plt.figure(figsize=(14, 8), dpi=MONITOR_DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    axes_dict = {
        "raster": fig.add_subplot(gs[0, 0]),
        "mountain": fig.add_subplot(gs[0, 1]),
        "mea": fig.add_subplot(gs[1, 0]),
        "rates": fig.add_subplot(gs[1, 1]),
    }

    def _update(frame_idx: int) -> list:
        """Redraw all panels for the given frame."""
        t_now = frame_times[frame_idx]

        for a in axes_dict.values():
            a.clear()

        # Temporarily restrict the monitor's visible window by filtering
        # on timestamps <= t_now.  We achieve this by calling the plot
        # methods with the appropriate window and then adjusting xlim.
        monitor.plot_raster(fig=fig, ax=axes_dict["raster"], window_s=window_s)
        monitor.plot_mountain(fig=fig, ax=axes_dict["mountain"], window_s=window_s)
        monitor.plot_mea_heatmap(
            fig=fig, ax=axes_dict["mea"], window_s=min(window_s, 2.0), _colorbar=False
        )
        monitor.plot_firing_rates(fig=fig, ax=axes_dict["rates"], window_s=window_s)

        # Override xlim to simulate playback position.
        for key in ("raster", "mountain", "rates"):
            axes_dict[key].set_xlim(t_now - window_s, t_now)

        apply_monitor_style(fig)
        fig.suptitle(
            f"BL-1 Session Replay  t={t_now - t_start:.1f} s  ({speed_up:.0f}x)",
            fontsize=11,
            color=TEXT_COLOR,
            fontweight="bold",
        )

        if additional_plot_func is not None:
            additional_plot_func(fig, axes_dict, frame_idx)

        return list(axes_dict.values())

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=fps, dpi=MONITOR_DPI)

    return anim
