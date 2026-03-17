"""Neuronal avalanche distribution plots.

Power-law distributions of avalanche sizes and durations are the
hallmark signature of criticality in neural systems (Beggs & Plenz
2003).  These plots show the empirical distributions on log-log axes
with theoretical reference slopes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bl1.visualization._style import (
    BLUE_LIGHT,
    RED_LIGHT,
    DPI,
    bl1_style,
)


def plot_avalanche_distributions(
    sizes: np.ndarray,
    durations: np.ndarray,
    figsize: tuple = (12, 5),
    title: str = "Neuronal Avalanche Distributions",
) -> Figure:
    """Log-log plots of avalanche size and duration distributions.

    Two panels:

    - **Left**: Size distribution with power-law reference
      (alpha = -1.5, Beggs & Plenz 2003).
    - **Right**: Duration distribution with power-law reference
      (beta = -2.0).

    Args:
        sizes: 1-D array of avalanche sizes (total spike counts).
        durations: 1-D array of avalanche durations (number of active
            bins).
        figsize: Figure dimensions.
        title: Overall figure title.

    Returns:
        matplotlib Figure.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)

    with bl1_style():
        fig, (ax_size, ax_dur) = plt.subplots(1, 2, figsize=figsize, dpi=DPI)

        # -- Size distribution -----------------------------------------------
        if len(sizes) > 0 and sizes.max() > 0:
            s_min = max(sizes.min(), 1)
            s_max = sizes.max()
            if s_max > s_min:
                bins = np.logspace(np.log10(s_min), np.log10(s_max), 30)
            else:
                bins = np.linspace(s_min * 0.9, s_max * 1.1, 10)

            ax_size.hist(
                sizes, bins=bins, density=True, alpha=0.7, color=BLUE_LIGHT,
                edgecolor="white",
            )

            # Reference power law: P(s) ~ s^{-1.5}
            if s_max > s_min and len(bins) > 2:
                s_ref = np.logspace(np.log10(bins[1]), np.log10(bins[-2]), 50)
                # Normalise reference line to pass through the data range
                scale = (s_ref[0] ** 1.5)  # anchor at left end
                ax_size.plot(
                    s_ref,
                    scale * s_ref ** (-1.5),
                    "r--",
                    lw=1.5,
                    label=r"$\alpha = -3/2$",
                )
                ax_size.legend(framealpha=0.8)

            ax_size.set_xscale("log")
            ax_size.set_yscale("log")

        ax_size.set_xlabel("Avalanche Size")
        ax_size.set_ylabel("P(size)")
        ax_size.set_title("Size Distribution")

        # -- Duration distribution -------------------------------------------
        if len(durations) > 0 and durations.max() > 0:
            d_min = max(durations.min(), 1)
            d_max = durations.max()
            if d_max > d_min:
                bins_d = np.logspace(np.log10(d_min), np.log10(d_max), 30)
            else:
                bins_d = np.linspace(d_min * 0.9, d_max * 1.1, 10)

            ax_dur.hist(
                durations, bins=bins_d, density=True, alpha=0.7, color=RED_LIGHT,
                edgecolor="white",
            )

            # Reference power law: P(d) ~ d^{-2.0}
            if d_max > d_min and len(bins_d) > 2:
                d_ref = np.logspace(np.log10(bins_d[1]), np.log10(bins_d[-2]), 50)
                scale_d = (d_ref[0] ** 2.0)
                ax_dur.plot(
                    d_ref,
                    scale_d * d_ref ** (-2.0),
                    "r--",
                    lw=1.5,
                    label=r"$\beta = -2$",
                )
                ax_dur.legend(framealpha=0.8)

            ax_dur.set_xscale("log")
            ax_dur.set_yscale("log")

        ax_dur.set_xlabel("Avalanche Duration (bins)")
        ax_dur.set_ylabel("P(duration)")
        ax_dur.set_title("Duration Distribution")

        fig.suptitle(title, y=1.02, fontsize=13)
        fig.tight_layout()
    return fig
