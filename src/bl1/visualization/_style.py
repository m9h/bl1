"""Shared plotting style and constants for BL-1 visualizations.

Provides a consistent look across all publication-quality figures:
  - seaborn-v0_8-paper base style
  - 150 DPI for crisp rasters and histograms
  - Colour palette inspired by ColorBrewer diverging blue-red
"""

from __future__ import annotations

import contextlib
import matplotlib.pyplot as plt


# -- Colour palette (ColorBrewer RdBu diverging) --
BLUE_E = "#2166AC"       # excitatory neurons
RED_I = "#B2182B"        # inhibitory neurons
BLUE_LIGHT = "#4393C3"   # population rate fills
RED_LIGHT = "#D6604D"    # secondary accent
GREY = "#636363"         # neutral / annotations
BLACK = "#1a1a1a"        # raster dots

DPI = 150


@contextlib.contextmanager
def bl1_style():
    """Context manager that activates the BL-1 publication style."""
    # seaborn-v0_8-paper may not be available in every matplotlib build;
    # fall back gracefully.
    try:
        with plt.style.context("seaborn-v0_8-paper"):
            yield
    except OSError:
        # Style sheet not found -- use default with manual tweaks
        with plt.rc_context(
            {
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.dpi": DPI,
                "savefig.dpi": DPI,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        ):
            yield
