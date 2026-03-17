"""Publication-quality visualization tools for BL-1 simulations."""
from bl1.visualization.raster import plot_raster, plot_raster_with_rate
from bl1.visualization.rates import plot_population_rate, plot_firing_rate_histogram
from bl1.visualization.mea import plot_mea_heatmap, plot_mea_activity
from bl1.visualization.bursts import plot_burst_overlay, plot_isi_distribution
from bl1.visualization.avalanche import plot_avalanche_distributions

__all__ = [
    "plot_raster",
    "plot_raster_with_rate",
    "plot_population_rate",
    "plot_firing_rate_histogram",
    "plot_mea_heatmap",
    "plot_mea_activity",
    "plot_burst_overlay",
    "plot_isi_distribution",
    "plot_avalanche_distributions",
]
