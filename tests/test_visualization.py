"""Tests for the visualization module (bl1.visualization).

All tests use the Agg backend for headless rendering and verify that
plot functions return valid matplotlib Figure objects without error.
Synthetic spike rasters (random binary arrays) stand in for real data.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def spike_raster(rng):
    """Synthetic (T=2000, N=200) spike raster, ~2% firing probability."""
    return rng.random((2000, 200)) < 0.02


@pytest.fixture()
def sparse_spike_raster(rng):
    """Very sparse raster for edge-case coverage."""
    r = np.zeros((500, 50), dtype=bool)
    # A handful of spikes
    for n in range(50):
        ts = rng.choice(500, size=rng.integers(0, 5), replace=False)
        r[ts, n] = True
    return r


# ---------------------------------------------------------------------------
# Raster plots
# ---------------------------------------------------------------------------

def test_plot_raster(spike_raster):
    from bl1.visualization.raster import plot_raster

    fig = plot_raster(spike_raster, dt_ms=0.5, neuron_subset=100)
    assert isinstance(fig, Figure)
    axes = fig.get_axes()
    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Time (ms)"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_raster_ei_boundary(spike_raster):
    from bl1.visualization.raster import plot_raster

    fig = plot_raster(
        spike_raster,
        dt_ms=0.5,
        neuron_subset=100,
        ei_boundary=160,  # 80% excitatory
    )
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_raster_time_range(spike_raster):
    from bl1.visualization.raster import plot_raster

    fig = plot_raster(
        spike_raster,
        dt_ms=0.5,
        time_range=(100.0, 500.0),
    )
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_raster_with_rate(spike_raster):
    from bl1.visualization.raster import plot_raster_with_rate

    fig = plot_raster_with_rate(spike_raster, dt_ms=0.5, neuron_subset=100)
    assert isinstance(fig, Figure)
    axes = fig.get_axes()
    # Two subplots: raster + rate
    assert len(axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rate plots
# ---------------------------------------------------------------------------

def test_plot_population_rate(spike_raster):
    from bl1.visualization.rates import plot_population_rate

    fig = plot_population_rate(spike_raster, dt_ms=0.5, smoothing_ms=50.0)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_population_rate_no_smoothing(spike_raster):
    from bl1.visualization.rates import plot_population_rate

    fig = plot_population_rate(spike_raster, dt_ms=0.5, smoothing_ms=0)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_firing_rate_histogram(spike_raster):
    from bl1.visualization.rates import plot_firing_rate_histogram

    fig = plot_firing_rate_histogram(spike_raster, dt_ms=0.5)
    assert isinstance(fig, Figure)
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Firing Rate (Hz)"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_rate_comparison(spike_raster, rng):
    from bl1.visualization.rates import plot_rate_comparison

    raster_b = rng.random((2000, 200)) < 0.04  # higher rate condition
    fig = plot_rate_comparison(
        {"Baseline": spike_raster, "Stimulated": raster_b},
        dt_ms=0.5,
    )
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# MEA plots
# ---------------------------------------------------------------------------

def test_plot_mea_heatmap(spike_raster, rng):
    from bl1.visualization.mea import plot_mea_heatmap

    N = spike_raster.shape[1]
    E = 64  # 8x8 grid

    # Create a simple neuron-electrode map (random assignment)
    elec_map = rng.random((E, N)) < 0.05  # ~5% coverage per electrode

    # 8x8 grid positions
    xs = np.linspace(800, 2200, 8)
    ys = np.linspace(800, 2200, 8)
    gx, gy = np.meshgrid(xs, ys)
    electrode_positions = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    fig = plot_mea_heatmap(
        spike_raster,
        elec_map,
        electrode_positions,
        window_ms=500.0,
        dt_ms=0.5,
    )
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_mea_activity(spike_raster, rng):
    from bl1.visualization.mea import plot_mea_activity

    N = spike_raster.shape[1]
    neuron_positions = rng.uniform(0, 3000, size=(N, 2))

    fig = plot_mea_activity(
        spike_raster,
        neuron_positions,
        dt_ms=0.5,
        time_point_ms=500.0,
    )
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_mea_activity_default_time(spike_raster, rng):
    from bl1.visualization.mea import plot_mea_activity

    N = spike_raster.shape[1]
    neuron_positions = rng.uniform(0, 3000, size=(N, 2))

    fig = plot_mea_activity(spike_raster, neuron_positions, dt_ms=0.5)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Burst plots
# ---------------------------------------------------------------------------

def test_plot_burst_overlay(spike_raster):
    from bl1.visualization.bursts import plot_burst_overlay

    # Fabricate burst tuples: (start_ms, end_ms, n_spikes, fraction_recruited)
    bursts = [
        (100.0, 200.0, 500, 0.6),
        (400.0, 480.0, 300, 0.4),
        (700.0, 850.0, 800, 0.8),
    ]

    fig = plot_burst_overlay(spike_raster, bursts, dt_ms=0.5)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_burst_overlay_no_bursts(spike_raster):
    from bl1.visualization.bursts import plot_burst_overlay

    fig = plot_burst_overlay(spike_raster, [], dt_ms=0.5)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_isi_distribution(spike_raster):
    from bl1.visualization.bursts import plot_isi_distribution

    fig = plot_isi_distribution(spike_raster, dt_ms=0.5)
    assert isinstance(fig, Figure)
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Inter-Spike Interval (ms)"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_isi_distribution_single_neuron(spike_raster):
    from bl1.visualization.bursts import plot_isi_distribution

    fig = plot_isi_distribution(spike_raster, dt_ms=0.5, neuron_idx=0)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Avalanche plots
# ---------------------------------------------------------------------------

def test_plot_avalanche_distributions(rng):
    from bl1.visualization.avalanche import plot_avalanche_distributions

    # Synthetic power-law-ish distributions
    sizes = rng.pareto(1.5, size=200) + 1
    durations = rng.pareto(2.0, size=200) + 1

    fig = plot_avalanche_distributions(sizes, durations)
    assert isinstance(fig, Figure)
    axes = fig.get_axes()
    # Two panels: size + duration
    assert len(axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_avalanche_distributions_empty():
    from bl1.visualization.avalanche import plot_avalanche_distributions

    fig = plot_avalanche_distributions(np.array([]), np.array([]))
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_sparse_raster_raster_with_rate(sparse_spike_raster):
    from bl1.visualization.raster import plot_raster_with_rate

    fig = plot_raster_with_rate(sparse_spike_raster, dt_ms=0.5)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_sparse_raster_isi(sparse_spike_raster):
    from bl1.visualization.bursts import plot_isi_distribution

    fig = plot_isi_distribution(sparse_spike_raster, dt_ms=0.5)
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------

def test_package_imports():
    """All public names should be importable from the package."""
    from bl1.visualization import (
        plot_raster,
        plot_raster_with_rate,
        plot_population_rate,
        plot_firing_rate_histogram,
        plot_mea_heatmap,
        plot_mea_activity,
        plot_burst_overlay,
        plot_isi_distribution,
        plot_avalanche_distributions,
    )
    # Just verifying the imports work; callable check
    assert callable(plot_raster)
    assert callable(plot_raster_with_rate)
    assert callable(plot_population_rate)
    assert callable(plot_firing_rate_histogram)
    assert callable(plot_mea_heatmap)
    assert callable(plot_mea_activity)
    assert callable(plot_burst_overlay)
    assert callable(plot_isi_distribution)
    assert callable(plot_avalanche_distributions)
