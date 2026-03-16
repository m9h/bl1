"""Tests for HD-MEA support in bl1.mea.

Covers sparse neuron-electrode mapping, electrode subset selection, and
local field potential approximation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.mea.electrode import (
    MEA,
    build_neuron_electrode_map,
    build_neuron_electrode_map_sparse,
    compute_lfp,
    select_electrode_subset,
)


# ---------------------------------------------------------------------------
# MaxOne HD creation
# ---------------------------------------------------------------------------

def test_maxone_creation():
    """MEA('maxone_hd') creates 26,400 electrodes with grid shape (120, 220)."""
    mea = MEA("maxone_hd")
    assert mea.n_electrodes == 26_400
    assert mea.config.grid_shape == (120, 220)
    assert mea.positions.shape == (26_400, 2)


# ---------------------------------------------------------------------------
# Sparse neuron-electrode map
# ---------------------------------------------------------------------------

def test_sparse_neuron_electrode_map():
    """Sparse map for 1000 neurons / 100 electrodes matches the dense version."""
    key = jax.random.PRNGKey(0)
    neuron_positions = jax.random.uniform(key, shape=(1000, 2)) * 3000.0

    mea = MEA("cl1_64ch")
    electrode_positions = mea.positions  # (64, 2)
    radius = mea.detection_radius_um

    dense_map = build_neuron_electrode_map(
        neuron_positions, electrode_positions, radius,
    )
    sparse_map = build_neuron_electrode_map_sparse(
        neuron_positions, electrode_positions, radius,
    )

    # Convert sparse to dense for comparison
    sparse_dense = sparse_map.todense()

    # Dense map is bool, sparse is float32 ones — compare via >0
    dense_bool = np.asarray(dense_map)
    sparse_bool = np.asarray(sparse_dense) > 0.5

    np.testing.assert_array_equal(dense_bool, sparse_bool)


def test_sparse_map_memory_efficient():
    """Sparse map for 1000 neurons x 1000 electrodes stores far fewer elements than 1M."""
    key = jax.random.PRNGKey(1)
    neuron_positions = jax.random.uniform(key, shape=(1000, 2)) * 3000.0

    # Create a custom 1000-electrode grid
    xs = jnp.linspace(0, 3000, 32)
    ys = jnp.linspace(0, 3000, 32)
    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    electrode_positions = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (1024, 2)

    radius = 50.0  # small radius to keep map sparse
    sparse_map = build_neuron_electrode_map_sparse(
        neuron_positions, electrode_positions, radius,
    )

    n_stored = sparse_map.nse  # number of stored elements
    n_dense = electrode_positions.shape[0] * neuron_positions.shape[0]

    # The sparse representation must store far fewer than the dense size
    assert n_stored < n_dense * 0.1, (
        f"Sparse map has {n_stored} stored elements out of {n_dense} dense entries "
        f"({n_stored / n_dense:.2%}) — expected < 10%"
    )


# ---------------------------------------------------------------------------
# Electrode subset selection
# ---------------------------------------------------------------------------

def test_select_electrode_subset_center():
    """Selecting 64 electrodes from center should all be near the array centroid."""
    mea = MEA("maxone_hd")
    subset_idx = select_electrode_subset(mea.config, region="center", n_electrodes=64)

    assert subset_idx.shape == (64,)

    # All selected electrodes should be within a reasonable distance of the centroid
    positions_np = np.asarray(mea.positions)
    centroid = positions_np.mean(axis=0)

    selected_positions = positions_np[np.asarray(subset_idx)]
    dists_to_center = np.sqrt(np.sum((selected_positions - centroid) ** 2, axis=-1))

    # The maxone grid spans roughly 220*17.5 = 3850 um in x and 120*17.5 = 2100 um in y.
    # The array half-diagonal is ~sqrt(1925^2 + 1050^2) ≈ 2193 um.
    # 64 electrodes from the center should be much closer than that.
    max_allowed = 200.0  # generous but still much less than the half-diagonal
    assert dists_to_center.max() < max_allowed, (
        f"Farthest 'center' electrode is {dists_to_center.max():.1f} um from centroid, "
        f"expected < {max_allowed} um"
    )


def test_select_electrode_subset_grid():
    """Grid selection produces roughly evenly spaced electrodes."""
    mea = MEA("maxone_hd")
    subset_idx = select_electrode_subset(mea.config, region="grid", n_electrodes=256)

    positions_np = np.asarray(mea.positions)
    selected = positions_np[np.asarray(subset_idx)]

    # Check that the selected electrodes span a substantial fraction of the
    # full array extent in both axes
    full_x_range = positions_np[:, 0].max() - positions_np[:, 0].min()
    full_y_range = positions_np[:, 1].max() - positions_np[:, 1].min()
    sel_x_range = selected[:, 0].max() - selected[:, 0].min()
    sel_y_range = selected[:, 1].max() - selected[:, 1].min()

    assert sel_x_range > 0.5 * full_x_range, (
        f"Grid selection x-range {sel_x_range:.0f} < 50% of full {full_x_range:.0f}"
    )
    assert sel_y_range > 0.5 * full_y_range, (
        f"Grid selection y-range {sel_y_range:.0f} < 50% of full {full_y_range:.0f}"
    )

    # With a regular sub-sampled grid the nearest-neighbor distances should
    # be fairly uniform.  Compute pairwise distances and check the median NN
    # distance has a reasonable CV (coefficient of variation < 1).
    from scipy.spatial import cKDTree
    tree = cKDTree(selected)
    nn_dists, _ = tree.query(selected, k=2)
    nn_dists = nn_dists[:, 1]  # exclude self-distance
    cv = nn_dists.std() / nn_dists.mean()
    assert cv < 1.0, f"Grid NN distance CV = {cv:.2f}, expected < 1.0 for roughly even spacing"


def test_select_electrode_subset_random():
    """Random selection returns the correct number of unique electrodes."""
    mea = MEA("maxone_hd")
    key = jax.random.PRNGKey(42)
    subset_idx = select_electrode_subset(
        mea.config, region="random", n_electrodes=512, key=key,
    )
    assert subset_idx.shape == (512,)
    # All indices should be unique
    unique = np.unique(np.asarray(subset_idx))
    assert len(unique) == 512


# ---------------------------------------------------------------------------
# LFP approximation
# ---------------------------------------------------------------------------

def test_compute_lfp_distance_falloff():
    """Neuron right at an electrode produces a larger LFP than a distant neuron."""
    electrode_positions = jnp.array([[1500.0, 1500.0]])  # single electrode

    # Neuron A: 2 um from electrode
    # Neuron B: 500 um from electrode
    neuron_positions = jnp.array([
        [1502.0, 1500.0],
        [2000.0, 1500.0],
    ])

    # Same positive outward current for both
    membrane_currents = jnp.array([1.0, 1.0])

    lfp = compute_lfp(neuron_positions, electrode_positions, membrane_currents)
    assert lfp.shape == (1,)

    # Compute individual contributions to verify relative magnitudes
    lfp_near = compute_lfp(
        neuron_positions[:1], electrode_positions, membrane_currents[:1],
    )
    lfp_far = compute_lfp(
        neuron_positions[1:], electrode_positions, membrane_currents[1:],
    )

    # Near neuron should produce a much larger potential
    assert abs(float(lfp_near[0])) > abs(float(lfp_far[0])), (
        f"|LFP_near| = {abs(float(lfp_near[0])):.4e}, "
        f"|LFP_far| = {abs(float(lfp_far[0])):.4e}"
    )


def test_compute_lfp_sign():
    """Negative (inward) membrane current produces negative extracellular potential."""
    electrode_positions = jnp.array([[1500.0, 1500.0]])
    neuron_positions = jnp.array([[1510.0, 1500.0]])  # 10 um away

    # Negative current -> inward (e.g., excitatory synaptic current)
    membrane_currents = jnp.array([-1.0])

    lfp = compute_lfp(neuron_positions, electrode_positions, membrane_currents)
    assert float(lfp[0]) < 0.0, (
        f"Expected negative LFP for inward current, got {float(lfp[0]):.4e}"
    )


def test_compute_lfp_superposition():
    """LFP from two neurons approximately equals sum of individual LFPs."""
    electrode_positions = jnp.array([[1500.0, 1500.0]])

    neuron_positions = jnp.array([
        [1520.0, 1500.0],
        [1500.0, 1530.0],
    ])
    membrane_currents = jnp.array([0.5, -0.3])

    # Combined LFP
    lfp_both = compute_lfp(
        neuron_positions, electrode_positions, membrane_currents,
    )

    # Individual LFPs
    lfp_a = compute_lfp(
        neuron_positions[:1], electrode_positions, membrane_currents[:1],
    )
    lfp_b = compute_lfp(
        neuron_positions[1:], electrode_positions, membrane_currents[1:],
    )

    np.testing.assert_allclose(
        float(lfp_both[0]),
        float(lfp_a[0]) + float(lfp_b[0]),
        rtol=1e-5,
    )
