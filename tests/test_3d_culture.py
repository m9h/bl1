"""Tests for 3D cortical culture support (organoids, spheroids, layered).

Verifies that BL-1 correctly handles (N, 3) neuron positions for placement,
connectivity, and MEA electrode mapping, while remaining backward compatible
with the existing 2D code path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from bl1.network.topology import (
    build_connectivity,
    place_neurons,
    place_neurons_layered,
    place_neurons_spheroid,
)
from bl1.mea.electrode import build_neuron_electrode_map

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(123)


def _ei_mask(n: int, exc_frac: float = 0.8) -> jnp.ndarray:
    """Return a boolean excitatory mask."""
    n_exc = int(n * exc_frac)
    return jnp.arange(n) < n_exc


# ---------------------------------------------------------------------------
# Test 1: 3D uniform placement
# ---------------------------------------------------------------------------

class TestPlaceNeurons3D:
    def test_place_neurons_3d_shape(self):
        """3D placement returns (N, 3) positions."""
        positions = place_neurons(KEY, 200, substrate_3d=(3000.0, 3000.0, 1300.0))
        assert positions.shape == (200, 3)

    def test_place_neurons_3d_bounds(self):
        """All 3D positions should lie within [0, w] x [0, h] x [0, d]."""
        w, h, d = 3000.0, 3000.0, 1300.0
        positions = place_neurons(KEY, 500, substrate_3d=(w, h, d))
        assert jnp.all(positions >= 0.0)
        assert jnp.all(positions[:, 0] <= w)
        assert jnp.all(positions[:, 1] <= h)
        assert jnp.all(positions[:, 2] <= d)

    def test_place_neurons_2d_default(self):
        """Without substrate_3d, still returns (N, 2)."""
        positions = place_neurons(KEY, 200, substrate_um=(3000.0, 3000.0))
        assert positions.shape == (200, 2)


# ---------------------------------------------------------------------------
# Test 2: Spheroid placement
# ---------------------------------------------------------------------------

class TestPlaceNeuronsSpheroid:
    def test_spheroid_shape(self):
        """Spheroid placement returns (N, 3) positions."""
        positions = place_neurons_spheroid(KEY, 300, radius_um=500.0)
        assert positions.shape == (300, 3)

    def test_spheroid_within_radius(self):
        """All positions must lie within the specified radius of the centre."""
        radius = 500.0
        center = (radius, radius, radius)
        positions = place_neurons_spheroid(
            KEY, 500, radius_um=radius, center_um=center,
        )
        center_arr = jnp.array(center)
        dists = jnp.sqrt(jnp.sum((positions - center_arr) ** 2, axis=-1))
        assert jnp.all(dists < radius + 1e-3), (
            f"Max distance from centre: {float(dists.max()):.2f}, radius: {radius}"
        )

    def test_spheroid_custom_center(self):
        """Positions should be centred around the custom centre."""
        center = (1000.0, 2000.0, 500.0)
        positions = place_neurons_spheroid(
            KEY, 500, radius_um=300.0, center_um=center,
        )
        mean_pos = jnp.mean(positions, axis=0)
        # Mean should be approximately at the centre (within ~50 um for 500 neurons)
        assert jnp.allclose(mean_pos, jnp.array(center), atol=80.0), (
            f"Mean position {mean_pos} should be near centre {center}"
        )


# ---------------------------------------------------------------------------
# Test 3: Layered placement
# ---------------------------------------------------------------------------

class TestPlaceNeuronsLayered:
    def test_layered_shape(self):
        """Layered placement returns (N, 3) positions."""
        positions = place_neurons_layered(KEY, 300)
        assert positions.shape == (300, 3)

    def test_layered_z_bounds(self):
        """All z-coordinates should be within [0, total_depth]."""
        depths = (100.0, 200.0, 400.0, 300.0, 200.0, 100.0)
        total_depth = sum(depths)
        positions = place_neurons_layered(
            KEY, 500, layer_depths_um=depths,
        )
        assert jnp.all(positions[:, 2] >= 0.0)
        assert jnp.all(positions[:, 2] <= total_depth + 1e-3)

    def test_layered_density_distribution(self):
        """Layer with highest density should contain the most neurons."""
        # Default: layer_densities=(0.05, 0.15, 0.3, 0.25, 0.15, 0.1)
        # Layer 2 (index 2) has the highest density (0.3)
        # and depth 400 um, starting at z=300 (100+200)
        depths = (100.0, 200.0, 400.0, 300.0, 200.0, 100.0)
        densities = (0.05, 0.15, 0.3, 0.25, 0.15, 0.1)
        positions = place_neurons_layered(
            KEY, 1000,
            layer_depths_um=depths,
            layer_densities=densities,
        )
        z = np.asarray(positions[:, 2])

        # Compute cumulative boundaries
        boundaries = [0.0]
        for d in depths:
            boundaries.append(boundaries[-1] + d)

        counts = []
        for i in range(len(depths)):
            count = np.sum((z >= boundaries[i]) & (z < boundaries[i + 1]))
            counts.append(count)

        # Layer 2 (index 2, density 0.3) should have the most neurons
        assert counts[2] == max(counts), (
            f"Layer 2 count {counts[2]} should be the maximum; counts={counts}"
        )

    def test_layered_xy_bounds(self):
        """x, y coordinates should be within substrate bounds."""
        substrate = (2000.0, 2000.0)
        positions = place_neurons_layered(
            KEY, 300, substrate_um=substrate,
        )
        assert jnp.all(positions[:, 0] >= 0.0)
        assert jnp.all(positions[:, 0] <= substrate[0])
        assert jnp.all(positions[:, 1] >= 0.0)
        assert jnp.all(positions[:, 1] <= substrate[1])


# ---------------------------------------------------------------------------
# Test 4: Connectivity with 3D positions
# ---------------------------------------------------------------------------

class TestConnectivity3D:
    def test_connectivity_3d_returns_bcoo(self):
        """Build connectivity with 3D positions returns BCOO matrices."""
        n = 200
        positions = place_neurons(KEY, n, substrate_3d=(3000.0, 3000.0, 1300.0))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(7)
        W_exc, W_inh, delays = build_connectivity(key_conn, positions, is_exc)

        assert isinstance(W_exc, BCOO)
        assert isinstance(W_inh, BCOO)
        assert isinstance(delays, BCOO)
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_connectivity_3d_no_self_connections(self):
        """3D connectivity should have no self-connections."""
        n = 200
        positions = place_neurons(KEY, n, substrate_3d=(3000.0, 3000.0, 1300.0))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(42)
        W_exc, W_inh, _ = build_connectivity(key_conn, positions, is_exc)

        for W, name in [(W_exc, "W_exc"), (W_inh, "W_inh")]:
            if W.data.shape[0] > 0:
                rows = W.indices[:, 0]
                cols = W.indices[:, 1]
                assert jnp.all(rows != cols), f"{name} has self-connections"


# ---------------------------------------------------------------------------
# Test 5: MEA electrode mapping with 3D neurons
# ---------------------------------------------------------------------------

class TestMEAMapping3D:
    def test_electrode_map_3d_shape(self):
        """Electrode map works with 3D neuron positions."""
        n = 200
        n_elec = 16
        neuron_pos = place_neurons(KEY, n, substrate_3d=(3000.0, 3000.0, 500.0))
        # 4x4 grid of electrodes on the z=0 surface
        elec_x = jnp.linspace(500.0, 2500.0, 4)
        elec_y = jnp.linspace(500.0, 2500.0, 4)
        gx, gy = jnp.meshgrid(elec_x, elec_y, indexing="xy")
        electrode_pos = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (16, 2)

        mask = build_neuron_electrode_map(neuron_pos, electrode_pos, radius_um=500.0)
        assert mask.shape == (n_elec, n)
        assert mask.dtype == jnp.bool_

    def test_electrode_map_3d_depth_matters(self):
        """Neurons deeper in z should be less likely to be detected."""
        # Place neurons at two known depths: z=0 (on surface) and z=400
        n = 10
        surface_neurons = jnp.zeros((n, 3)).at[:, 0].set(1500.0).at[:, 1].set(1500.0)
        deep_neurons = jnp.zeros((n, 3)).at[:, 0].set(1500.0).at[:, 1].set(1500.0).at[:, 2].set(400.0)
        all_neurons = jnp.concatenate([surface_neurons, deep_neurons], axis=0)

        electrode_pos = jnp.array([[1500.0, 1500.0]])  # single electrode at (1500, 1500)
        radius = 200.0

        mask = build_neuron_electrode_map(all_neurons, electrode_pos, radius_um=radius)
        # Surface neurons (z=0) are at distance 0 from electrode -> detected
        # Deep neurons (z=400) are at distance 400 -> NOT detected (radius=200)
        assert jnp.all(mask[0, :n]), "Surface neurons should be detected"
        assert not jnp.any(mask[0, n:]), "Deep neurons should NOT be detected at radius=200"


# ---------------------------------------------------------------------------
# Test 6: 3D distance-dependent connectivity
# ---------------------------------------------------------------------------

class TestDistanceDependent3D:
    def test_close_pairs_denser_than_distant_3d(self):
        """In 3D, nearby neuron pairs should have more connections than distant ones."""
        n = 500
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, substrate_3d=(3000.0, 3000.0, 1300.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, _ = build_connectivity(key_conn, positions, is_exc)

        pos_np = np.asarray(positions)

        # Build dense boolean connectivity (union of E and I)
        connected = np.zeros((n, n), dtype=bool)
        if W_exc.data.shape[0] > 0:
            rows_e = np.asarray(W_exc.indices[:, 0])
            cols_e = np.asarray(W_exc.indices[:, 1])
            connected[rows_e, cols_e] = True
        if W_inh.data.shape[0] > 0:
            rows_i = np.asarray(W_inh.indices[:, 0])
            cols_i = np.asarray(W_inh.indices[:, 1])
            connected[rows_i, cols_i] = True

        # Compute 3D pairwise distances
        diff = pos_np[:, None, :] - pos_np[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)

        close_mask = dist < 200.0
        far_mask = dist > 800.0

        n_close = close_mask.sum()
        n_far = far_mask.sum()

        if n_close > 0 and n_far > 0:
            density_close = connected[close_mask].sum() / n_close
            density_far = connected[far_mask].sum() / n_far
            assert density_close > density_far, (
                f"Close density ({density_close:.4f}) should exceed "
                f"far density ({density_far:.4f})"
            )
        else:
            pytest.skip("Not enough close or far pairs for comparison")


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility — existing 2D code unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatible2D:
    def test_place_neurons_2d_unchanged(self):
        """Original 2D place_neurons call still works identically."""
        positions = place_neurons(KEY, 200, (3000.0, 3000.0))
        assert positions.shape == (200, 2)
        assert jnp.all(positions >= 0.0)
        assert jnp.all(positions <= 3000.0)

    def test_connectivity_2d_unchanged(self):
        """Original 2D connectivity still works."""
        n = 200
        positions = place_neurons(KEY, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(7)
        W_exc, W_inh, delays = build_connectivity(key_conn, positions, is_exc)

        assert isinstance(W_exc, BCOO)
        assert W_exc.shape == (n, n)
        assert isinstance(W_inh, BCOO)
        assert W_inh.shape == (n, n)
        assert isinstance(delays, BCOO)
        assert delays.shape == (n, n)

    def test_electrode_map_2d_unchanged(self):
        """Original 2D electrode mapping still works."""
        n = 100
        neuron_pos = place_neurons(KEY, n, (3000.0, 3000.0))
        electrode_pos = jnp.array([[1500.0, 1500.0], [1000.0, 1000.0]])
        mask = build_neuron_electrode_map(neuron_pos, electrode_pos, radius_um=200.0)
        assert mask.shape == (2, n)
        assert mask.dtype == jnp.bool_

    def test_2d_reproducibility(self):
        """2D placement with the same key produces the same result as before."""
        pos_a = place_neurons(jax.random.PRNGKey(42), 100, (3000.0, 3000.0))
        pos_b = place_neurons(jax.random.PRNGKey(42), 100, (3000.0, 3000.0))
        assert jnp.array_equal(pos_a, pos_b)
