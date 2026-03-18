"""Targeted tests to increase coverage of bl1.network.topology and bl1.network.types.

Focuses on uncovered code paths:
- topology.py: _build_connectivity_spatial, _build_connectivity_spatial_fast,
  place_neurons_layered edge case (count <= 0), build_connectivity large-N dispatch
- types.py: Culture.create with all placement strategies, error handling
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(0)


def _ei_mask(n: int, exc_frac: float = 0.8) -> jnp.ndarray:
    """Return a boolean excitatory mask."""
    n_exc = int(n * exc_frac)
    return jnp.arange(n) < n_exc


# ===========================================================================
# topology.py — _build_connectivity_spatial (lines 290-466)
# ===========================================================================


class TestBuildConnectivitySpatial:
    """Directly test the spatial-hashing connectivity builder."""

    def test_spatial_returns_bcoo_2d(self):
        """_build_connectivity_spatial returns BCOO matrices for 2D positions."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial(
            key_conn, positions, is_exc
        )
        assert isinstance(W_exc, BCOO)
        assert isinstance(W_inh, BCOO)
        assert isinstance(delays, BCOO)
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_spatial_returns_bcoo_3d(self):
        """_build_connectivity_spatial works with 3D positions."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 40
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, substrate_3d=(3000.0, 3000.0, 1300.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial(
            key_conn, positions, is_exc
        )
        assert isinstance(W_exc, BCOO)
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_spatial_no_self_connections(self):
        """Spatial builder should not produce self-connections."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, _ = _build_connectivity_spatial(key_conn, positions, is_exc)

        for W, name in [(W_exc, "W_exc"), (W_inh, "W_inh")]:
            if W.data.shape[0] > 0:
                rows = np.asarray(W.indices[:, 0])
                cols = np.asarray(W.indices[:, 1])
                assert np.all(rows != cols), f"{name} has self-connections"

    def test_spatial_delays_positive(self):
        """All delay values from spatial builder should be >= 1."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        _, _, delays = _build_connectivity_spatial(key_conn, positions, is_exc)

        if delays.data.shape[0] > 0:
            assert jnp.all(delays.data >= 1.0), (
                f"Found delay values < 1: min = {float(delays.data.min())}"
            )

    def test_spatial_ei_split(self):
        """Excitatory pre-synaptic neurons should only appear in W_exc."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        is_exc_np = np.asarray(is_exc)
        W_exc, W_inh, _ = _build_connectivity_spatial(key_conn, positions, is_exc)

        if W_exc.data.shape[0] > 0:
            rows = np.asarray(W_exc.indices[:, 0])
            cols = np.asarray(W_exc.indices[:, 1])
            # Either row or col convention: pre-synaptic must be excitatory
            assert np.all(is_exc_np[rows]) or np.all(is_exc_np[cols])

        if W_inh.data.shape[0] > 0:
            rows = np.asarray(W_inh.indices[:, 0])
            cols = np.asarray(W_inh.indices[:, 1])
            assert np.all(~is_exc_np[rows]) or np.all(~is_exc_np[cols])

    def test_spatial_dedup_produces_unique_pairs(self):
        """After deduplication, each (row, col) pair should appear at most once."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (2000.0, 2000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial(key_conn, positions, is_exc)

        for W, name in [(W_exc, "W_exc"), (W_inh, "W_inh"), (delays, "delays")]:
            if W.data.shape[0] > 0:
                indices_np = np.asarray(W.indices)
                pair_ids = indices_np[:, 0].astype(np.int64) * n + indices_np[:, 1].astype(
                    np.int64
                )
                assert len(pair_ids) == len(np.unique(pair_ids)), (
                    f"{name} contains duplicate (row, col) pairs"
                )

    def test_spatial_custom_params(self):
        """Spatial builder with custom weight and probability parameters."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1000.0, 1000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial(
            key_conn,
            positions,
            is_exc,
            lambda_um=100.0,
            p_max=0.5,
            g_exc=0.1,
            g_inh=0.4,
            v_axon_um_per_ms=200.0,
            dt=1.0,
        )
        assert W_exc.shape == (n, n)
        # With p_max=0.5 on a small substrate, we should have some connections
        total_nnz = W_exc.data.shape[0] + W_inh.data.shape[0]
        assert total_nnz > 0, "Expected at least some connections with p_max=0.5"

    def test_spatial_all_inhibitory(self):
        """When all neurons are inhibitory, W_exc should be empty."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1000.0, 1000.0))
        is_exc = jnp.zeros(n, dtype=jnp.bool_)  # all inhibitory
        W_exc, W_inh, _ = _build_connectivity_spatial(
            key_conn, positions, is_exc, p_max=0.5
        )
        assert W_exc.data.shape[0] == 0, "W_exc should be empty when all neurons are inhibitory"
        # W_inh should have entries
        assert W_inh.data.shape[0] > 0

    def test_spatial_all_excitatory(self):
        """When all neurons are excitatory, W_inh should be empty."""
        from bl1.network.topology import _build_connectivity_spatial, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1000.0, 1000.0))
        is_exc = jnp.ones(n, dtype=jnp.bool_)  # all excitatory
        W_exc, W_inh, _ = _build_connectivity_spatial(
            key_conn, positions, is_exc, p_max=0.5
        )
        assert W_inh.data.shape[0] == 0, "W_inh should be empty when all neurons are excitatory"
        assert W_exc.data.shape[0] > 0


# ===========================================================================
# topology.py — _build_connectivity_spatial_fast (lines 504-621)
# ===========================================================================


class TestBuildConnectivitySpatialFast:
    """Directly test the KD-tree-based connectivity builder."""

    def test_fast_returns_bcoo_2d(self):
        """_build_connectivity_spatial_fast returns BCOO for 2D positions."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial_fast(
            key_conn, positions, is_exc
        )
        assert isinstance(W_exc, BCOO)
        assert isinstance(W_inh, BCOO)
        assert isinstance(delays, BCOO)
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_fast_returns_bcoo_3d(self):
        """_build_connectivity_spatial_fast works with 3D positions."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 40
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, substrate_3d=(3000.0, 3000.0, 1300.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial_fast(
            key_conn, positions, is_exc
        )
        assert isinstance(W_exc, BCOO)
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_fast_no_self_connections(self):
        """KD-tree builder should not produce self-connections."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, _ = _build_connectivity_spatial_fast(key_conn, positions, is_exc)

        for W, name in [(W_exc, "W_exc"), (W_inh, "W_inh")]:
            if W.data.shape[0] > 0:
                rows = np.asarray(W.indices[:, 0])
                cols = np.asarray(W.indices[:, 1])
                assert np.all(rows != cols), f"{name} has self-connections"

    def test_fast_delays_positive(self):
        """All delay values from KD-tree builder should be >= 1."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        _, _, delays = _build_connectivity_spatial_fast(key_conn, positions, is_exc)

        if delays.data.shape[0] > 0:
            assert jnp.all(delays.data >= 1.0), (
                f"Found delay values < 1: min = {float(delays.data.min())}"
            )

    def test_fast_ei_split(self):
        """Pre-synaptic E/I split is correct in the fast builder."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 50
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (2000.0, 2000.0))
        is_exc = _ei_mask(n)
        is_exc_np = np.asarray(is_exc)
        W_exc, W_inh, _ = _build_connectivity_spatial_fast(key_conn, positions, is_exc)

        # The fast builder uses W[post, pre] convention: rows=post, cols=pre
        if W_exc.data.shape[0] > 0:
            cols = np.asarray(W_exc.indices[:, 1])  # pre-synaptic = column
            assert np.all(is_exc_np[cols]), (
                "W_exc pre-synaptic neurons (cols) should all be excitatory"
            )

        if W_inh.data.shape[0] > 0:
            cols = np.asarray(W_inh.indices[:, 1])  # pre-synaptic = column
            assert np.all(~is_exc_np[cols]), (
                "W_inh pre-synaptic neurons (cols) should all be inhibitory"
            )

    def test_fast_weight_magnitudes(self):
        """Weights should be close to the specified g_exc / g_inh (within noise)."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 50
        g_exc, g_inh = 0.05, 0.20
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (2000.0, 2000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, _ = _build_connectivity_spatial_fast(
            key_conn, positions, is_exc, g_exc=g_exc, g_inh=g_inh, p_max=0.5
        )

        if W_exc.data.shape[0] > 0:
            exc_data = np.asarray(W_exc.data)
            # Weights are g_exc * noise where noise is 1 +/- 10%
            assert np.all(exc_data >= g_exc * 0.89)
            assert np.all(exc_data <= g_exc * 1.11)

        if W_inh.data.shape[0] > 0:
            inh_data = np.asarray(W_inh.data)
            assert np.all(inh_data >= g_inh * 0.89)
            assert np.all(inh_data <= g_inh * 1.11)

    def test_fast_empty_when_neurons_far_apart(self):
        """When neurons are far apart, no connections should be made."""
        from bl1.network.topology import _build_connectivity_spatial_fast

        # Place 4 neurons very far apart (beyond 3*lambda cutoff)
        positions = jnp.array([
            [0.0, 0.0],
            [100000.0, 0.0],
            [0.0, 100000.0],
            [100000.0, 100000.0],
        ])
        is_exc = jnp.array([True, True, False, False])
        W_exc, W_inh, delays = _build_connectivity_spatial_fast(
            KEY, positions, is_exc, lambda_um=200.0
        )
        # With cutoff=600um and neurons 100000um apart, no connections
        assert W_exc.data.shape[0] == 0
        assert W_inh.data.shape[0] == 0
        assert delays.data.shape[0] == 0

    def test_fast_all_inhibitory(self):
        """When all neurons inhibitory, W_exc should be empty."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1000.0, 1000.0))
        is_exc = jnp.zeros(n, dtype=jnp.bool_)
        W_exc, W_inh, _ = _build_connectivity_spatial_fast(
            key_conn, positions, is_exc, p_max=0.5
        )
        assert W_exc.data.shape[0] == 0
        assert W_inh.data.shape[0] > 0

    def test_fast_all_excitatory(self):
        """When all neurons excitatory, W_inh should be empty."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1000.0, 1000.0))
        is_exc = jnp.ones(n, dtype=jnp.bool_)
        W_exc, W_inh, _ = _build_connectivity_spatial_fast(
            key_conn, positions, is_exc, p_max=0.5
        )
        assert W_inh.data.shape[0] == 0
        assert W_exc.data.shape[0] > 0

    def test_fast_custom_params(self):
        """KD-tree builder with custom parameters."""
        from bl1.network.topology import _build_connectivity_spatial_fast, place_neurons

        n = 40
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (1500.0, 1500.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = _build_connectivity_spatial_fast(
            key_conn,
            positions,
            is_exc,
            lambda_um=150.0,
            p_max=0.3,
            g_exc=0.08,
            g_inh=0.32,
            v_axon_um_per_ms=250.0,
            dt=0.25,
        )
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)
        assert delays.shape == (n, n)

    def test_fast_zero_connections_after_draw(self):
        """With extremely low p_max, no connections survive the random draw.

        This covers the early-return path at lines 560-565.
        """
        from bl1.network.topology import _build_connectivity_spatial_fast

        # Neurons within cutoff but p_max so low that no connections survive
        positions = jnp.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
        ])
        is_exc = jnp.array([True, True, False])
        W_exc, W_inh, delays = _build_connectivity_spatial_fast(
            KEY, positions, is_exc, lambda_um=200.0, p_max=1e-15
        )
        # With p_max=1e-15 the probability is essentially zero
        assert W_exc.data.shape[0] == 0
        assert W_inh.data.shape[0] == 0
        assert delays.data.shape[0] == 0


# ===========================================================================
# topology.py — build_connectivity dispatching to large-N path (line 688)
# ===========================================================================


class TestBuildConnectivityDispatch:
    """Test that build_connectivity dispatches based on network size."""

    def test_small_network_uses_dense(self):
        """Networks below the threshold use the dense path."""
        from bl1.network.topology import _DENSE_THRESHOLD, build_connectivity, place_neurons

        n = 50  # well below threshold
        assert n < _DENSE_THRESHOLD
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000.0, 3000.0))
        is_exc = _ei_mask(n)
        W_exc, W_inh, delays = build_connectivity(key_conn, positions, is_exc)
        assert W_exc.shape == (n, n)

    def test_threshold_value(self):
        """Verify the dense threshold constant."""
        from bl1.network.topology import _DENSE_THRESHOLD

        assert _DENSE_THRESHOLD == 10_000

    def test_large_network_dispatches_to_spatial_fast(self):
        """Networks at or above the threshold use the spatial_fast path.

        We monkeypatch the threshold to a small value to avoid creating
        a 10K-neuron network.
        """
        import bl1.network.topology as topo
        from bl1.network.topology import build_connectivity, place_neurons

        n = 30
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (2000.0, 2000.0))
        is_exc = _ei_mask(n)

        original_threshold = topo._DENSE_THRESHOLD
        try:
            topo._DENSE_THRESHOLD = 10  # force the spatial_fast path
            W_exc, W_inh, delays = build_connectivity(key_conn, positions, is_exc)
            assert W_exc.shape == (n, n)
            assert W_inh.shape == (n, n)
            assert delays.shape == (n, n)
        finally:
            topo._DENSE_THRESHOLD = original_threshold


# ===========================================================================
# topology.py — place_neurons_layered edge case (line 178: count <= 0)
# ===========================================================================


class TestPlaceNeuronsLayeredEdge:
    """Test edge cases in place_neurons_layered."""

    def test_layered_zero_density_layer(self):
        """A layer with zero density should produce zero neurons in that layer."""
        from bl1.network.topology import place_neurons_layered

        # Layer 0 has zero density, so count will be 0 -> triggers line 177
        depths = (100.0, 200.0, 300.0)
        densities = (0.0, 0.5, 0.5)
        positions = place_neurons_layered(
            KEY,
            20,
            substrate_um=(1000.0, 1000.0),
            layer_depths_um=depths,
            layer_densities=densities,
        )
        assert positions.shape == (20, 3)
        z = np.asarray(positions[:, 2])
        # No neurons should be in layer 0 (z < 100)
        assert np.sum(z < 100.0) == 0

    def test_layered_mismatched_lengths_raises(self):
        """Mismatched layer_depths_um and layer_densities should raise."""
        from bl1.network.topology import place_neurons_layered

        with pytest.raises(AssertionError, match="same length"):
            place_neurons_layered(
                KEY,
                20,
                layer_depths_um=(100.0, 200.0),
                layer_densities=(0.5,),
            )

    def test_layered_single_layer(self):
        """A single layer should place all neurons in that depth range."""
        from bl1.network.topology import place_neurons_layered

        depth = 500.0
        positions = place_neurons_layered(
            KEY,
            30,
            substrate_um=(2000.0, 2000.0),
            layer_depths_um=(depth,),
            layer_densities=(1.0,),
        )
        assert positions.shape == (30, 3)
        z = np.asarray(positions[:, 2])
        assert np.all(z >= 0.0)
        assert np.all(z <= depth + 1e-3)


# ===========================================================================
# types.py — Culture.create (lines 261-343)
# ===========================================================================


class TestCultureCreate:
    """Test Culture.create with various placement and connectivity options."""

    def test_create_default_2d(self):
        """Default Culture.create returns valid structures."""
        from bl1.network.types import Culture, CultureState, NetworkParams

        net_params, state, izh_params = Culture.create(KEY, n_neurons=20, dt=0.5)

        assert isinstance(net_params, NetworkParams)
        assert isinstance(state, CultureState)
        assert net_params.positions.shape == (20, 2)
        assert net_params.is_excitatory.shape == (20,)
        assert net_params.W_exc.shape == (20, 20)
        assert net_params.W_inh.shape == (20, 20)
        assert net_params.delays.shape == (20, 20)

        # CultureState fields
        assert state.v.shape == (20,)
        assert state.u.shape == (20,)
        assert state.spikes.shape == (20,)
        assert state.g_ampa.shape == (20,)
        assert state.g_gaba_a.shape == (20,)
        assert state.stdp_pre_trace.shape == (20,)
        assert state.stdp_post_trace.shape == (20,)
        assert state.W_exc.shape == (20, 20)
        assert float(state.t) == 0.0

    def test_create_3d_uniform(self):
        """Culture.create with 3D uniform placement."""
        from bl1.network.types import Culture

        net_params, state, izh_params = Culture.create(
            KEY,
            n_neurons=20,
            placement="uniform",
            substrate_3d=(3000.0, 3000.0, 1300.0),
            dt=0.5,
        )
        assert net_params.positions.shape == (20, 3)

    def test_create_spheroid(self):
        """Culture.create with spheroid placement."""
        from bl1.network.types import Culture

        net_params, state, izh_params = Culture.create(
            KEY,
            n_neurons=25,
            placement="spheroid",
            spheroid_radius_um=400.0,
            dt=0.5,
        )
        assert net_params.positions.shape == (25, 3)
        # Check positions are within the spheroid
        center = jnp.array([400.0, 400.0, 400.0])
        dists = jnp.sqrt(jnp.sum((net_params.positions - center) ** 2, axis=-1))
        assert jnp.all(dists < 400.0 + 1e-3)

    def test_create_spheroid_custom_center(self):
        """Culture.create with spheroid placement and custom center."""
        from bl1.network.types import Culture

        center = (1000.0, 2000.0, 500.0)
        net_params, _, _ = Culture.create(
            KEY,
            n_neurons=20,
            placement="spheroid",
            spheroid_radius_um=300.0,
            spheroid_center_um=center,
            dt=0.5,
        )
        assert net_params.positions.shape == (20, 3)

    def test_create_layered(self):
        """Culture.create with layered placement."""
        from bl1.network.types import Culture

        net_params, state, izh_params = Culture.create(
            KEY,
            n_neurons=30,
            placement="layered",
            dt=0.5,
        )
        assert net_params.positions.shape == (30, 3)

    def test_create_layered_custom_depths(self):
        """Culture.create with layered placement and custom layer parameters."""
        from bl1.network.types import Culture

        net_params, _, _ = Culture.create(
            KEY,
            n_neurons=20,
            placement="layered",
            layer_depths_um=(200.0, 400.0, 200.0),
            layer_densities=(0.3, 0.5, 0.2),
            dt=0.5,
        )
        assert net_params.positions.shape == (20, 3)

    def test_create_invalid_neuron_model(self):
        """Culture.create with unsupported neuron model should raise."""
        from bl1.network.types import Culture

        with pytest.raises(ValueError, match="Unsupported neuron model"):
            Culture.create(KEY, n_neurons=10, neuron_model="hodgkin_huxley")

    def test_create_invalid_connectivity(self):
        """Culture.create with unsupported connectivity should raise."""
        from bl1.network.types import Culture

        with pytest.raises(ValueError, match="Unsupported connectivity"):
            Culture.create(KEY, n_neurons=10, connectivity="random_graph")

    def test_create_invalid_placement(self):
        """Culture.create with unsupported placement should raise."""
        from bl1.network.types import Culture

        with pytest.raises(ValueError, match="Unsupported placement"):
            Culture.create(KEY, n_neurons=10, placement="hexagonal")

    def test_create_custom_connectivity_params(self):
        """Culture.create with custom connectivity parameters."""
        from bl1.network.types import Culture

        net_params, state, izh_params = Culture.create(
            KEY,
            n_neurons=20,
            lambda_um=150.0,
            p_max=0.3,
            g_exc=0.08,
            g_inh=0.32,
            dt=1.0,
        )
        assert net_params.W_exc.shape == (20, 20)

    def test_create_custom_ei_ratio(self):
        """Culture.create respects the ei_ratio parameter."""
        from bl1.network.types import Culture

        net_params, _, _ = Culture.create(
            KEY, n_neurons=20, ei_ratio=0.5, dt=0.5
        )
        n_exc = int(np.asarray(net_params.is_excitatory).sum())
        assert n_exc == 10

    def test_create_state_initial_values(self):
        """CultureState should have sensible initial values."""
        from bl1.network.types import Culture

        _, state, izh_params = Culture.create(KEY, n_neurons=15, dt=0.5)
        # v should be -65 mV
        np.testing.assert_allclose(np.asarray(state.v), -65.0)
        # u should be b * v
        expected_u = np.asarray(izh_params.b) * np.asarray(state.v)
        np.testing.assert_allclose(np.asarray(state.u), expected_u, atol=1e-5)
        # No spikes at t=0
        assert not np.any(np.asarray(state.spikes))
        # Conductances start at zero
        np.testing.assert_allclose(np.asarray(state.g_ampa), 0.0)
        np.testing.assert_allclose(np.asarray(state.g_gaba_a), 0.0)
        # Traces start at zero
        np.testing.assert_allclose(np.asarray(state.stdp_pre_trace), 0.0)
        np.testing.assert_allclose(np.asarray(state.stdp_post_trace), 0.0)


# ===========================================================================
# types.py — NetworkParams and CultureState NamedTuple construction
# ===========================================================================


class TestNetworkParamsAndCultureState:
    """Test direct construction and field access of the NamedTuples."""

    def test_network_params_fields(self):
        """NetworkParams can be constructed and fields accessed."""
        from bl1.network.types import NetworkParams

        pos = jnp.zeros((5, 2))
        is_exc = jnp.ones(5, dtype=jnp.bool_)
        W = BCOO.fromdense(jnp.zeros((5, 5)))
        np_ = NetworkParams(
            positions=pos,
            is_excitatory=is_exc,
            W_exc=W,
            W_inh=W,
            delays=W,
        )
        assert np_.positions.shape == (5, 2)
        assert np_.is_excitatory.shape == (5,)

    def test_culture_state_fields(self):
        """CultureState can be constructed and fields accessed."""
        from bl1.network.types import CultureState

        N = 5
        W = BCOO.fromdense(jnp.zeros((N, N)))
        state = CultureState(
            v=jnp.zeros(N),
            u=jnp.zeros(N),
            spikes=jnp.zeros(N, dtype=jnp.bool_),
            g_ampa=jnp.zeros(N),
            g_gaba_a=jnp.zeros(N),
            stdp_pre_trace=jnp.zeros(N),
            stdp_post_trace=jnp.zeros(N),
            W_exc=W,
            t=jnp.array(0.0),
        )
        assert state.v.shape == (N,)
        assert float(state.t) == 0.0


# ===========================================================================
# types.py — fallback IzhikevichParams and create_population (lines 99-168)
# These are only used when bl1.core.izhikevich is not importable.
# We test the fallback functions directly via their module-level definitions.
# ===========================================================================


class TestFallbackPopulation:
    """Test the fallback create_population logic that lives in types.py.

    Since _HAS_CORE is True in this environment, we test the core
    create_population instead (which is what types.py actually uses).
    This ensures the code path through Culture.create step 2 is covered.
    """

    def test_create_population_shapes(self):
        """create_population returns correct shapes."""
        from bl1.core.izhikevich import create_population

        params, state, is_exc = create_population(KEY, 20, ei_ratio=0.8)
        assert params.a.shape == (20,)
        assert params.b.shape == (20,)
        assert params.c.shape == (20,)
        assert params.d.shape == (20,)
        assert is_exc.shape == (20,)
        n_exc = int(np.asarray(is_exc).sum())
        assert n_exc == 16  # 80% of 20

    def test_create_population_different_ratios(self):
        """create_population with different E/I ratios."""
        from bl1.core.izhikevich import create_population

        _, _, is_exc = create_population(KEY, 30, ei_ratio=0.5)
        n_exc = int(np.asarray(is_exc).sum())
        assert n_exc == 15


# ===========================================================================
# topology.py — place_neurons_spheroid edge cases
# ===========================================================================


class TestPlaceNeuronsSpheroidEdge:
    """Additional edge-case tests for spheroid placement."""

    def test_spheroid_default_center(self):
        """Default center should be (radius, radius, radius)."""
        from bl1.network.topology import place_neurons_spheroid

        radius = 300.0
        positions = place_neurons_spheroid(KEY, 50, radius_um=radius)
        center = jnp.array([radius, radius, radius])
        dists = jnp.sqrt(jnp.sum((positions - center) ** 2, axis=-1))
        assert jnp.all(dists < radius + 1e-3)

    def test_spheroid_small_n(self):
        """Spheroid placement with very few neurons."""
        from bl1.network.topology import place_neurons_spheroid

        positions = place_neurons_spheroid(KEY, 3, radius_um=100.0)
        assert positions.shape == (3, 3)


# ===========================================================================
# Integration: spatial and fast builders produce similar statistics
# ===========================================================================


class TestSpatialVsFast:
    """Compare the two large-network builders for statistical consistency."""

    def test_both_produce_connections(self):
        """Both spatial builders should produce non-empty connectivity."""
        from bl1.network.topology import (
            _build_connectivity_spatial,
            _build_connectivity_spatial_fast,
            place_neurons,
        )

        n = 50
        key_pos, key_conn1, key_conn2 = jax.random.split(KEY, 3)
        positions = place_neurons(key_pos, n, (2000.0, 2000.0))
        is_exc = _ei_mask(n)

        W_exc_s, W_inh_s, _ = _build_connectivity_spatial(
            key_conn1, positions, is_exc, p_max=0.5
        )
        W_exc_f, W_inh_f, _ = _build_connectivity_spatial_fast(
            key_conn2, positions, is_exc, p_max=0.5
        )

        # Both should produce some connections
        nnz_spatial = W_exc_s.data.shape[0] + W_inh_s.data.shape[0]
        nnz_fast = W_exc_f.data.shape[0] + W_inh_f.data.shape[0]
        assert nnz_spatial > 0, "Spatial builder produced no connections"
        assert nnz_fast > 0, "Fast builder produced no connections"
