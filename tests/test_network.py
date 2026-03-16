"""Tests for network topology (bl1.network.topology)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from bl1.network.topology import build_connectivity, place_neurons


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(42)


def _ei_mask(n: int, exc_frac: float = 0.8) -> jnp.ndarray:
    """Return a boolean excitatory mask: first exc_frac of neurons are excitatory."""
    n_exc = int(n * exc_frac)
    return jnp.arange(n) < n_exc


# ---------------------------------------------------------------------------
# place_neurons tests
# ---------------------------------------------------------------------------

class TestPlaceNeurons:
    def test_place_neurons_shape(self):
        """place_neurons returns shape (n_neurons, 2)."""
        positions = place_neurons(KEY, 1000, (3000, 3000))
        assert positions.shape == (1000, 2)

    def test_place_neurons_bounds(self):
        """All positions should lie within [0, 3000] x [0, 3000]."""
        positions = place_neurons(KEY, 1000, (3000, 3000))
        assert jnp.all(positions >= 0.0)
        assert jnp.all(positions <= 3000.0)


# ---------------------------------------------------------------------------
# build_connectivity tests
# ---------------------------------------------------------------------------

class TestBuildConnectivitySmall:
    def test_build_connectivity_returns_bcoo(self):
        """W_exc and W_inh should be BCOO matrices of shape (100, 100) with no self-connections."""
        n = 100
        positions = place_neurons(KEY, n, (3000, 3000))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(7)
        W_exc, W_inh, delays = build_connectivity(key_conn, positions, is_exc)

        # Check types
        assert isinstance(W_exc, BCOO), f"W_exc should be BCOO, got {type(W_exc)}"
        assert isinstance(W_inh, BCOO), f"W_inh should be BCOO, got {type(W_inh)}"

        # Check shapes
        assert W_exc.shape == (n, n)
        assert W_inh.shape == (n, n)

        # Check no self-connections: for every stored entry, row != col
        if W_exc.data.shape[0] > 0:
            rows_exc = W_exc.indices[:, 0]
            cols_exc = W_exc.indices[:, 1]
            assert jnp.all(rows_exc != cols_exc), "W_exc has self-connections"

        if W_inh.data.shape[0] > 0:
            rows_inh = W_inh.indices[:, 0]
            cols_inh = W_inh.indices[:, 1]
            assert jnp.all(rows_inh != cols_inh), "W_inh has self-connections"


class TestConnectivityEISplit:
    def test_connectivity_ei_split(self):
        """Excitatory neurons should only appear as pre-synaptic in W_exc,
        inhibitory neurons only in W_inh.

        Pre-synaptic neuron indices are in the column dimension under the
        W[post, pre] convention.  However, the connectivity builder uses
        rows as the source/pre-synaptic dimension (W[pre, post]).  We test
        both the column index (W[j,i] convention) and fall back to row index
        if needed, to catch either convention.
        """
        n = 500
        positions = place_neurons(KEY, n, (3000, 3000))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(99)
        W_exc, W_inh, _ = build_connectivity(key_conn, positions, is_exc)

        is_exc_np = np.asarray(is_exc)

        # W_exc: all pre-synaptic indices should be excitatory neurons.
        # Check column indices (convention W[post, pre]) first.
        if W_exc.data.shape[0] > 0:
            cols_exc = np.asarray(W_exc.indices[:, 1])
            rows_exc = np.asarray(W_exc.indices[:, 0])
            # At least one convention should hold: either columns or rows are all excitatory
            col_convention_ok = np.all(is_exc_np[cols_exc])
            row_convention_ok = np.all(is_exc_np[rows_exc])
            assert col_convention_ok or row_convention_ok, (
                "Neither row nor column indices of W_exc are all excitatory neurons"
            )

        # W_inh: all pre-synaptic indices should be inhibitory neurons.
        if W_inh.data.shape[0] > 0:
            cols_inh = np.asarray(W_inh.indices[:, 1])
            rows_inh = np.asarray(W_inh.indices[:, 0])
            col_convention_ok = np.all(~is_exc_np[cols_inh])
            row_convention_ok = np.all(~is_exc_np[rows_inh])
            assert col_convention_ok or row_convention_ok, (
                "Neither row nor column indices of W_inh are all inhibitory neurons"
            )


class TestDistanceDependentProbability:
    def test_close_pairs_denser_than_distant(self):
        """Nearby neuron pairs should have more connections than distant pairs."""
        n = 1000
        key_pos, key_conn = jax.random.split(KEY)
        positions = place_neurons(key_pos, n, (3000, 3000))
        is_exc = _ei_mask(n)
        W_exc, W_inh, _ = build_connectivity(key_conn, positions, is_exc)

        pos_np = np.asarray(positions)

        # Build a dense boolean connectivity matrix (union of E and I)
        connected = np.zeros((n, n), dtype=bool)

        if W_exc.data.shape[0] > 0:
            rows_e = np.asarray(W_exc.indices[:, 0])
            cols_e = np.asarray(W_exc.indices[:, 1])
            connected[rows_e, cols_e] = True

        if W_inh.data.shape[0] > 0:
            rows_i = np.asarray(W_inh.indices[:, 0])
            cols_i = np.asarray(W_inh.indices[:, 1])
            connected[rows_i, cols_i] = True

        # Compute pairwise distances
        diff = pos_np[:, None, :] - pos_np[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Mask out diagonal
        np.fill_diagonal(dist, np.inf)

        close_mask = dist < 200.0
        far_mask = dist > 800.0

        n_close_pairs = close_mask.sum()
        n_far_pairs = far_mask.sum()

        if n_close_pairs > 0 and n_far_pairs > 0:
            density_close = connected[close_mask].sum() / n_close_pairs
            density_far = connected[far_mask].sum() / n_far_pairs
            assert density_close > density_far, (
                f"Close density ({density_close:.4f}) should exceed far density ({density_far:.4f})"
            )
        else:
            pytest.skip("Not enough close or far pairs for meaningful comparison")


class TestDelaysPositive:
    def test_delays_positive(self):
        """All delay values should be >= 1 (minimum 1 timestep)."""
        n = 200
        positions = place_neurons(KEY, n, (3000, 3000))
        is_exc = _ei_mask(n)
        key_conn = jax.random.PRNGKey(13)
        _, _, delays = build_connectivity(key_conn, positions, is_exc)

        assert isinstance(delays, BCOO)
        if delays.data.shape[0] > 0:
            assert jnp.all(delays.data >= 1.0), (
                f"Found delay values < 1: min = {delays.data.min()}"
            )


class TestBuildConnectivityReproducibility:
    def test_reproducibility(self):
        """Same PRNG key should produce identical connectivity."""
        n = 200
        positions = place_neurons(KEY, n, (3000, 3000))
        is_exc = _ei_mask(n)

        key_conn = jax.random.PRNGKey(77)
        W_exc_a, W_inh_a, delays_a = build_connectivity(key_conn, positions, is_exc)

        key_conn = jax.random.PRNGKey(77)
        W_exc_b, W_inh_b, delays_b = build_connectivity(key_conn, positions, is_exc)

        # Compare data arrays
        assert jnp.allclose(W_exc_a.data, W_exc_b.data), "W_exc data not reproducible"
        assert jnp.array_equal(W_exc_a.indices, W_exc_b.indices), "W_exc indices not reproducible"

        assert jnp.allclose(W_inh_a.data, W_inh_b.data), "W_inh data not reproducible"
        assert jnp.array_equal(W_inh_a.indices, W_inh_b.indices), "W_inh indices not reproducible"

        assert jnp.allclose(delays_a.data, delays_b.data), "delays data not reproducible"
        assert jnp.array_equal(delays_a.indices, delays_b.indices), "delays indices not reproducible"
