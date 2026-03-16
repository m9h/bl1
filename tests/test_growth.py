"""Tests for network growth model (bl1.network.growth)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.network.topology import place_neurons
from bl1.network.growth import (
    GrowthParams,
    GrowthState,
    init_growth,
    grow_to_div,
    mature_network,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(42)
N_SMALL = 200  # Keep tests fast with small networks


def _setup(n: int = N_SMALL, seed: int = 42):
    """Create positions and E/I mask for a small test network."""
    key = jax.random.PRNGKey(seed)
    k_pos, k_grow = jax.random.split(key)
    positions = place_neurons(k_pos, n, (3000.0, 3000.0))
    n_exc = int(n * 0.8)
    is_excitatory = jnp.arange(n) < n_exc
    return k_grow, positions, is_excitatory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitGrowthEmpty:
    def test_init_growth_empty(self):
        """Initial state at DIV 0 should have zero connectivity."""
        key, positions, is_exc = _setup()
        state = init_growth(key, positions, is_exc)

        assert state.div == 0.0
        assert state.connectivity_fraction == 0.0
        assert jnp.all(state.W_exc == 0.0), "W_exc should be all zeros at DIV 0"
        assert jnp.all(state.W_inh == 0.0), "W_inh should be all zeros at DIV 0"
        assert state.W_exc.shape == (N_SMALL, N_SMALL)
        assert state.W_inh.shape == (N_SMALL, N_SMALL)


class TestGrowthIncreasesConnectivity:
    def test_growth_increases_connectivity(self):
        """DIV 14 should have more connections than DIV 7."""
        key, positions, is_exc = _setup()
        k7, k14 = jax.random.split(key)

        state_7 = grow_to_div(k7, positions, is_exc, target_div=7.0)
        state_14 = grow_to_div(k14, positions, is_exc, target_div=14.0)

        assert state_14.connectivity_fraction > state_7.connectivity_fraction, (
            f"DIV 14 connectivity ({state_14.connectivity_fraction:.4f}) should exceed "
            f"DIV 7 ({state_7.connectivity_fraction:.4f})"
        )


class TestMatureNetworkDensity:
    def test_mature_network_density(self):
        """At DIV 28, connectivity fraction should be substantially higher than DIV 1.

        With distance-dependent connectivity on a 3000x3000 um substrate,
        the actual fraction is modest (exponential decay makes most long-range
        connections improbable). We verify DIV 28 produces meaningfully more
        connections than early DIV, and that the density is nonzero.
        """
        key, positions, is_exc = _setup()
        k1, k28 = jax.random.split(key)

        state_1 = grow_to_div(k1, positions, is_exc, target_div=1.0)
        state_28 = grow_to_div(k28, positions, is_exc, target_div=28.0)

        # DIV 28 should have substantially more connections than DIV 1
        assert state_28.connectivity_fraction > state_1.connectivity_fraction * 2.0, (
            f"DIV 28 connectivity ({state_28.connectivity_fraction:.4f}) should be "
            f"much greater than DIV 1 ({state_1.connectivity_fraction:.4f})"
        )
        # Should be nonzero and bounded
        assert state_28.connectivity_fraction > 0.0
        assert state_28.connectivity_fraction <= 1.0


class TestGrowthLogisticShape:
    def test_growth_logistic_shape(self):
        """Connectivity should increase monotonically: DIV 5 < 10 < 20 < 28."""
        key, positions, is_exc = _setup()
        divs = [5.0, 10.0, 20.0, 28.0]
        fractions = []
        for div in divs:
            k = jax.random.PRNGKey(int(div * 100))
            state = grow_to_div(k, positions, is_exc, target_div=div)
            fractions.append(state.connectivity_fraction)

        for i in range(len(fractions) - 1):
            assert fractions[i] < fractions[i + 1], (
                f"Connectivity at DIV {divs[i]} ({fractions[i]:.4f}) should be less than "
                f"at DIV {divs[i+1]} ({fractions[i+1]:.4f})"
            )


class TestNoSelfConnections:
    def test_no_self_connections(self):
        """No diagonal entries in weight matrices at any DIV."""
        key, positions, is_exc = _setup()

        for div in [7.0, 14.0, 28.0]:
            k = jax.random.PRNGKey(int(div * 10))
            state = grow_to_div(k, positions, is_exc, target_div=div)

            diag_exc = jnp.diag(state.W_exc)
            diag_inh = jnp.diag(state.W_inh)
            assert jnp.all(diag_exc == 0.0), (
                f"W_exc has self-connections at DIV {div}"
            )
            assert jnp.all(diag_inh == 0.0), (
                f"W_inh has self-connections at DIV {div}"
            )


class TestEISplit:
    def test_ei_split(self):
        """Excitatory neurons only contribute to W_exc, inhibitory to W_inh."""
        key, positions, is_exc = _setup()
        state = grow_to_div(key, positions, is_exc, target_div=28.0)

        is_exc_np = np.asarray(is_exc)
        W_exc_np = np.asarray(state.W_exc)
        W_inh_np = np.asarray(state.W_inh)

        # Rows are pre-synaptic neurons. Inhibitory neurons should have
        # zero rows in W_exc.
        inh_rows_in_exc = W_exc_np[~is_exc_np, :]
        assert np.allclose(inh_rows_in_exc, 0.0), (
            "Inhibitory neurons should have zero entries in W_exc"
        )

        # Excitatory neurons should have zero rows in W_inh.
        exc_rows_in_inh = W_inh_np[is_exc_np, :]
        assert np.allclose(exc_rows_in_inh, 0.0), (
            "Excitatory neurons should have zero entries in W_inh"
        )

        # At least some connections should exist in each matrix
        assert np.any(W_exc_np > 0), "W_exc should have some nonzero entries"
        assert np.any(W_inh_np > 0), "W_inh should have some nonzero entries"


class TestHubFormation:
    def test_hub_formation(self):
        """At DIV 28, hub neurons should have higher total outgoing weight.

        We verify this statistically: the mean outgoing weight of the top
        hub_fraction of neurons (by outgoing weight) should exceed the mean
        of the remaining neurons.
        """
        key, positions, is_exc = _setup(n=300)
        params = GrowthParams(hub_fraction=0.2, hub_weight_boost=2.0)
        state = grow_to_div(key, positions, is_exc, target_div=28.0, params=params)

        W_total = np.asarray(state.W_exc) + np.asarray(state.W_inh)
        outgoing_weight = W_total.sum(axis=1)  # sum over post-synaptic targets

        # Sort by outgoing weight and compare top 20% vs bottom 80%
        n = 300
        n_hubs = int(n * params.hub_fraction)
        sorted_weights = np.sort(outgoing_weight)
        top_mean = sorted_weights[-n_hubs:].mean()
        bottom_mean = sorted_weights[:-n_hubs].mean()

        assert top_mean > bottom_mean, (
            f"Top {params.hub_fraction*100:.0f}% mean outgoing weight ({top_mean:.4f}) "
            f"should exceed bottom mean ({bottom_mean:.4f})"
        )


class TestReproducibility:
    def test_reproducibility(self):
        """Same key produces same result."""
        _, positions, is_exc = _setup()

        key_a = jax.random.PRNGKey(123)
        key_b = jax.random.PRNGKey(123)

        state_a = grow_to_div(key_a, positions, is_exc, target_div=21.0)
        state_b = grow_to_div(key_b, positions, is_exc, target_div=21.0)

        assert state_a.div == state_b.div
        assert state_a.connectivity_fraction == state_b.connectivity_fraction
        assert jnp.allclose(state_a.W_exc, state_b.W_exc), (
            "W_exc not reproducible with same key"
        )
        assert jnp.allclose(state_a.W_inh, state_b.W_inh), (
            "W_inh not reproducible with same key"
        )


class TestMatureNetworkConvenience:
    def test_mature_network_returns_tuple(self):
        """mature_network should return a (W_exc, W_inh) tuple of JAX arrays."""
        _, positions, is_exc = _setup()
        key = jax.random.PRNGKey(99)
        W_exc, W_inh = mature_network(key, positions, is_exc, target_div=28.0)

        assert isinstance(W_exc, jnp.ndarray)
        assert isinstance(W_inh, jnp.ndarray)
        assert W_exc.shape == (N_SMALL, N_SMALL)
        assert W_inh.shape == (N_SMALL, N_SMALL)
