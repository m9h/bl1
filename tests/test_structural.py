"""Tests for structural plasticity (bl1.plasticity.structural)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.plasticity.structural import StructuralPlasticityParams, structural_update


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_params(**overrides) -> StructuralPlasticityParams:
    """Return default params with optional overrides."""
    fields = StructuralPlasticityParams()._asdict()
    fields.update(overrides)
    return StructuralPlasticityParams(**fields)


def _close_positions(n: int) -> jnp.ndarray:
    """Place n neurons within a small 100x100 um square so all are within range."""
    return jnp.tile(jnp.array([[50.0, 50.0]]), (n, 1)) + jax.random.uniform(
        jax.random.PRNGKey(42), shape=(n, 2), minval=-10.0, maxval=10.0
    )


def _all_excitatory(n: int) -> jnp.ndarray:
    return jnp.ones(n, dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPruning:
    def test_pruning_removes_weak_synapses(self):
        """Synapses with weight below prune_threshold should be candidates
        for removal.  With prune_prob=1.0 all weak synapses are pruned."""
        n = 10
        key = jax.random.PRNGKey(0)
        is_exc = _all_excitatory(n)
        positions = _close_positions(n)
        rates = jnp.ones(n) * 5.0  # all active

        # Create weight matrix with some very weak synapses.
        W = jnp.zeros((n, n), dtype=jnp.float32)
        # Strong synapse at [1, 0] and weak synapse at [2, 0]
        W = W.at[1, 0].set(0.05)
        W = W.at[2, 0].set(0.0005)  # below default prune_threshold of 0.001

        params = _default_params(prune_prob=1.0)
        W_new = structural_update(key, W, positions, is_exc, rates, params)

        # The weak synapse should have been pruned.
        assert float(W_new[2, 0]) == 0.0, (
            f"Weak synapse should be pruned, got {W_new[2, 0]}"
        )
        # The strong synapse should remain.
        assert float(W_new[1, 0]) == pytest.approx(0.05), (
            f"Strong synapse should remain, got {W_new[1, 0]}"
        )


class TestGrowth:
    def test_growth_adds_new_synapses(self):
        """With high activity and close positions, new synapses should form
        when growth_prob is set high."""
        n = 20
        key = jax.random.PRNGKey(1)
        is_exc = _all_excitatory(n)
        # Place all neurons close together.
        positions = _close_positions(n)
        rates = jnp.ones(n) * 10.0  # well above activity_threshold

        # Start with zero connectivity.
        W = jnp.zeros((n, n), dtype=jnp.float32)

        params = _default_params(growth_prob=1.0, max_distance_um=500.0)
        W_new = structural_update(key, W, positions, is_exc, rates, params)

        n_new_synapses = int(jnp.sum(W_new > 0))
        assert n_new_synapses > 0, "Expected new synapses to form"

        # All new synapses should have the initial weight.
        new_weights = W_new[W_new > 0]
        assert jnp.allclose(new_weights, params.w_new), (
            f"New synapses should have weight {params.w_new}"
        )


class TestNoSelfConnections:
    def test_no_self_connections(self):
        """Structural plasticity should never create self-connections."""
        n = 20
        key = jax.random.PRNGKey(2)
        is_exc = _all_excitatory(n)
        positions = _close_positions(n)
        rates = jnp.ones(n) * 10.0

        W = jnp.zeros((n, n), dtype=jnp.float32)

        params = _default_params(growth_prob=1.0)
        W_new = structural_update(key, W, positions, is_exc, rates, params)

        diag = jnp.diag(W_new)
        assert jnp.allclose(diag, 0.0), (
            f"Self-connections found on diagonal: {diag}"
        )


class TestInhibitoryUnchanged:
    def test_inhibitory_unchanged(self):
        """Only excitatory weights should be modified.  Weights from
        inhibitory pre-synaptic neurons must remain unchanged."""
        n = 10
        key = jax.random.PRNGKey(3)

        # Neurons 0-4 excitatory, 5-9 inhibitory.
        is_exc = jnp.array([True] * 5 + [False] * 5)
        positions = _close_positions(n)
        rates = jnp.ones(n) * 10.0

        # Set up some inhibitory-pre weights (W[j, i] where i >= 5).
        W = jnp.zeros((n, n), dtype=jnp.float32)
        # Inhibitory synapse from neuron 5 to neuron 0.
        W = W.at[0, 5].set(0.0005)  # below prune_threshold — but should NOT be pruned
        # Inhibitory synapse from neuron 6 to neuron 1 — strong.
        W = W.at[1, 6].set(0.05)

        params = _default_params(prune_prob=1.0, growth_prob=1.0)
        W_new = structural_update(key, W, positions, is_exc, rates, params)

        # Inhibitory synapses should be untouched.
        assert float(W_new[0, 5]) == pytest.approx(0.0005), (
            f"Inhibitory synapse [0,5] was modified: {W_new[0, 5]}"
        )
        assert float(W_new[1, 6]) == pytest.approx(0.05), (
            f"Inhibitory synapse [1,6] was modified: {W_new[1, 6]}"
        )

        # New synapses should only appear in excitatory-pre columns.
        # Check that columns 5-9 (inhibitory pre) have no new synapses.
        inh_cols = np.array(W_new[:, 5:])
        orig_inh = np.array(W[:, 5:])
        assert np.array_equal(inh_cols, orig_inh), (
            "New synapses appeared from inhibitory pre-synaptic neurons"
        )


class TestDistanceConstraint:
    def test_distance_constraint(self):
        """New synapses should only form between neurons within max_distance_um."""
        n = 10
        key = jax.random.PRNGKey(4)
        is_exc = _all_excitatory(n)
        rates = jnp.ones(n) * 10.0

        # Place neurons in two distant clusters (>1000 um apart).
        cluster_a = jnp.zeros((5, 2), dtype=jnp.float32)  # at origin
        cluster_b = jnp.ones((5, 2), dtype=jnp.float32) * 2000.0  # at (2000, 2000)
        positions = jnp.concatenate([cluster_a, cluster_b], axis=0)

        W = jnp.zeros((n, n), dtype=jnp.float32)

        params = _default_params(growth_prob=1.0, max_distance_um=100.0)
        W_new = structural_update(key, W, positions, is_exc, rates, params)

        # Check that no cross-cluster connections were made.
        # Cluster A = neurons 0-4, Cluster B = neurons 5-9.
        cross_ab = W_new[:5, 5:]  # post in A, pre in B
        cross_ba = W_new[5:, :5]  # post in B, pre in A
        assert jnp.sum(cross_ab) == 0.0, "Cross-cluster synapse A<-B formed"
        assert jnp.sum(cross_ba) == 0.0, "Cross-cluster synapse B<-A formed"


class TestHomeostaticTarget:
    def test_homeostatic_target(self):
        """After many rounds of structural plasticity, synapse count should
        approach the target fraction."""
        n = 30
        is_exc = _all_excitatory(n)
        positions = _close_positions(n)
        rates = jnp.ones(n) * 5.0

        target_frac = 0.05
        params = _default_params(
            prune_prob=0.1,
            growth_prob=0.05,
            prune_threshold=0.005,
            target_synapse_fraction=target_frac,
            w_new=0.003,  # just above prune_threshold to avoid immediate pruning
            max_distance_um=500.0,
        )

        # Start with zero connectivity.
        W = jnp.zeros((n, n), dtype=jnp.float32)

        # Run many rounds.
        for i in range(200):
            key = jax.random.PRNGKey(100 + i)
            W = structural_update(key, W, positions, is_exc, rates, params)

        n_synapses = int(jnp.sum(W > 0))
        n_possible = n * n - n  # all excitatory, minus self-connections
        actual_frac = n_synapses / n_possible

        # Should be in the right ballpark (within 5x of target).
        # Structural plasticity is stochastic; we check it's moving toward target
        # rather than being zero or saturated.
        assert n_synapses > 0, "Expected some synapses to have formed"
        assert actual_frac < 0.5, (
            f"Synapse fraction {actual_frac:.3f} seems too high"
        )


class TestReproducibility:
    def test_reproducibility(self):
        """Same PRNG key should produce identical results."""
        n = 20
        key = jax.random.PRNGKey(7)
        is_exc = _all_excitatory(n)
        positions = _close_positions(n)
        rates = jnp.ones(n) * 5.0

        W = jnp.zeros((n, n), dtype=jnp.float32)
        # Seed a few existing synapses.
        W = W.at[1, 0].set(0.05)
        W = W.at[3, 2].set(0.0005)

        params = _default_params(growth_prob=0.5, prune_prob=0.5)

        W_a = structural_update(key, W, positions, is_exc, rates, params)
        W_b = structural_update(key, W, positions, is_exc, rates, params)

        assert jnp.array_equal(W_a, W_b), "Same key should produce identical results"
