"""Tests for homeostatic synaptic scaling (bl1.plasticity.homeostatic)."""

import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from bl1.plasticity.homeostatic import (
    HomeostaticParams,
    HomeostaticState,
    init_homeostatic_state,
    update_rate_estimate,
    homeostatic_scaling,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_params() -> HomeostaticParams:
    return HomeostaticParams()  # uses class defaults


def _all_excitatory(n: int) -> jnp.ndarray:
    return jnp.ones(n, dtype=jnp.bool_)


def _no_spikes(n: int) -> jnp.ndarray:
    return jnp.zeros(n, dtype=jnp.bool_)


def _all_spikes(n: int) -> jnp.ndarray:
    return jnp.ones(n, dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# Tests: init_homeostatic_state
# ---------------------------------------------------------------------------

class TestInitHomeostaticState:
    def test_init_homeostatic_state(self):
        """init_homeostatic_state(100) returns rate_estimate of shape (100,)
        filled with the initial rate."""
        state = init_homeostatic_state(100)
        assert state.rate_estimate.shape == (100,)
        assert jnp.allclose(state.rate_estimate, 5.0)

    def test_init_custom_rate(self):
        """Custom initial_rate should be reflected in the state."""
        state = init_homeostatic_state(10, initial_rate=2.0)
        assert jnp.allclose(state.rate_estimate, 2.0)


# ---------------------------------------------------------------------------
# Tests: rate estimation
# ---------------------------------------------------------------------------

class TestRateEstimateIncreasesWithSpikes:
    def test_rate_estimate_increases_with_spikes(self):
        """After many timesteps with spikes, rate estimate should increase
        above the initial value."""
        n = 10
        state = init_homeostatic_state(n, initial_rate=5.0)
        spikes = _all_spikes(n)

        # Run 2000 steps (1000 ms at dt=0.5) with every neuron firing
        for _ in range(2000):
            state = update_rate_estimate(state, spikes, dt_ms=0.5)

        # instantaneous rate when firing every step = 1000/0.5 = 2000 Hz
        # rate estimate should have risen well above 5 Hz
        assert jnp.all(state.rate_estimate > 5.0), (
            f"Expected rate > 5.0 after sustained spiking, got {state.rate_estimate}"
        )


class TestRateEstimateDecreasesWithoutSpikes:
    def test_rate_estimate_decreases_without_spikes(self):
        """After many timesteps without spikes, rate estimate should
        decrease below the initial value."""
        n = 10
        state = init_homeostatic_state(n, initial_rate=5.0)
        spikes = _no_spikes(n)

        # Use a shorter tau to get meaningful decay in fewer steps
        tau = 500.0  # 500 ms time constant

        # Run 2000 steps (= 1000 ms at dt=0.5) with no spikes.
        # Analytical: rate = 5.0 * exp(-1000/500) = 5.0 * exp(-2) ~ 0.677
        for _ in range(2000):
            state = update_rate_estimate(state, spikes, dt_ms=0.5, tau_rate_ms=tau)

        expected = 5.0 * jnp.exp(-1000.0 / tau)
        assert jnp.allclose(state.rate_estimate, expected, rtol=1e-3), (
            f"Expected rate ~ {expected:.3f}, got {state.rate_estimate[0]:.3f}"
        )
        assert jnp.all(state.rate_estimate < 1.0), (
            f"Expected rate < 1.0 after silence, got {state.rate_estimate}"
        )


# ---------------------------------------------------------------------------
# Tests: homeostatic scaling direction
# ---------------------------------------------------------------------------

class TestScalingUpward:
    def test_scaling_upward(self):
        """When rate < target, weights should increase."""
        n = 4
        params = _default_params()
        # All neurons firing below target
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 1.0, dtype=jnp.float32),  # well below r_target=5
        )
        is_exc = _all_excitatory(n)

        # Create a weight matrix with some E->E connections
        W = jnp.zeros((n, n), dtype=jnp.float32)
        W = W.at[0, 1].set(0.05)
        W = W.at[1, 2].set(0.05)
        W = W.at[2, 3].set(0.05)

        _, W_new = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        # All existing connections should have increased
        assert W_new[0, 1] > 0.05, f"Expected upscaling, got {W_new[0, 1]}"
        assert W_new[1, 2] > 0.05, f"Expected upscaling, got {W_new[1, 2]}"
        assert W_new[2, 3] > 0.05, f"Expected upscaling, got {W_new[2, 3]}"


class TestScalingDownward:
    def test_scaling_downward(self):
        """When rate > target, weights should decrease."""
        n = 4
        params = _default_params()
        # All neurons firing above target
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 20.0, dtype=jnp.float32),  # above r_target=5
        )
        is_exc = _all_excitatory(n)

        W = jnp.zeros((n, n), dtype=jnp.float32)
        W = W.at[0, 1].set(0.05)
        W = W.at[1, 2].set(0.05)
        W = W.at[2, 3].set(0.05)

        _, W_new = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        # All existing connections should have decreased
        assert W_new[0, 1] < 0.05, f"Expected downscaling, got {W_new[0, 1]}"
        assert W_new[1, 2] < 0.05, f"Expected downscaling, got {W_new[1, 2]}"
        assert W_new[2, 3] < 0.05, f"Expected downscaling, got {W_new[2, 3]}"


# ---------------------------------------------------------------------------
# Tests: weight bounds
# ---------------------------------------------------------------------------

class TestWeightBounds:
    def test_weight_bounds_upper(self):
        """Repeated upscaling should not push weights above w_max."""
        n = 2
        params = _default_params()
        # Very low rate -> strong upscaling pressure
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 0.0, dtype=jnp.float32),
        )
        is_exc = _all_excitatory(n)

        W = jnp.zeros((n, n), dtype=jnp.float32).at[0, 1].set(0.19)

        # Apply scaling many times
        for _ in range(10000):
            _, W = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        assert W[0, 1] <= params.w_max + 1e-7, (
            f"Weight {W[0, 1]} exceeded w_max {params.w_max}"
        )

    def test_weight_bounds_lower(self):
        """Repeated downscaling should not push weights below w_min."""
        n = 2
        params = _default_params()
        # Very high rate -> strong downscaling pressure
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 1000.0, dtype=jnp.float32),
        )
        is_exc = _all_excitatory(n)

        W = jnp.zeros((n, n), dtype=jnp.float32).at[0, 1].set(0.002)

        # Apply scaling many times
        for _ in range(10000):
            _, W = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        assert W[0, 1] >= params.w_min - 1e-7, (
            f"Weight {W[0, 1]} went below w_min {params.w_min}"
        )


# ---------------------------------------------------------------------------
# Tests: E->E only
# ---------------------------------------------------------------------------

class TestOnlyEEConnectionsScale:
    def test_only_ee_connections_scale(self):
        """E->I, I->E, and I->I connections should not be modified."""
        n = 4
        params = _default_params()
        # Low rate to drive upscaling
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 0.0, dtype=jnp.float32),
        )

        # Neurons 0, 1 are excitatory; 2, 3 are inhibitory
        is_exc = jnp.array([True, True, False, False])

        # Set up one connection of each type:
        # E->E: W[0, 1] = 0.05  (post=0 exc, pre=1 exc)
        # E->I: W[2, 1] = 0.05  (post=2 inh, pre=1 exc)
        # I->E: W[0, 2] = 0.05  (post=0 exc, pre=2 inh)
        # I->I: W[2, 3] = 0.05  (post=2 inh, pre=3 inh)
        W = jnp.zeros((n, n), dtype=jnp.float32)
        W = W.at[0, 1].set(0.05)  # E->E
        W = W.at[2, 1].set(0.05)  # E->I
        W = W.at[0, 2].set(0.05)  # I->E
        W = W.at[2, 3].set(0.05)  # I->I

        _, W_new = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        # Only E->E should change
        assert W_new[0, 1] != 0.05, "E->E weight should have changed"
        assert W_new[0, 1] > 0.05, "E->E weight should have increased (rate < target)"

        # All other connection types should be unchanged
        assert jnp.isclose(W_new[2, 1], 0.05, atol=1e-7), (
            f"E->I weight changed from 0.05 to {W_new[2, 1]}"
        )
        assert jnp.isclose(W_new[0, 2], 0.05, atol=1e-7), (
            f"I->E weight changed from 0.05 to {W_new[0, 2]}"
        )
        assert jnp.isclose(W_new[2, 3], 0.05, atol=1e-7), (
            f"I->I weight changed from 0.05 to {W_new[2, 3]}"
        )


# ---------------------------------------------------------------------------
# Tests: zero connections preserved
# ---------------------------------------------------------------------------

class TestZeroConnectionsPreserved:
    def test_zero_connections_preserved(self):
        """Zero entries in W (absent connections) should stay zero."""
        n = 4
        params = _default_params()
        # Low rate to drive upscaling pressure
        state = HomeostaticState(
            rate_estimate=jnp.full(n, 0.0, dtype=jnp.float32),
        )
        is_exc = _all_excitatory(n)

        # Only one connection is non-zero
        W = jnp.zeros((n, n), dtype=jnp.float32).at[0, 1].set(0.05)

        _, W_new = homeostatic_scaling(state, params, W, is_exc, dt_ms=1.0)

        # The non-zero connection should have changed
        assert W_new[0, 1] > 0.0

        # All other entries should remain exactly zero
        W_check = W_new.at[0, 1].set(0.0)
        assert jnp.allclose(W_check, 0.0), (
            "Zero connections were modified by homeostatic scaling"
        )


# ---------------------------------------------------------------------------
# Tests: sparse (BCOO) path
# ---------------------------------------------------------------------------

class TestSparseSupport:
    def test_sparse_scaling_matches_dense(self):
        """BCOO sparse path should produce the same result as the dense path."""
        n = 6
        params = _default_params()
        state = HomeostaticState(
            rate_estimate=jnp.array([1.0, 2.0, 8.0, 10.0, 5.0, 3.0],
                                    dtype=jnp.float32),
        )
        is_exc = jnp.array([True, True, True, True, False, True])

        # Build a dense weight matrix with a handful of E->E connections
        W_dense = jnp.zeros((n, n), dtype=jnp.float32)
        W_dense = W_dense.at[0, 1].set(0.05)
        W_dense = W_dense.at[1, 3].set(0.08)
        W_dense = W_dense.at[2, 0].set(0.03)
        W_dense = W_dense.at[3, 5].set(0.06)
        # Include an I->E connection that should not scale
        W_dense = W_dense.at[0, 4].set(0.04)

        # Dense result
        _, W_dense_new = homeostatic_scaling(
            state, params, W_dense, is_exc, dt_ms=1.0,
        )

        # Sparse result
        W_sparse = BCOO.fromdense(W_dense)
        _, W_sparse_new = homeostatic_scaling(
            state, params, W_sparse, is_exc, dt_ms=1.0,
        )

        # Materialise sparse result for comparison
        W_sparse_dense = W_sparse_new.todense()

        assert jnp.allclose(W_dense_new, W_sparse_dense, atol=1e-6), (
            "Sparse and dense homeostatic scaling produced different results"
        )
