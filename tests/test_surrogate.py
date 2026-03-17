"""Tests for surrogate gradients, differentiable neuron steps, and regularization.

Covers:
- Forward-pass correctness (surrogate produces same binary spikes as hard threshold)
- Non-zero gradients through spike thresholds
- Differentiable Izhikevich step produces spikes and non-zero gradients
- Firing rate regularization penalties
- parameter_sensitivity with surrogate gradients for spike-count metrics
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    V_PEAK,
    V_REST,
    izhikevich_step,
    izhikevich_step_surrogate,
)
from bl1.core.surrogate import (
    superspike_threshold,
    sigmoid_threshold,
    fast_sigmoid_threshold,
    arctan_threshold,
)
from bl1.core.regularization import (
    firing_rate_penalty,
    sparsity_penalty,
    silence_penalty,
)
from bl1.analysis.sensitivity import (
    parameter_sensitivity,
    mean_firing_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neurons(n: int, a=0.02, b=0.2, c=-65.0, d=8.0):
    """Create a small population of identical RS neurons at rest."""
    params = IzhikevichParams(
        a=jnp.full(n, a),
        b=jnp.full(n, b),
        c=jnp.full(n, c),
        d=jnp.full(n, d),
    )
    state = NeuronState(
        v=jnp.full(n, V_REST),
        u=jnp.full(n, b * V_REST),
        spikes=jnp.zeros(n, dtype=jnp.bool_),
    )
    return params, state


# ---------------------------------------------------------------------------
# Surrogate gradient forward-pass tests
# ---------------------------------------------------------------------------

class TestSurrogateForward:
    """Forward pass of surrogate thresholds must match the hard threshold."""

    def test_superspike_forward_matches_hard(self):
        """SuperSpike forward pass produces same binary spikes as v >= threshold."""
        v = jnp.array([-70.0, -10.0, 0.0, 29.9, 30.0, 30.1, 50.0])
        threshold = 30.0
        hard = (v >= threshold).astype(jnp.float32)
        surrogate = superspike_threshold(v, threshold, beta=10.0)
        np.testing.assert_array_equal(np.asarray(hard), np.asarray(surrogate))

    def test_sigmoid_forward_matches_hard(self):
        """Sigmoid surrogate forward pass matches hard threshold."""
        v = jnp.array([-70.0, 0.0, 29.999, 30.0, 30.001, 100.0])
        threshold = 30.0
        hard = (v >= threshold).astype(jnp.float32)
        surrogate = sigmoid_threshold(v, threshold, beta=5.0)
        np.testing.assert_array_equal(np.asarray(hard), np.asarray(surrogate))

    def test_fast_sigmoid_forward_matches_hard(self):
        """Fast sigmoid surrogate forward pass matches hard threshold."""
        v = jnp.array([-100.0, 29.5, 30.0, 30.5])
        threshold = 30.0
        hard = (v >= threshold).astype(jnp.float32)
        surrogate = fast_sigmoid_threshold(v, threshold, beta=10.0)
        np.testing.assert_array_equal(np.asarray(hard), np.asarray(surrogate))

    def test_arctan_forward_matches_hard(self):
        """ArcTan surrogate forward pass matches hard threshold."""
        v = jnp.array([-50.0, 20.0, 30.0, 40.0])
        threshold = 30.0
        hard = (v >= threshold).astype(jnp.float32)
        surrogate = arctan_threshold(v, threshold, beta=10.0)
        np.testing.assert_array_equal(np.asarray(hard), np.asarray(surrogate))


# ---------------------------------------------------------------------------
# Surrogate gradient backward-pass tests
# ---------------------------------------------------------------------------

class TestSurrogateGradient:
    """Backward pass must produce non-zero gradients near threshold."""

    def test_superspike_gradient_nonzero(self):
        """jax.grad through SuperSpike is nonzero near threshold."""
        def spike_sum(v):
            return jnp.sum(superspike_threshold(v, 30.0, beta=10.0))

        v = jnp.array([29.5, 30.0, 30.5])
        grad = jax.grad(spike_sum)(v)
        grad_np = np.asarray(grad)
        # At least one gradient should be nonzero (near threshold)
        assert np.any(np.abs(grad_np) > 1e-6), (
            f"SuperSpike gradients are all zero: {grad_np}"
        )

    def test_sigmoid_gradient_nonzero(self):
        """jax.grad through sigmoid surrogate is nonzero near threshold."""
        def spike_sum(v):
            return jnp.sum(sigmoid_threshold(v, 30.0, beta=5.0))

        v = jnp.array([29.0, 30.0, 31.0])
        grad = jax.grad(spike_sum)(v)
        grad_np = np.asarray(grad)
        assert np.any(np.abs(grad_np) > 1e-6), (
            f"Sigmoid gradients are all zero: {grad_np}"
        )

    def test_fast_sigmoid_gradient_nonzero(self):
        """jax.grad through fast sigmoid is nonzero near threshold."""
        def spike_sum(v):
            return jnp.sum(fast_sigmoid_threshold(v, 30.0, beta=10.0))

        v = jnp.array([29.8, 30.0, 30.2])
        grad = jax.grad(spike_sum)(v)
        grad_np = np.asarray(grad)
        assert np.any(np.abs(grad_np) > 1e-6), (
            f"Fast sigmoid gradients are all zero: {grad_np}"
        )

    def test_arctan_gradient_nonzero(self):
        """jax.grad through arctan surrogate is nonzero near threshold."""
        def spike_sum(v):
            return jnp.sum(arctan_threshold(v, 30.0, beta=10.0))

        v = jnp.array([29.8, 30.0, 30.2])
        grad = jax.grad(spike_sum)(v)
        grad_np = np.asarray(grad)
        assert np.any(np.abs(grad_np) > 1e-6), (
            f"ArcTan gradients are all zero: {grad_np}"
        )

    def test_superspike_gradient_peaks_at_threshold(self):
        """Surrogate gradient should be maximal at the threshold."""
        def spike_sum(v):
            return jnp.sum(superspike_threshold(v, 30.0, beta=10.0))

        v = jnp.array([25.0, 28.0, 30.0, 32.0, 35.0])
        grad = jax.grad(spike_sum)(v)
        grad_np = np.asarray(grad)
        # Gradient at threshold (index 2) should be the largest
        assert np.argmax(np.abs(grad_np)) == 2, (
            f"Gradient not maximal at threshold: {grad_np}"
        )


# ---------------------------------------------------------------------------
# Izhikevich surrogate step tests
# ---------------------------------------------------------------------------

class TestIzhikevichSurrogate:

    def test_izhikevich_surrogate_spikes(self):
        """Surrogate step produces spikes like the regular step for driven neuron."""
        params, state = _make_neurons(1)
        I_ext = jnp.array([15.0])

        # Run regular step
        spike_count_regular = 0
        s = state
        for _ in range(400):
            s = izhikevich_step(s, params, I_ext, dt=0.5)
            spike_count_regular += int(s.spikes[0])

        # Run surrogate step
        spike_count_surrogate = 0
        s = state
        for _ in range(400):
            s = izhikevich_step_surrogate(
                s, params, I_ext, dt=0.5,
                surrogate_fn=superspike_threshold, beta=10.0,
            )
            spike_count_surrogate += int(s.spikes[0])

        # Both should spike (exact count may differ slightly due to soft reset)
        assert spike_count_regular > 0, "Regular step produced no spikes"
        assert spike_count_surrogate > 0, "Surrogate step produced no spikes"
        # Spike counts should be in same ballpark
        assert abs(spike_count_regular - spike_count_surrogate) <= 5, (
            f"Spike counts diverged: regular={spike_count_regular}, "
            f"surrogate={spike_count_surrogate}"
        )

    def test_gradient_through_izhikevich(self):
        """jax.grad of mean_firing_rate through izhikevich_step_surrogate is nonzero."""
        N, T = 5, 200
        params, state = _make_neurons(N)
        I_ext = jnp.full((T, N), 15.0)

        # Convert initial state spikes to float32 to match surrogate step output
        init_state = NeuronState(
            v=state.v, u=state.u,
            spikes=state.spikes.astype(jnp.float32),
        )

        def simulate_rate(params):
            def scan_fn(carry, I_t):
                s = izhikevich_step_surrogate(
                    carry, params, I_t, dt=0.5,
                    surrogate_fn=superspike_threshold, beta=10.0,
                )
                return s, s.spikes

            _, spikes = jax.lax.scan(scan_fn, init_state, I_ext)
            return mean_firing_rate(spikes, dt_ms=0.5)

        grads = jax.grad(simulate_rate)(params)

        # At least one parameter field should have nonzero gradients
        any_nonzero = False
        for field in ("a", "b", "c", "d"):
            g = np.asarray(getattr(grads, field))
            if not np.allclose(g, 0.0, atol=1e-10):
                any_nonzero = True
                break

        assert any_nonzero, (
            f"All gradients are zero through surrogate Izhikevich step -- "
            f"a: {np.asarray(grads.a)}, b: {np.asarray(grads.b)}, "
            f"c: {np.asarray(grads.c)}, d: {np.asarray(grads.d)}"
        )

    def test_surrogate_no_spike_no_reset(self):
        """Below-threshold input: surrogate step should not spike or reset."""
        params, state = _make_neurons(1)
        I_ext = jnp.array([0.0])

        s = state
        for _ in range(100):
            s = izhikevich_step_surrogate(
                s, params, I_ext, dt=0.5,
                surrogate_fn=superspike_threshold, beta=10.0,
            )
            assert not bool(s.spikes[0]), "Neuron spiked with zero input"


# ---------------------------------------------------------------------------
# Regularization tests
# ---------------------------------------------------------------------------

class TestRegularization:

    def test_firing_rate_penalty_zero_at_target(self):
        """Penalty is 0 when firing rate equals target."""
        # 1000 steps at dt=0.5ms => 500ms total
        # target 10 Hz => 5 spikes in 500ms => 5/1000 = 0.005 spikes/step
        T, N = 1000, 10
        target_rate = 10.0
        dt_ms = 0.5

        # Exact target: rate = spike_mean * (1000/dt)
        # spike_mean = target_rate * dt / 1000 = 10 * 0.5 / 1000 = 0.005
        spike_mean = target_rate * dt_ms / 1000.0
        spike_data = jnp.full((T, N), spike_mean)

        penalty = firing_rate_penalty(spike_data, target_rate_hz=target_rate, dt_ms=dt_ms)
        assert float(penalty) < 1e-6, f"Penalty should be ~0 at target rate, got {float(penalty)}"

    def test_firing_rate_penalty_positive_off_target(self):
        """Penalty is positive when rate differs from target."""
        T, N = 1000, 10
        # All neurons spike every step => rate = 1.0 * 1000/0.5 = 2000 Hz
        spike_data = jnp.ones((T, N))
        penalty = firing_rate_penalty(spike_data, target_rate_hz=5.0, dt_ms=0.5)
        assert float(penalty) > 0.0, f"Penalty should be > 0 off target, got {float(penalty)}"

    def test_firing_rate_penalty_huber(self):
        """Huber penalty should work and be finite."""
        T, N = 1000, 5
        spike_data = jnp.ones((T, N)) * 0.1
        penalty = firing_rate_penalty(spike_data, target_rate_hz=5.0, dt_ms=0.5, penalty_type="huber")
        assert jnp.isfinite(penalty), f"Huber penalty is not finite: {penalty}"
        assert float(penalty) > 0.0

    def test_sparsity_penalty_zero_below_max(self):
        """Penalty is 0 when all rates are below max."""
        T, N = 1000, 10
        # rate = 0.005 * 2000 = 10 Hz (below max of 20)
        spike_data = jnp.full((T, N), 0.005)
        penalty = sparsity_penalty(spike_data, max_rate_hz=20.0, dt_ms=0.5)
        assert float(penalty) < 1e-6, f"Sparsity penalty should be ~0 below max, got {float(penalty)}"

    def test_sparsity_penalty_positive_above_max(self):
        """Penalty is positive when rates exceed max."""
        T, N = 1000, 10
        # All ones => 2000 Hz, way above 20
        spike_data = jnp.ones((T, N))
        penalty = sparsity_penalty(spike_data, max_rate_hz=20.0, dt_ms=0.5)
        assert float(penalty) > 0.0, f"Sparsity penalty should be > 0 above max, got {float(penalty)}"

    def test_silence_penalty_zero_above_min(self):
        """Penalty is 0 when all rates are above min."""
        T, N = 1000, 10
        # rate = 0.01 * 2000 = 20 Hz (above min of 0.5)
        spike_data = jnp.full((T, N), 0.01)
        penalty = silence_penalty(spike_data, min_rate_hz=0.5, dt_ms=0.5)
        assert float(penalty) < 1e-6, f"Silence penalty should be ~0 above min, got {float(penalty)}"

    def test_silence_penalty_positive_below_min(self):
        """Penalty is positive when rates are below min."""
        T, N = 1000, 10
        spike_data = jnp.zeros((T, N))  # 0 Hz
        penalty = silence_penalty(spike_data, min_rate_hz=0.5, dt_ms=0.5)
        assert float(penalty) > 0.0, f"Silence penalty should be > 0 below min, got {float(penalty)}"

    def test_regularization_differentiable(self):
        """All penalty functions should be differentiable."""
        T, N = 100, 5
        spike_data = jnp.ones((T, N)) * 0.01

        # firing_rate_penalty
        grad1 = jax.grad(lambda s: firing_rate_penalty(s, target_rate_hz=5.0))(spike_data)
        assert jnp.all(jnp.isfinite(grad1)), "firing_rate_penalty gradient not finite"

        # sparsity_penalty
        grad2 = jax.grad(lambda s: sparsity_penalty(s, max_rate_hz=20.0))(spike_data)
        assert jnp.all(jnp.isfinite(grad2)), "sparsity_penalty gradient not finite"

        # silence_penalty
        grad3 = jax.grad(lambda s: silence_penalty(s, min_rate_hz=0.5))(spike_data)
        assert jnp.all(jnp.isfinite(grad3)), "silence_penalty gradient not finite"


# ---------------------------------------------------------------------------
# Integration: sensitivity with surrogate
# ---------------------------------------------------------------------------

class TestSensitivityWithSurrogate:

    def test_sensitivity_with_surrogate(self):
        """parameter_sensitivity with surrogate produces nonzero gradients for spike-count metrics.

        This is THE key test: previously, parameter_sensitivity used the hard threshold
        izhikevich_step, so jax.grad of mean_firing_rate (a spike-count metric) returned
        zero gradients. With the surrogate, gradients flow through the threshold.
        """
        N, T = 10, 300
        key = jax.random.PRNGKey(42)
        I_ext = 10.0 + 5.0 * jax.random.normal(key, shape=(T, N))

        params, state = _make_neurons(N)
        a_vals = jnp.linspace(0.015, 0.025, N)
        params = params._replace(a=a_vals)

        grads = parameter_sensitivity(
            mean_firing_rate, params, state, I_ext,
            dt=0.5, n_steps=T,
        )

        # Check that at least one parameter field has non-zero gradients
        any_nonzero = False
        for field_name in ("a", "b", "c", "d"):
            grad_field = np.asarray(getattr(grads, field_name))
            if not np.allclose(grad_field, 0.0, atol=1e-10):
                any_nonzero = True
                break

        assert any_nonzero, (
            "All gradients are zero for spike-count metric with surrogate -- "
            f"a: {np.asarray(grads.a)}, b: {np.asarray(grads.b)}, "
            f"c: {np.asarray(grads.c)}, d: {np.asarray(grads.d)}"
        )

    def test_sensitivity_shape_preserved(self):
        """parameter_sensitivity with surrogate returns same-shaped gradients."""
        N, T = 5, 100
        params, state = _make_neurons(N)
        I_ext = jnp.full((T, N), 15.0)

        grads = parameter_sensitivity(
            mean_firing_rate, params, state, I_ext,
            dt=0.5, n_steps=T,
        )

        assert isinstance(grads, IzhikevichParams)
        for field_name in ("a", "b", "c", "d"):
            grad_field = getattr(grads, field_name)
            param_field = getattr(params, field_name)
            assert grad_field.shape == param_field.shape, (
                f"Shape mismatch for '{field_name}': {grad_field.shape} vs {param_field.shape}"
            )
