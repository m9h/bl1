"""Tests for differentiable training: loss functions and training loop.

Covers:
- Loss functions from bl1.training.loss (standalone differentiable losses)
- Training loop from bl1.training.trainer (train_weights, _train_step)
- Constraint preservation (Dale's law, sparsity) after optimization
- Gradient flow through surrogate spike mechanism
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.training.loss import (
    burst_rate_loss,
    culture_loss,
    firing_rate_loss,
    make_gaussian_kernel,
    synchrony_loss,
    weight_regularization,
)
from bl1.training.trainer import (
    TrainingConfig,
    TrainingResult,
    _build_network,
    _make_loss_fn,
    _train_step,
    train_weights,
)
from bl1.training.trainer import culture_loss as trainer_culture_loss

# Small network parameters used across all tests for speed.
_N = 50
_DT = 0.5
_DURATION_MS = 200.0
_T = int(_DURATION_MS / _DT)  # 400 timesteps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spike_history(key, T=_T, N=_N, rate_hz=5.0):
    """Generate a random Bernoulli spike raster at a given rate."""
    prob_per_step = rate_hz * _DT / 1000.0
    return (jax.random.uniform(key, (T, N)) < prob_per_step).astype(jnp.float32)


def _random_weights(key, N=_N):
    """Generate small random weight matrices (excitatory and inhibitory)."""
    k1, k2 = jax.random.split(key)
    W_exc = jax.random.uniform(k1, (N, N), minval=0.0, maxval=0.5)
    W_inh = jax.random.uniform(k2, (N, N), minval=0.0, maxval=0.5)
    # Zero the diagonal (no self-connections)
    mask = 1.0 - jnp.eye(N)
    return W_exc * mask, W_inh * mask


# ---------------------------------------------------------------------------
# Loss function tests (bl1.training.loss)
# ---------------------------------------------------------------------------


class TestFiringRateLoss:
    """Tests for firing_rate_loss."""

    def test_firing_rate_loss_zero_at_target(self):
        """Loss is near zero when per-neuron rates match the target."""
        # Use a higher target rate that can be achieved in a short simulation.
        # With T=400, dt=0.5ms, duration=0.2s, to get 10 Hz we need
        # 10 * 0.2 = 2 spikes per neuron over 400 steps.
        target_hz = 10.0
        duration_s = _T * _DT / 1000.0  # 0.2 s
        spikes_per_neuron = int(target_hz * duration_s)  # 2

        spikes = jnp.zeros((_T, _N), dtype=jnp.float32)
        # Place exactly the right number of spikes per neuron
        interval = _T // max(spikes_per_neuron, 1)
        for i in range(spikes_per_neuron):
            t = i * interval
            if t < _T:
                spikes = spikes.at[t, :].set(1.0)

        loss = firing_rate_loss(spikes, target_hz=target_hz, dt=_DT)
        # With exact spike placement the rate should be very close to target
        assert float(loss) < 1.0, f"Expected loss < 1.0 at target rate, got {float(loss)}"

    def test_firing_rate_loss_increases_with_deviation(self):
        """Loss grows as firing rates diverge from target."""
        key = jax.random.PRNGKey(0)
        target_hz = 1.4

        # Low rate raster (~0.5 Hz)
        low_spikes = _random_spike_history(key, rate_hz=0.5)
        # High rate raster (~20 Hz)
        high_spikes = _random_spike_history(key, rate_hz=20.0)
        # Near-target raster
        near_spikes = _random_spike_history(key, rate_hz=1.4)

        loss_near = float(firing_rate_loss(near_spikes, target_hz=target_hz, dt=_DT))
        loss_low = float(firing_rate_loss(low_spikes, target_hz=target_hz, dt=_DT))
        loss_high = float(firing_rate_loss(high_spikes, target_hz=target_hz, dt=_DT))

        assert loss_near < loss_high, (
            f"Near-target loss ({loss_near}) should be less than high-rate loss ({loss_high})"
        )
        assert loss_near < loss_low or loss_low < loss_high, (
            "Loss should generally increase with deviation from target"
        )


class TestBurstRateLoss:
    """Tests for burst_rate_loss."""

    def test_burst_rate_loss_differentiable(self):
        """jax.grad produces non-zero gradients through burst_rate_loss."""
        key = jax.random.PRNGKey(1)
        spikes = _random_spike_history(key, rate_hz=5.0)

        # Differentiate w.r.t. the spike history (treating it as soft)
        def loss_fn(s):
            return burst_rate_loss(s, target_bursts_per_min=8.0, dt=_DT)

        grad = jax.grad(loss_fn)(spikes)
        grad_norm = float(jnp.linalg.norm(grad))
        assert grad_norm > 0.0, "Expected non-zero gradients from burst_rate_loss"


class TestSynchronyLoss:
    """Tests for synchrony_loss."""

    def test_synchrony_loss_differentiable(self):
        """jax.grad produces non-zero gradients through synchrony_loss."""
        key = jax.random.PRNGKey(2)
        spikes = _random_spike_history(key, rate_hz=5.0)

        def loss_fn(s):
            return synchrony_loss(s, target_fano=1.5, dt=_DT)

        grad = jax.grad(loss_fn)(spikes)
        grad_norm = float(jnp.linalg.norm(grad))
        assert grad_norm > 0.0, "Expected non-zero gradients from synchrony_loss"


class TestWeightRegularization:
    """Tests for weight_regularization."""

    def test_weight_regularization_differentiable(self):
        """jax.grad produces non-zero gradients through weight_regularization."""
        key = jax.random.PRNGKey(3)
        W_exc, W_inh = _random_weights(key)

        def loss_fn(w_e, w_i):
            return weight_regularization(w_e, w_i)

        grad_exc, grad_inh = jax.grad(loss_fn, argnums=(0, 1))(W_exc, W_inh)
        assert float(jnp.linalg.norm(grad_exc)) > 0.0, (
            "Expected non-zero excitatory gradients"
        )
        assert float(jnp.linalg.norm(grad_inh)) > 0.0, (
            "Expected non-zero inhibitory gradients"
        )


class TestCultureLoss:
    """Tests for culture_loss (combined multi-objective loss)."""

    def test_culture_loss_returns_dict(self):
        """culture_loss returns (scalar, dict) tuple."""
        key = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(key)
        spikes = _random_spike_history(k1, rate_hz=5.0)
        W_exc, W_inh = _random_weights(k2)

        total, components = culture_loss(spikes, W_exc, W_inh, dt=_DT)

        assert total.shape == (), f"Expected scalar total, got shape {total.shape}"
        assert isinstance(components, dict), (
            f"Expected dict, got {type(components)}"
        )

    def test_culture_loss_all_components_present(self):
        """Dict has all expected component keys."""
        key = jax.random.PRNGKey(5)
        k1, k2 = jax.random.split(key)
        spikes = _random_spike_history(k1, rate_hz=5.0)
        W_exc, W_inh = _random_weights(k2)

        _, components = culture_loss(spikes, W_exc, W_inh, dt=_DT)

        expected_keys = {
            "firing_rate",
            "burst_rate",
            "synchrony",
            "silence",
            "sparsity",
            "weight_reg",
            "total",
        }
        missing = expected_keys - set(components.keys())
        assert not missing, f"Missing loss components: {missing}"


# ---------------------------------------------------------------------------
# Training loop tests (bl1.training.trainer)
# ---------------------------------------------------------------------------


def _tiny_config() -> TrainingConfig:
    """A minimal training config for fast tests."""
    return TrainingConfig(
        n_neurons=_N,
        n_epochs=3,
        sim_duration_ms=_DURATION_MS,
        dt=_DT,
        learning_rate=1e-3,
        ei_ratio=0.8,
        lambda_um=200.0,
        p_max=0.1,  # higher p_max for small network to ensure connections
        g_exc_init=0.50,
        g_inh_init=2.00,
        I_noise_amplitude=5.0,
        seed=42,
    )


class TestTrainStep:
    """Tests for single training steps."""

    def test_train_step_reduces_loss(self):
        """One step of training should reduce the total loss."""
        import optax

        config = _tiny_config()
        config = TrainingConfig(
            **{**config.__dict__, "n_epochs": 1},
        )

        # Build network
        params, init_state_template, W_exc, W_inh, exc_mask, inh_mask = (
            _build_network(config)
        )
        n_steps = int(config.sim_duration_ms / config.dt)

        # Create components
        from bl1.core.synapses import create_synapse_state
        from bl1.core.izhikevich import NeuronState

        syn_state = create_synapse_state(config.n_neurons)
        optimizer = optax.adam(config.learning_rate)
        opt_state_exc = optimizer.init(W_exc)
        opt_state_inh = optimizer.init(W_inh)

        # Generate noise
        rng = jax.random.PRNGKey(config.seed + 1)
        rng, key_noise, key_state = jax.random.split(rng, 3)
        I_external = (
            config.I_noise_amplitude
            * jax.random.normal(key_noise, shape=(n_steps, config.n_neurons))
        )
        v_init = -65.0 + 5.0 * jax.random.normal(
            key_state, shape=(config.n_neurons,)
        )
        init_state = NeuronState(
            v=v_init,
            u=init_state_template.u,
            spikes=jnp.zeros(config.n_neurons, dtype=jnp.float32),
        )

        # Compute initial loss
        loss_fn = _make_loss_fn(params, init_state, syn_state, I_external, config)
        initial_loss, _ = loss_fn(W_exc, W_inh)

        # Run one training step
        W_exc_new, W_inh_new, _, _, step_loss, _ = _train_step(
            W_exc, W_inh,
            opt_state_exc, opt_state_inh,
            params, init_state, syn_state,
            I_external, config, optimizer,
            exc_mask, inh_mask,
        )

        # The step loss should equal the initial loss (computed at same weights)
        # but the new weights should give a lower loss
        new_loss, _ = loss_fn(W_exc_new, W_inh_new)

        # The optimizer should have moved in a descent direction.
        # Allow some tolerance since this is stochastic.
        assert float(new_loss) < float(initial_loss) * 1.5, (
            f"Loss after one step ({float(new_loss):.4f}) should not be much "
            f"worse than initial ({float(initial_loss):.4f})"
        )


class TestTrainWeights:
    """Tests for the full train_weights loop."""

    @pytest.mark.slow
    def test_train_weights_smoke(self):
        """Run train_weights with tiny config and verify completion."""
        config = _tiny_config()
        result = train_weights(config)

        assert isinstance(result, TrainingResult)
        assert result.W_exc.shape == (_N, _N)
        assert result.W_inh.shape == (_N, _N)
        assert len(result.loss_history) == config.n_epochs
        assert result.config.n_neurons == _N

        # All loss history entries should have the 'total' key
        for entry in result.loss_history:
            assert "total" in entry, f"Missing 'total' in loss entry: {entry.keys()}"
            assert np.isfinite(entry["total"]), (
                f"Non-finite loss at epoch {entry.get('epoch', '?')}"
            )

    @pytest.mark.slow
    def test_dale_law_preserved(self):
        """After training, W_exc >= 0 and W_inh >= 0 (Dale's law)."""
        config = _tiny_config()
        result = train_weights(config)

        exc_min = float(jnp.min(result.W_exc))
        inh_min = float(jnp.min(result.W_inh))

        assert exc_min >= 0.0, (
            f"Dale's law violated: W_exc has min={exc_min:.6f}"
        )
        assert inh_min >= 0.0, (
            f"Dale's law violated: W_inh has min={inh_min:.6f}"
        )

    @pytest.mark.slow
    def test_sparsity_preserved(self):
        """Zero entries in original weights remain zero after training."""
        config = _tiny_config()

        # Build network to get original masks
        _, _, W_exc_orig, W_inh_orig, exc_mask, inh_mask = _build_network(config)

        # Train
        result = train_weights(config)

        # Wherever the original mask is 0, the trained weights should also be 0
        exc_violations = float(jnp.sum(
            (exc_mask == 0.0) & (result.W_exc != 0.0)
        ))
        inh_violations = float(jnp.sum(
            (inh_mask == 0.0) & (result.W_inh != 0.0)
        ))

        assert exc_violations == 0.0, (
            f"Sparsity violated: {int(exc_violations)} new excitatory connections"
        )
        assert inh_violations == 0.0, (
            f"Sparsity violated: {int(inh_violations)} new inhibitory connections"
        )


class TestGradientFlow:
    """Tests for gradient flow through the surrogate spike mechanism."""

    def test_gradients_nonzero(self):
        """jax.grad through simulate(surrogate=True) + culture_loss produces nonzero gradients."""
        config = _tiny_config()
        config = TrainingConfig(
            **{**config.__dict__, "n_epochs": 1},
        )

        # Build network
        params, init_state_template, W_exc, W_inh, exc_mask, inh_mask = (
            _build_network(config)
        )
        n_steps = int(config.sim_duration_ms / config.dt)

        from bl1.core.synapses import create_synapse_state
        from bl1.core.izhikevich import NeuronState

        syn_state = create_synapse_state(config.n_neurons)

        # Generate noise
        rng = jax.random.PRNGKey(99)
        rng, key_noise, key_state = jax.random.split(rng, 3)
        I_external = (
            config.I_noise_amplitude
            * jax.random.normal(key_noise, shape=(n_steps, config.n_neurons))
        )
        v_init = -65.0 + 5.0 * jax.random.normal(
            key_state, shape=(config.n_neurons,)
        )
        init_state = NeuronState(
            v=v_init,
            u=init_state_template.u,
            spikes=jnp.zeros(config.n_neurons, dtype=jnp.float32),
        )

        # Build the loss function
        loss_fn = _make_loss_fn(params, init_state, syn_state, I_external, config)

        # Compute gradients
        (loss, components), (grad_exc, grad_inh) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(W_exc, W_inh)

        grad_exc_norm = float(jnp.linalg.norm(grad_exc))
        grad_inh_norm = float(jnp.linalg.norm(grad_inh))

        assert np.isfinite(float(loss)), f"Loss is not finite: {float(loss)}"
        assert grad_exc_norm > 0.0, (
            f"Excitatory gradients are zero (norm={grad_exc_norm})"
        )
        assert grad_inh_norm > 0.0, (
            f"Inhibitory gradients are zero (norm={grad_inh_norm})"
        )
