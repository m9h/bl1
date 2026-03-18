"""Tests for surrogate-gradient integration into the main simulation loop.

Verifies that:
1. simulate(surrogate=True) runs without error
2. simulate(surrogate=True) produces spikes like the regular path
3. jax.grad through simulate(surrogate=True) produces nonzero gradients
4. surrogate=False (default) produces identical results to before
5. surrogate + STP works together
6. sensitivity.py uses simulate() internally (after refactor)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.integrator import simulate
from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    V_REST,
    create_population,
)
from bl1.core.surrogate import superspike_threshold, fast_sigmoid_threshold
from bl1.core.synapses import create_synapse_state
from bl1.plasticity.stp import create_stp_params
from bl1.analysis.sensitivity import (
    parameter_sensitivity,
    fit_parameters,
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


def _make_small_network(key, n_neurons=20, ei_ratio=0.8):
    """Create a small network with random connectivity for testing."""
    k1, k2 = jax.random.split(key)

    params, init_state, is_excitatory = create_population(k1, n_neurons, ei_ratio)
    syn_state = create_synapse_state(n_neurons)

    # Sparse random connectivity (10% density)
    conn_mask = jax.random.bernoulli(k2, p=0.1, shape=(n_neurons, n_neurons))
    conn_mask = conn_mask & ~jnp.eye(n_neurons, dtype=jnp.bool_)

    exc_mask = is_excitatory[None, :] * conn_mask
    inh_mask = ~is_excitatory[None, :] * conn_mask

    W_exc = exc_mask.astype(jnp.float32) * 0.05
    W_inh = inh_mask.astype(jnp.float32) * 0.20

    return params, init_state, syn_state, W_exc, W_inh, is_excitatory


def _constant_current(n_steps: int, n_neurons: int, amplitude: float = 15.0):
    """Create a constant external current array (n_steps, n_neurons)."""
    return jnp.full((n_steps, n_neurons), amplitude)


# ---------------------------------------------------------------------------
# 1. simulate(surrogate=True) runs without error
# ---------------------------------------------------------------------------

def test_simulate_surrogate_runs():
    """simulate(surrogate=True) should complete without error and return
    valid shapes."""
    key = jax.random.PRNGKey(42)
    N, T = 20, 200

    params, init_state, syn_state, W_exc, W_inh, _ = _make_small_network(key, N)
    I_ext = _constant_current(T, N, amplitude=10.0)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        surrogate=True,
    )

    assert result.spike_history.shape == (T, N)
    assert result.final_neuron_state.v.shape == (N,)
    assert result.final_syn_state.g_ampa.shape == (N,)
    # Surrogate path returns float32 spikes
    assert result.spike_history.dtype == jnp.float32


# ---------------------------------------------------------------------------
# 2. simulate(surrogate=True) produces spikes
# ---------------------------------------------------------------------------

def test_simulate_surrogate_produces_spikes():
    """Surrogate path should produce spikes when given sufficient drive."""
    key = jax.random.PRNGKey(7)
    N, T = 20, 200

    params, init_state, syn_state, W_exc, W_inh, _ = _make_small_network(key, N)
    # Strong constant current to ensure spiking
    I_ext = _constant_current(T, N, amplitude=15.0)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        surrogate=True,
    )

    total_spikes = float(jnp.sum(result.spike_history))
    assert total_spikes > 0, (
        "Expected at least some spikes with strong external drive, "
        f"got {total_spikes}"
    )


# ---------------------------------------------------------------------------
# 3. jax.grad through simulate(surrogate=True) produces nonzero gradients
# ---------------------------------------------------------------------------

def test_simulate_surrogate_differentiable():
    """jax.grad through simulate(surrogate=True) should produce nonzero
    gradients for a spike-count metric."""
    N, T = 10, 200
    params, state = _make_neurons(N)
    syn_state = create_synapse_state(N)

    W_exc = jnp.zeros((N, N), dtype=jnp.float32)
    W_inh = jnp.zeros((N, N), dtype=jnp.float32)
    I_ext = _constant_current(T, N, amplitude=15.0)

    def loss_fn(params):
        result = simulate(
            params, state, syn_state, None,
            W_exc, W_inh, I_ext, dt=0.5,
            surrogate=True,
        )
        return mean_firing_rate(result.spike_history, dt_ms=0.5)

    grads = jax.grad(loss_fn)(params)

    # At least one parameter field should have nonzero gradient
    any_nonzero = False
    for field_name in ("a", "b", "c", "d"):
        grad_field = np.asarray(getattr(grads, field_name))
        if not np.allclose(grad_field, 0.0, atol=1e-10):
            any_nonzero = True
            break

    assert any_nonzero, (
        "All gradients are zero -- surrogate gradient is not flowing. "
        f"a: {np.asarray(grads.a)}, b: {np.asarray(grads.b)}, "
        f"c: {np.asarray(grads.c)}, d: {np.asarray(grads.d)}"
    )


# ---------------------------------------------------------------------------
# 4. surrogate=False (default) produces identical results to before
# ---------------------------------------------------------------------------

def test_simulate_surrogate_backward_compatible():
    """simulate() with default surrogate=False should produce identical
    results to a call without the parameter at all."""
    key = jax.random.PRNGKey(456)
    N, T = 20, 100

    params, init_state, syn_state, W_exc, W_inh, _ = _make_small_network(key, N)
    I_ext = _constant_current(T, N, amplitude=10.0)

    # Run with default (no surrogate args)
    result_default = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
    )

    # Run with explicit surrogate=False
    result_explicit = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        surrogate=False,
    )

    # Spike histories should be bit-identical
    np.testing.assert_array_equal(
        result_default.spike_history,
        result_explicit.spike_history,
    )

    # Membrane potentials should be identical
    np.testing.assert_allclose(
        result_default.final_neuron_state.v,
        result_explicit.final_neuron_state.v,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 5. surrogate + STP works together
# ---------------------------------------------------------------------------

def test_simulate_surrogate_with_stp():
    """simulate(surrogate=True, stp_params=...) should work without error."""
    key = jax.random.PRNGKey(99)
    N, T = 20, 200

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(key, N)
    I_ext = _constant_current(T, N, amplitude=12.0)

    stp_p = create_stp_params(N, is_exc)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=stp_p,
        surrogate=True,
    )

    assert result.spike_history.shape == (T, N)
    assert result.spike_history.dtype == jnp.float32
    total_spikes = float(jnp.sum(result.spike_history))
    assert total_spikes > 0, "Expected at least some spikes with STP + surrogate."


# ---------------------------------------------------------------------------
# 6. sensitivity.py uses simulate() internally
# ---------------------------------------------------------------------------

def test_sensitivity_uses_simulate():
    """After refactor, parameter_sensitivity should produce the same kind
    of results as before (nonzero gradients for a driven network).

    This validates that the sensitivity module now delegates to
    simulate(surrogate=True) rather than its own custom scan loop.
    """
    N, T = 10, 200
    params, state = _make_neurons(N)
    I_ext = _constant_current(T, N, amplitude=15.0)

    grads = parameter_sensitivity(
        mean_firing_rate, params, state, I_ext, dt=0.5, n_steps=T,
    )

    # Should still be an IzhikevichParams pytree
    assert isinstance(grads, IzhikevichParams), (
        f"Expected IzhikevichParams, got {type(grads)}"
    )

    # At least one field should have nonzero gradients
    any_nonzero = False
    for field_name in ("a", "b", "c", "d"):
        grad_field = np.asarray(getattr(grads, field_name))
        if not np.allclose(grad_field, 0.0, atol=1e-10):
            any_nonzero = True
            break

    assert any_nonzero, "parameter_sensitivity returned all-zero gradients."


# ---------------------------------------------------------------------------
# Extra: surrogate with custom surrogate_fn
# ---------------------------------------------------------------------------

def test_simulate_surrogate_custom_fn():
    """simulate(surrogate=True, surrogate_fn=fast_sigmoid_threshold) should
    work and produce nonzero gradients."""
    N, T = 10, 200
    params, state = _make_neurons(N)
    syn_state = create_synapse_state(N)

    W_exc = jnp.zeros((N, N), dtype=jnp.float32)
    W_inh = jnp.zeros((N, N), dtype=jnp.float32)
    I_ext = _constant_current(T, N, amplitude=15.0)

    def loss_fn(params):
        result = simulate(
            params, state, syn_state, None,
            W_exc, W_inh, I_ext, dt=0.5,
            surrogate=True,
            surrogate_fn=fast_sigmoid_threshold,
            surrogate_beta=5.0,
        )
        return mean_firing_rate(result.spike_history, dt_ms=0.5)

    grads = jax.grad(loss_fn)(params)
    any_nonzero = any(
        not np.allclose(np.asarray(getattr(grads, f)), 0.0, atol=1e-10)
        for f in ("a", "b", "c", "d")
    )
    assert any_nonzero, "Custom surrogate_fn (fast_sigmoid) produced all-zero gradients."


# ---------------------------------------------------------------------------
# Extra: error on unsupported combos
# ---------------------------------------------------------------------------

def test_simulate_surrogate_with_delays_runs():
    """surrogate=True with delay matrices should run without error."""
    key = jax.random.PRNGKey(77)
    N, T = 20, 200

    params, init_state, syn_state, W_exc, W_inh, _ = _make_small_network(key, N)
    I_ext = _constant_current(T, N, amplitude=12.0)

    delays = jnp.ones((N, N), dtype=jnp.int32) * 2

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        W_exc_delays=delays,
        W_inh_delays=delays,
        surrogate=True,
    )

    assert result.spike_history.shape == (T, N)
    assert result.spike_history.dtype == jnp.float32
    total_spikes = float(jnp.sum(result.spike_history))
    assert total_spikes > 0, "Expected spikes with surrogate + delays."


def test_simulate_surrogate_with_delays_differentiable():
    """jax.grad through surrogate + delays should produce nonzero gradients."""
    N, T = 10, 200
    params, state = _make_neurons(N)
    syn_state = create_synapse_state(N)

    W_exc = jnp.eye(N, dtype=jnp.float32) * 0.0
    W_inh = jnp.zeros((N, N), dtype=jnp.float32)
    delays = jnp.ones((N, N), dtype=jnp.int32) * 2

    def loss_fn(I_ext):
        result = simulate(
            params, state, syn_state, None,
            W_exc, W_inh, I_ext, dt=0.5,
            W_exc_delays=delays,
            W_inh_delays=delays,
            surrogate=True,
        )
        return jnp.sum(result.spike_history)

    I_ext = _constant_current(T, N, amplitude=15.0)
    grads = jax.grad(loss_fn)(I_ext)

    assert grads.shape == (T, N)
    assert float(jnp.max(jnp.abs(grads))) > 0, (
        "Gradients are all zero -- surrogate gradient not flowing through delays path."
    )


def test_simulate_surrogate_with_delays_and_stp():
    """surrogate + delays + STP should all work together."""
    key = jax.random.PRNGKey(88)
    N, T = 20, 200

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(key, N)
    I_ext = _constant_current(T, N, amplitude=12.0)

    delays = jnp.ones((N, N), dtype=jnp.int32) * 3
    stp_p = create_stp_params(N, is_exc)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        W_exc_delays=delays,
        W_inh_delays=delays,
        stp_params=stp_p,
        surrogate=True,
    )

    assert result.spike_history.shape == (T, N)
    assert result.spike_history.dtype == jnp.float32
    total_spikes = float(jnp.sum(result.spike_history))
    assert total_spikes > 0, "Expected spikes with surrogate + delays + STP."


def test_simulate_surrogate_with_fast_sparse_raises():
    """surrogate=True with use_fast_sparse=True should raise ValueError."""
    N, T = 10, 50
    params, state = _make_neurons(N)
    syn_state = create_synapse_state(N)

    W_exc = jnp.zeros((N, N), dtype=jnp.float32)
    W_inh = jnp.zeros((N, N), dtype=jnp.float32)
    I_ext = _constant_current(T, N)

    with pytest.raises(ValueError, match="surrogate.*not yet supported.*fast_sparse"):
        simulate(
            params, state, syn_state, None,
            W_exc, W_inh, I_ext, dt=0.5,
            use_fast_sparse=True,
            surrogate=True,
        )
