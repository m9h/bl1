"""Smoke tests for the BL-1 performance benchmark script.

These tests verify that the benchmark functions import correctly and
can execute at a trivially small scale without errors.
"""

import jax
import jax.numpy as jnp
import pytest


def test_benchmark_imports():
    """Verify that benchmark entry points are importable."""
    from benchmarks.profile_scale import (
        benchmark_network_creation,
        benchmark_simulation,
        run_benchmarks,
    )


def test_benchmark_network_creation():
    """Verify network creation benchmark runs for 100 neurons."""
    from benchmarks.profile_scale import benchmark_network_creation

    key = jax.random.PRNGKey(0)
    positions, params, state, is_exc, W_exc, W_inh = benchmark_network_creation(key, 100)

    assert positions.shape == (100, 2)
    assert params.a.shape == (100,)
    assert state.v.shape == (100,)
    assert is_exc.shape == (100,)


def test_benchmark_small_network():
    """Verify benchmark simulation runs for 100 neurons, 100 steps."""
    from benchmarks.profile_scale import benchmark_network_creation, benchmark_simulation
    from bl1.core.synapses import create_synapse_state

    key = jax.random.PRNGKey(42)
    positions, params, state, is_exc, W_exc, W_inh = benchmark_network_creation(key, 100)
    syn_state = create_synapse_state(100)

    # Without STDP
    t_jit, t_sim, total_spikes = benchmark_simulation(
        params, state, syn_state, W_exc, W_inh,
        n_steps=100, dt=0.5, with_stdp=False,
    )
    assert t_jit > 0
    assert t_sim > 0
    assert total_spikes >= 0


def test_benchmark_small_network_with_stdp():
    """Verify benchmark simulation with STDP runs for 100 neurons, 50 steps."""
    from benchmarks.profile_scale import benchmark_network_creation, benchmark_simulation
    from bl1.core.synapses import create_synapse_state

    key = jax.random.PRNGKey(42)
    positions, params, state, is_exc, W_exc, W_inh = benchmark_network_creation(key, 100)
    syn_state = create_synapse_state(100)

    t_jit, t_sim, total_spikes = benchmark_simulation(
        params, state, syn_state, W_exc, W_inh,
        n_steps=50, dt=0.5, with_stdp=True, is_exc=is_exc,
    )
    assert t_jit > 0
    assert t_sim > 0
    assert total_spikes >= 0
