"""Tests for STP integration into the simulation loop.

Verifies that:
1. simulate() with stp_params runs without error.
2. STP reduces excitatory firing under sustained stimulation.
3. simulate() without stp_params produces identical results (backward compat).
4. STP modulates AMPA conductance (lower after repeated activity).
5. STP works with the delay path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.integrator import simulate
from bl1.core.izhikevich import create_population
from bl1.core.synapses import create_synapse_state
from bl1.plasticity.stp import STPParams, create_stp_params, init_stp_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_network(key, n_neurons=50, ei_ratio=0.8):
    """Create a small deterministic network for testing."""
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


def _make_external_current(key, n_neurons, T, amplitude=8.0):
    """Generate external Poisson-like drive current."""
    return amplitude * jax.random.bernoulli(
        key, p=0.02, shape=(T, n_neurons)
    ).astype(jnp.float32)


# ---------------------------------------------------------------------------
# 1. simulate with STP runs without error
# ---------------------------------------------------------------------------

def test_simulate_with_stp():
    """simulate() with stp_params should run and return valid results."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    N = 50
    T = 200

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(k1, N)
    I_ext = _make_external_current(k2, N, T)

    stp_p = create_stp_params(N, is_exc)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=stp_p,
    )

    assert result.spike_history.shape == (T, N)
    assert result.final_neuron_state.v.shape == (N,)
    assert result.final_syn_state.g_ampa.shape == (N,)

    total_spikes = int(jnp.sum(result.spike_history))
    assert total_spikes > 0, "Expected at least some spikes with external drive."


# ---------------------------------------------------------------------------
# 2. STP reduces excitatory firing under sustained stimulation
# ---------------------------------------------------------------------------

def test_stp_reduces_excitatory_firing():
    """With STP (depressing excitatory synapses), repeated stimulation
    should produce fewer total spikes than without STP."""
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)
    N = 80
    T = 600  # 300 ms -- enough time for depression to accumulate

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(k1, N)

    # Strong sustained drive to trigger recurrent activity
    I_ext = _make_external_current(k2, N, T, amplitude=12.0)

    # Without STP
    result_no_stp = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
    )
    spikes_no_stp = int(jnp.sum(result_no_stp.spike_history))

    # With STP
    stp_p = create_stp_params(N, is_exc)
    result_stp = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=stp_p,
    )
    spikes_stp = int(jnp.sum(result_stp.spike_history))

    # STP depression should reduce total spikes (excitatory synapses
    # depress, reducing recurrent amplification)
    assert spikes_stp <= spikes_no_stp, (
        f"STP should reduce or maintain spike count, but got "
        f"STP={spikes_stp} > no-STP={spikes_no_stp}"
    )


# ---------------------------------------------------------------------------
# 3. backward compatibility: no stp_params gives identical results
# ---------------------------------------------------------------------------

def test_stp_backward_compatible():
    """simulate() without stp_params should give identical results to the
    original code path."""
    key = jax.random.PRNGKey(456)
    k1, k2 = jax.random.split(key)
    N = 60
    T = 100

    params, init_state, syn_state, W_exc, W_inh, _ = _make_small_network(k1, N)
    I_ext = _make_external_current(k2, N, T)

    # Run without stp_params (default)
    result_default = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
    )

    # Run explicitly passing None
    result_none = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=None,
    )

    # Spike histories should be bit-identical
    np.testing.assert_array_equal(
        result_default.spike_history,
        result_none.spike_history,
    )

    # Membrane potentials should be identical
    np.testing.assert_allclose(
        result_default.final_neuron_state.v,
        result_none.final_neuron_state.v,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. STP modulates conductance
# ---------------------------------------------------------------------------

def test_stp_modulates_conductance():
    """After a burst of activity, AMPA conductance should be lower with STP
    than without (because excitatory synapses are depressed)."""
    key = jax.random.PRNGKey(789)
    k1, k2 = jax.random.split(key)
    N = 50
    T = 400  # 200 ms

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(k1, N)
    I_ext = _make_external_current(k2, N, T, amplitude=15.0)

    # Without STP
    result_no_stp = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
    )

    # With STP
    stp_p = create_stp_params(N, is_exc)
    result_stp = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=stp_p,
    )

    # AMPA conductance should be lower (or equal) with STP
    # because excitatory depression reduces effective spike amplitude
    g_ampa_no_stp = float(jnp.sum(result_no_stp.final_syn_state.g_ampa))
    g_ampa_stp = float(jnp.sum(result_stp.final_syn_state.g_ampa))

    assert g_ampa_stp <= g_ampa_no_stp, (
        f"Expected STP to reduce AMPA conductance: "
        f"STP={g_ampa_stp:.6f} > no-STP={g_ampa_no_stp:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. STP with delays runs without error
# ---------------------------------------------------------------------------

def test_simulate_stp_with_delays():
    """simulate() with both stp_params and delay matrices should work."""
    key = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 50
    T = 200

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(k1, N)
    I_ext = _make_external_current(k2, N, T)

    # Random delays (1-5 timesteps for connected pairs)
    conn_mask = (W_exc + W_inh) > 0
    delay_vals = jax.random.randint(k3, (N, N), 1, 6)
    delays = jnp.where(conn_mask, delay_vals, 0).astype(jnp.int32)

    stp_p = create_stp_params(N, is_exc)

    result = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        W_exc_delays=delays,
        W_inh_delays=delays,
        stp_params=stp_p,
    )

    assert result.spike_history.shape == (T, N)
    total_spikes = int(jnp.sum(result.spike_history))
    assert total_spikes > 0, "Expected at least some spikes."


# ---------------------------------------------------------------------------
# 6. STP with delays vs STP without delays differ
# ---------------------------------------------------------------------------

def test_stp_delayed_vs_instant_different():
    """STP + delays should produce different results from STP alone."""
    key = jax.random.PRNGKey(77)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 60
    T = 400

    params, init_state, syn_state, W_exc, W_inh, is_exc = _make_small_network(k1, N)
    I_ext = _make_external_current(k2, N, T, amplitude=10.0)

    conn_mask = (W_exc + W_inh) > 0
    delay_vals = jax.random.randint(k3, (N, N), 2, 8)
    delays = jnp.where(conn_mask, delay_vals, 0).astype(jnp.int32)

    stp_p = create_stp_params(N, is_exc)

    # STP without delays
    result_no_delay = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        stp_params=stp_p,
    )

    # STP with delays
    result_delayed = simulate(
        params, init_state, syn_state, None,
        W_exc, W_inh, I_ext, dt=0.5,
        W_exc_delays=delays,
        W_inh_delays=delays,
        stp_params=stp_p,
    )

    v_no_delay = np.asarray(result_no_delay.final_neuron_state.v)
    v_delayed = np.asarray(result_delayed.final_neuron_state.v)
    assert not np.allclose(v_no_delay, v_delayed, atol=1e-3), (
        "Expected STP+delays to produce different final voltages than STP alone."
    )
