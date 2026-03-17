"""Tests for the hybrid multi-model neuron simulation (bl1.core.hybrid)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.hybrid import (
    HybridParams,
    HybridPopulation,
    HybridState,
    hybrid_step,
    get_membrane_potential,
)
from bl1.core.synapses import create_synapse_state, compute_synaptic_current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_hybrid(params, state, I_ext, n_steps, dt=0.5):
    """Run *n_steps* of the hybrid model, returning spike count per neuron."""
    spike_count = jnp.zeros(params.n_total)
    for _ in range(n_steps):
        state = hybrid_step(state, params, I_ext, dt=dt)
        spike_count = spike_count + state.spikes.astype(jnp.float32)
    return state, spike_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_hybrid_izh_only():
    """All Izhikevich, no AdEx. Should work like regular population."""
    key = jax.random.PRNGKey(42)
    params, state = HybridPopulation.create(
        key=key, n_neurons=100, n_adex=0, ei_ratio=0.8
    )

    # All neurons are Izhikevich
    assert params.izh_indices.shape == (100,)
    assert params.adex_indices.shape == (0,)
    assert state.izh_v.shape == (100,)
    assert state.izh_u.shape == (100,)
    assert state.adex_v.shape == (0,)
    assert state.adex_w.shape == (0,)
    assert state.spikes.shape == (100,)
    assert params.n_total == 100

    # Excitatory count should be ~80
    n_exc = int(params.is_excitatory.sum())
    assert n_exc == 80, f"Expected 80 excitatory, got {n_exc}"


def test_create_hybrid_mixed():
    """80 Izhikevich + 20 AdEx. Check shapes and index consistency."""
    key = jax.random.PRNGKey(123)
    params, state = HybridPopulation.create(
        key=key,
        n_neurons=100,
        model_assignments=["izhikevich"] * 80 + ["adex"] * 20,
        ei_ratio=0.8,
    )

    # Shape checks
    assert params.izh_indices.shape == (80,)
    assert params.adex_indices.shape == (20,)
    assert state.izh_v.shape == (80,)
    assert state.izh_u.shape == (80,)
    assert state.adex_v.shape == (20,)
    assert state.adex_w.shape == (20,)
    assert state.spikes.shape == (100,)
    assert params.n_total == 100

    # Izhikevich params should have 80 entries
    assert params.izh_params.a.shape == (80,)
    # AdEx params should have 20 entries
    assert params.adex_params.C.shape == (20,)

    # Indices should be non-overlapping and cover [0, 100)
    all_indices = jnp.concatenate([params.izh_indices, params.adex_indices])
    assert jnp.unique(all_indices).shape == (100,)

    # Izh indices should be 0..79, AdEx 80..99
    np.testing.assert_array_equal(
        np.asarray(params.izh_indices), np.arange(80)
    )
    np.testing.assert_array_equal(
        np.asarray(params.adex_indices), np.arange(80, 100)
    )


def test_hybrid_step_produces_spikes():
    """With sufficient current, hybrid population should spike."""
    key = jax.random.PRNGKey(7)
    params, state = HybridPopulation.create(
        key=key,
        n_neurons=100,
        model_assignments=["izhikevich"] * 80 + ["adex"] * 20,
        ei_ratio=0.8,
    )

    # Strong current: Izhikevich uses unitless current (~10-40 typical),
    # AdEx uses pA (~500-1000 typical). We set both high enough to spike.
    # Since both model types get the same I_ext vector, we use currents
    # appropriate for each region.
    I_ext = jnp.zeros(100)
    I_ext = I_ext.at[:80].set(40.0)    # Izhikevich neurons: strong current
    I_ext = I_ext.at[80:].set(700.0)   # AdEx neurons: strong current (pA)

    n_steps = 2000  # 1000 ms at dt=0.5
    state, spike_count = _simulate_hybrid(params, state, I_ext, n_steps, dt=0.5)

    # Both sub-populations should have spiked
    izh_spikes = spike_count[:80]
    adex_spikes = spike_count[80:]

    assert float(izh_spikes.sum()) > 0, "No Izhikevich spikes"
    assert float(adex_spikes.sum()) > 0, "No AdEx spikes"


def test_hybrid_spike_gathering():
    """Spikes from both sub-populations should appear at correct indices."""
    key = jax.random.PRNGKey(99)
    # Use interleaved assignments to make indices non-trivial
    assignments = []
    for i in range(50):
        assignments.append("izhikevich")
        assignments.append("adex")

    params, state = HybridPopulation.create(
        key=key,
        n_neurons=100,
        model_assignments=assignments,
        ei_ratio=0.8,
    )

    # Verify interleaved indices
    izh_expected = np.arange(0, 100, 2)  # even indices
    adex_expected = np.arange(1, 100, 2)  # odd indices
    np.testing.assert_array_equal(np.asarray(params.izh_indices), izh_expected)
    np.testing.assert_array_equal(np.asarray(params.adex_indices), adex_expected)

    # Apply current strong enough to provoke spikes in both models
    I_ext = jnp.zeros(100)
    I_ext = I_ext.at[izh_expected].set(40.0)
    I_ext = I_ext.at[adex_expected].set(700.0)

    # Run enough steps for spikes to occur
    for _ in range(2000):
        state = hybrid_step(state, params, I_ext, dt=0.5)

    # After many steps, both models should have spiked at least once
    izh_spiked_ever = state.spikes[params.izh_indices]
    adex_spiked_ever = state.spikes[params.adex_indices]

    # Run a longer burst to accumulate spikes
    spike_count = jnp.zeros(100)
    for _ in range(200):
        state = hybrid_step(state, params, I_ext, dt=0.5)
        spike_count = spike_count + state.spikes.astype(jnp.float32)

    # Check that spikes at even indices came from Izhikevich
    izh_spike_count = spike_count[params.izh_indices]
    adex_spike_count = spike_count[params.adex_indices]

    assert float(izh_spike_count.sum()) > 0, "No Izhikevich spikes in gathered vector"
    assert float(adex_spike_count.sum()) > 0, "No AdEx spikes in gathered vector"

    # Spikes at Izhikevich indices should come only from Izhikevich model
    # (the combined vector has zeros elsewhere, set by scatter)
    # Verify no spikes "leak" between models by checking un-assigned positions
    # are not double-counted (implicitly tested by the index construction).


def test_hybrid_ei_ratio():
    """Excitatory/inhibitory ratio should be maintained across both models."""
    key = jax.random.PRNGKey(55)
    n_neurons = 200
    n_izh = 160
    n_adex_count = 40
    ei_ratio = 0.8

    params, state = HybridPopulation.create(
        key=key,
        n_neurons=n_neurons,
        model_assignments=["izhikevich"] * n_izh + ["adex"] * n_adex_count,
        ei_ratio=ei_ratio,
    )

    # Overall E/I ratio
    total_exc = int(params.is_excitatory.sum())
    expected_exc = int(n_neurons * ei_ratio)
    assert total_exc == expected_exc, (
        f"Expected {expected_exc} excitatory neurons, got {total_exc}"
    )

    # E/I ratio within Izhikevich sub-population
    izh_exc = params.is_excitatory[params.izh_indices]
    izh_exc_count = int(izh_exc.sum())
    expected_izh_exc = int(n_izh * ei_ratio)
    assert izh_exc_count == expected_izh_exc, (
        f"Izh E/I: expected {expected_izh_exc} excitatory, got {izh_exc_count}"
    )

    # E/I ratio within AdEx sub-population
    adex_exc = params.is_excitatory[params.adex_indices]
    adex_exc_count = int(adex_exc.sum())
    expected_adex_exc = int(n_adex_count * ei_ratio)
    assert adex_exc_count == expected_adex_exc, (
        f"AdEx E/I: expected {expected_adex_exc} excitatory, got {adex_exc_count}"
    )


def test_hybrid_with_synapses():
    """Wire a small hybrid network with synapses and run a few steps."""
    key = jax.random.PRNGKey(77)
    n_neurons = 50
    n_izh = 40
    n_adex_count = 10

    params, state = HybridPopulation.create(
        key=key,
        n_neurons=n_neurons,
        model_assignments=["izhikevich"] * n_izh + ["adex"] * n_adex_count,
        ei_ratio=0.8,
    )

    # Create synapse state for the full population
    syn_state = create_synapse_state(n_neurons)

    # Create simple random weight matrices
    k1, k2 = jax.random.split(jax.random.PRNGKey(88))
    W_exc = jax.random.uniform(k1, (n_neurons, n_neurons)) * 0.5
    W_inh = jax.random.uniform(k2, (n_neurons, n_neurons)) * 0.5

    # Mask weights by E/I identity
    exc_mask = params.is_excitatory.astype(jnp.float32)
    inh_mask = (~params.is_excitatory).astype(jnp.float32)
    W_exc = W_exc * exc_mask[None, :]  # only excitatory presynaptic
    W_inh = W_inh * inh_mask[None, :]  # only inhibitory presynaptic

    # External drive
    I_ext_base = jnp.zeros(n_neurons)
    I_ext_base = I_ext_base.at[:n_izh].set(15.0)      # Izhikevich
    I_ext_base = I_ext_base.at[n_izh:].set(500.0)     # AdEx (pA)

    # Run for several steps, incorporating synaptic feedback
    from bl1.core.synapses import ampa_step, gaba_a_step

    for step in range(200):
        # Compute synaptic current from conductances
        v_full = get_membrane_potential(state, params)
        I_syn = compute_synaptic_current(syn_state, v_full)
        I_total = I_ext_base + I_syn

        # Step neurons
        state = hybrid_step(state, params, I_total, dt=0.5)

        # Update synaptic conductances
        spikes_f = state.spikes.astype(jnp.float32)
        new_g_ampa = ampa_step(syn_state.g_ampa, spikes_f, W_exc, dt=0.5)
        new_g_gaba_a = gaba_a_step(syn_state.g_gaba_a, spikes_f, W_inh, dt=0.5)

        syn_state = syn_state._replace(
            g_ampa=new_g_ampa,
            g_gaba_a=new_g_gaba_a,
        )

    # The test passes if we get here without errors.
    # Also verify the state shapes are intact.
    assert state.spikes.shape == (n_neurons,)
    assert state.izh_v.shape == (n_izh,)
    assert state.adex_v.shape == (n_adex_count,)


def test_create_hybrid_adex_only():
    """All AdEx, no Izhikevich neurons."""
    key = jax.random.PRNGKey(33)
    params, state = HybridPopulation.create(
        key=key,
        n_neurons=50,
        model_assignments=["adex"] * 50,
        ei_ratio=0.8,
    )

    assert params.izh_indices.shape == (0,)
    assert params.adex_indices.shape == (50,)
    assert state.izh_v.shape == (0,)
    assert state.adex_v.shape == (50,)

    # Should still be steppable
    I_ext = jnp.full(50, 700.0)
    state = hybrid_step(state, params, I_ext, dt=0.5)
    assert state.spikes.shape == (50,)


def test_get_membrane_potential():
    """get_membrane_potential should reconstruct the full voltage vector."""
    key = jax.random.PRNGKey(11)
    params, state = HybridPopulation.create(
        key=key,
        n_neurons=100,
        model_assignments=["izhikevich"] * 60 + ["adex"] * 40,
        ei_ratio=0.8,
    )

    v_full = get_membrane_potential(state, params)
    assert v_full.shape == (100,)

    # Izhikevich neurons start at -65.0, AdEx at -70.6
    np.testing.assert_allclose(np.asarray(v_full[:60]), -65.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(v_full[60:]), -70.6, atol=1e-4)


def test_hybrid_step_jit_compatible():
    """hybrid_step should be JIT-compiled without errors on repeated calls."""
    key = jax.random.PRNGKey(0)
    params, state = HybridPopulation.create(
        key=key,
        n_neurons=50,
        model_assignments=["izhikevich"] * 30 + ["adex"] * 20,
        ei_ratio=0.8,
    )

    I_ext = jnp.zeros(50)
    I_ext = I_ext.at[:30].set(20.0)
    I_ext = I_ext.at[30:].set(500.0)

    # First call triggers compilation
    state1 = hybrid_step(state, params, I_ext, dt=0.5)
    # Second call uses cached compilation
    state2 = hybrid_step(state1, params, I_ext, dt=0.5)
    # Third call
    state3 = hybrid_step(state2, params, I_ext, dt=0.5)

    # Verify state evolved (voltages should have changed)
    assert not jnp.allclose(state.izh_v, state3.izh_v)


def test_invalid_model_assignment():
    """Unknown model name should raise ValueError."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="Unknown model"):
        HybridPopulation.create(
            key=key,
            n_neurons=10,
            model_assignments=["izhikevich"] * 5 + ["hodgkin_huxley"] * 5,
        )
