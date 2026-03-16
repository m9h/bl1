"""Tests for conductance-based synapse models (bl1.core.synapses)."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from bl1.core.synapses import (
    E_AMPA,
    E_GABA_A,
    TAU_AMPA,
    TAU_GABA_A,
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_synapse_state():
    """create_synapse_state should return zeros for both conductances."""
    ss = create_synapse_state(100)

    assert ss.g_ampa.shape == (100,)
    assert ss.g_gaba_a.shape == (100,)
    npt.assert_allclose(np.asarray(ss.g_ampa), 0.0)
    npt.assert_allclose(np.asarray(ss.g_gaba_a), 0.0)


def test_ampa_decay():
    """AMPA conductance with no spikes should decay exponentially."""
    n = 5
    g = jnp.ones(n)
    spikes = jnp.zeros(n, dtype=jnp.bool_)
    weights = jnp.zeros((n, n))
    dt = 0.5

    # One step
    g1 = ampa_step(g, spikes, weights, dt=dt)
    expected_one = float(jnp.exp(-dt / TAU_AMPA))  # exp(-0.5/2.0) ~ 0.7788
    npt.assert_allclose(np.asarray(g1), expected_one, atol=1e-5)

    # Many steps — should approach zero
    g_many = g
    for _ in range(200):
        g_many = ampa_step(g_many, spikes, weights, dt=dt)
    npt.assert_allclose(np.asarray(g_many), 0.0, atol=1e-10)


def test_gaba_a_decay():
    """GABA_A conductance with no spikes should decay exponentially."""
    n = 5
    g = jnp.ones(n)
    spikes = jnp.zeros(n, dtype=jnp.bool_)
    weights = jnp.zeros((n, n))
    dt = 0.5

    # One step
    g1 = gaba_a_step(g, spikes, weights, dt=dt)
    expected_one = float(jnp.exp(-dt / TAU_GABA_A))  # exp(-0.5/6.0) ~ 0.9200
    npt.assert_allclose(np.asarray(g1), expected_one, atol=1e-4)

    # Many steps — should approach zero
    g_many = g
    for _ in range(500):
        g_many = gaba_a_step(g_many, spikes, weights, dt=dt)
    npt.assert_allclose(np.asarray(g_many), 0.0, atol=1e-10)


def test_ampa_spike_response():
    """A presynaptic spike through a weight matrix should increase postsynaptic g."""
    # 2 neurons: neuron 1 -> neuron 0, weight 0.05
    W = jnp.array([[0.0, 0.05],
                    [0.0, 0.0]])
    g = jnp.zeros(2)
    spikes = jnp.array([False, True])  # neuron 1 spikes

    g_new = ampa_step(g, spikes, W, dt=0.5)

    # g_new[0] = 0 * decay + W[0,:] @ spikes = 0.05
    # g_new[1] = 0 * decay + W[1,:] @ spikes = 0.0
    npt.assert_allclose(float(g_new[0]), 0.05, atol=1e-6)
    npt.assert_allclose(float(g_new[1]), 0.0, atol=1e-6)


def test_synaptic_current_excitatory():
    """AMPA current at rest should be positive (depolarising)."""
    v = jnp.array([-65.0])
    ss = SynapseState(g_ampa=jnp.array([1.0]), g_gaba_a=jnp.array([0.0]))

    I_syn = compute_synaptic_current(ss, v)

    # I = 1.0 * (0 - (-65)) = 65.0
    expected = 1.0 * (E_AMPA - (-65.0))
    npt.assert_allclose(float(I_syn[0]), expected, atol=1e-5)
    assert float(I_syn[0]) > 0, "Excitatory AMPA current should be positive"


def test_synaptic_current_inhibitory():
    """GABA_A current at rest should be negative (hyperpolarising)."""
    v = jnp.array([-65.0])
    ss = SynapseState(g_ampa=jnp.array([0.0]), g_gaba_a=jnp.array([1.0]))

    I_syn = compute_synaptic_current(ss, v)

    # I = 1.0 * (-75 - (-65)) = -10.0
    expected = 1.0 * (E_GABA_A - (-65.0))
    npt.assert_allclose(float(I_syn[0]), expected, atol=1e-5)
    assert float(I_syn[0]) < 0, "Inhibitory GABA_A current should be negative"


def test_synaptic_current_at_reversal():
    """At the reversal potential, driving force (and therefore current) should be ~0."""
    # AMPA at reversal (v = E_AMPA = 0)
    v_ampa = jnp.array([E_AMPA])
    ss_ampa = SynapseState(g_ampa=jnp.array([1.0]), g_gaba_a=jnp.array([0.0]))
    I_ampa = compute_synaptic_current(ss_ampa, v_ampa)
    npt.assert_allclose(float(I_ampa[0]), 0.0, atol=1e-7)

    # GABA_A at reversal (v = E_GABA_A = -75)
    v_gaba = jnp.array([E_GABA_A])
    ss_gaba = SynapseState(g_ampa=jnp.array([0.0]), g_gaba_a=jnp.array([1.0]))
    I_gaba = compute_synaptic_current(ss_gaba, v_gaba)
    npt.assert_allclose(float(I_gaba[0]), 0.0, atol=1e-7)
