"""Tests for Phase 2 synapse models: NMDA and GABA_B (bl1.core.synapses).

These tests verify the dual-exponential kinetics, Mg2+ voltage-dependent
block, and backward compatibility with the extended SynapseState.
"""

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from bl1.core.synapses import (
    E_AMPA,
    E_GABA_A,
    E_GABA_B,
    E_NMDA,
    MG_CONC,
    TAU_GABA_B_DECAY,
    TAU_GABA_B_RISE,
    TAU_NMDA_DECAY,
    TAU_NMDA_RISE,
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
    gaba_b_step,
    nmda_mg_block,
    nmda_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(n: int = 1) -> jnp.ndarray:
    return jnp.zeros(n)


def _make_zero_syn_state(n: int = 1) -> SynapseState:
    """Create a SynapseState with all fields zeroed."""
    return create_synapse_state(n)


def _simulate_nmda_pulse(n_steps: int, dt: float = 0.5, weight: float = 1.0):
    """Simulate NMDA response to a single spike at t=0.

    Returns arrays of time (ms) and g_nmda at each step.
    """
    n = 1
    W = jnp.array([[weight]])
    g_rise = jnp.zeros(n)
    g_decay = jnp.zeros(n)

    times = []
    conductances = []

    for step in range(n_steps):
        # Spike only at t=0
        spikes = jnp.array([1.0]) if step == 0 else jnp.array([0.0])
        g_rise, g_decay, g_nmda = nmda_step(g_rise, g_decay, spikes, W, dt)
        times.append(step * dt)
        conductances.append(float(g_nmda[0]))

    return np.array(times), np.array(conductances)


def _simulate_gaba_b_pulse(n_steps: int, dt: float = 0.5, weight: float = 1.0):
    """Simulate GABA_B response to a single spike at t=0.

    Returns arrays of time (ms) and g_gaba_b at each step.
    """
    n = 1
    W = jnp.array([[weight]])
    g_rise = jnp.zeros(n)
    g_decay = jnp.zeros(n)

    times = []
    conductances = []

    for step in range(n_steps):
        spikes = jnp.array([1.0]) if step == 0 else jnp.array([0.0])
        g_rise, g_decay, g_gaba_b = gaba_b_step(g_rise, g_decay, spikes, W, dt)
        times.append(step * dt)
        conductances.append(float(g_gaba_b[0]))

    return np.array(times), np.array(conductances)


# ---------------------------------------------------------------------------
# Test 1: NMDA Mg2+ block at resting potential
# ---------------------------------------------------------------------------

def test_nmda_mg_block_at_rest():
    """At v=-65 mV, Mg block should be ~0.08 (strongly blocked)."""
    v = jnp.array([-65.0])
    block = nmda_mg_block(v)
    val = float(block[0])

    # Analytical: 1 / (1 + 1.0/3.57 * exp(-0.062 * -65))
    expected = 1.0 / (1.0 + (MG_CONC / 3.57) * math.exp(-0.062 * (-65.0)))

    npt.assert_allclose(val, expected, atol=1e-5)
    # Should be in the range ~0.05 to ~0.15 (strongly blocked)
    assert 0.03 < val < 0.15, f"Mg block at rest should be small, got {val}"


# ---------------------------------------------------------------------------
# Test 2: NMDA Mg2+ block at depolarized potential
# ---------------------------------------------------------------------------

def test_nmda_mg_block_depolarized():
    """At v=0 mV, Mg block should be ~0.78 (mostly unblocked)."""
    v = jnp.array([0.0])
    block = nmda_mg_block(v)
    val = float(block[0])

    expected = 1.0 / (1.0 + (MG_CONC / 3.57) * math.exp(0.0))
    npt.assert_allclose(val, expected, atol=1e-5)
    # Should be ~0.78
    assert 0.7 < val < 0.85, f"Mg block at 0 mV should be ~0.78, got {val}"


# ---------------------------------------------------------------------------
# Test 3: Mg2+ block monotonically increasing with voltage
# ---------------------------------------------------------------------------

def test_nmda_mg_block_shape():
    """Block should be monotonically increasing with voltage from -80 to +20."""
    voltages = jnp.linspace(-80.0, 20.0, 200)
    block = nmda_mg_block(voltages)
    block_np = np.asarray(block)

    # Check monotonically increasing
    diffs = np.diff(block_np)
    assert np.all(diffs > 0), "Mg2+ block should be monotonically increasing with voltage"

    # Check bounds: all values in (0, 1)
    assert np.all(block_np > 0)
    assert np.all(block_np < 1)


# ---------------------------------------------------------------------------
# Test 4: NMDA dual-exponential shape
# ---------------------------------------------------------------------------

def test_nmda_dual_exponential_shape():
    """After a single spike, NMDA conductance should rise fast (2ms) then
    decay slowly (100ms). Peak should occur around t_peak and conductance
    at 200ms should still be significant (>10% of peak)."""
    dt = 0.5
    n_steps = 800  # 400 ms
    times, g_nmda = _simulate_nmda_pulse(n_steps, dt)

    # Find peak
    peak_idx = np.argmax(g_nmda)
    peak_time = times[peak_idx]
    peak_val = g_nmda[peak_idx]

    # Analytical t_peak for difference of exponentials
    t_peak_analytical = (TAU_NMDA_DECAY * TAU_NMDA_RISE
                         / (TAU_NMDA_DECAY - TAU_NMDA_RISE)
                         * math.log(TAU_NMDA_DECAY / TAU_NMDA_RISE))

    # Peak should be near analytical value (within a few ms due to discrete stepping)
    assert abs(peak_time - t_peak_analytical) < 2.0, (
        f"NMDA peak at {peak_time} ms, expected ~{t_peak_analytical:.1f} ms"
    )

    # Peak should be positive
    assert peak_val > 0, f"Peak NMDA conductance should be positive, got {peak_val}"

    # Conductance at 200ms should still be >10% of peak (slow decay)
    idx_200ms = int(200.0 / dt)
    g_at_200ms = g_nmda[idx_200ms]
    assert g_at_200ms > 0.10 * peak_val, (
        f"NMDA at 200ms ({g_at_200ms:.4f}) should be >10% of peak ({peak_val:.4f})"
    )

    # Conductance should eventually decay toward zero
    g_at_end = g_nmda[-1]
    assert g_at_end < 0.05 * peak_val, (
        f"NMDA at 400ms ({g_at_end:.4f}) should be <5% of peak ({peak_val:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 5: NMDA current is voltage-dependent
# ---------------------------------------------------------------------------

def test_nmda_current_voltage_dependent():
    """At rest (-65mV), NMDA current should be small (Mg block).
    At -20mV, should be much larger."""
    # Create a synapse state with some NMDA conductance
    g_val = 1.0
    ss = SynapseState(
        g_ampa=jnp.array([0.0]),
        g_gaba_a=jnp.array([0.0]),
        g_nmda_rise=jnp.array([0.0]),
        g_nmda_decay=jnp.array([g_val]),  # g_nmda = decay - rise = 1.0
        g_gaba_b_rise=jnp.array([0.0]),
        g_gaba_b_decay=jnp.array([0.0]),
    )

    # At rest: strongly blocked
    v_rest = jnp.array([-65.0])
    I_rest = compute_synaptic_current(ss, v_rest)

    # At -20 mV: less blocked, still has driving force
    v_depol = jnp.array([-20.0])
    I_depol = compute_synaptic_current(ss, v_depol)

    # Both should be positive (E_NMDA=0 is above both voltages)
    assert float(I_rest[0]) > 0, "NMDA current at -65mV should be positive"
    assert float(I_depol[0]) > 0, "NMDA current at -20mV should be positive"

    # Current at -20 mV should be much larger than at -65 mV due to Mg unblock
    # Even though driving force is smaller at -20, the Mg unblock effect dominates
    ratio = float(I_depol[0]) / float(I_rest[0])
    assert ratio > 2.0, (
        f"NMDA current at -20mV should be much larger than at -65mV, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 6: GABA_B slow kinetics
# ---------------------------------------------------------------------------

def test_gaba_b_slow_kinetics():
    """After spike, GABA_B peak should occur much later than GABA_A peak.
    GABA_B peak should be around 50-100ms (vs immediate for GABA_A).
    Simulate 500ms and check."""
    dt = 0.5
    n_steps = 1000  # 500 ms

    # GABA_B response
    times, g_gaba_b = _simulate_gaba_b_pulse(n_steps, dt)
    gaba_b_peak_idx = np.argmax(g_gaba_b)
    gaba_b_peak_time = times[gaba_b_peak_idx]

    # Analytical t_peak for GABA_B
    t_peak_analytical = (TAU_GABA_B_DECAY * TAU_GABA_B_RISE
                         / (TAU_GABA_B_DECAY - TAU_GABA_B_RISE)
                         * math.log(TAU_GABA_B_DECAY / TAU_GABA_B_RISE))

    # Peak should be near analytical value
    assert abs(gaba_b_peak_time - t_peak_analytical) < 5.0, (
        f"GABA_B peak at {gaba_b_peak_time} ms, expected ~{t_peak_analytical:.1f} ms"
    )

    # GABA_B peak should be much later than what GABA_A would be (~0 ms for single-exp)
    assert gaba_b_peak_time > 40.0, (
        f"GABA_B peak should be >40ms, got {gaba_b_peak_time} ms"
    )

    # GABA_B conductance should still be significant at 200 ms
    idx_200ms = int(200.0 / dt)
    g_at_200ms = g_gaba_b[idx_200ms]
    peak_val = g_gaba_b[gaba_b_peak_idx]
    assert g_at_200ms > 0.2 * peak_val, (
        f"GABA_B at 200ms should still be >20% of peak"
    )

    # Should be decaying toward zero at 500ms
    g_at_end = g_gaba_b[-1]
    assert g_at_end < 0.15 * peak_val, (
        f"GABA_B at 500ms ({g_at_end:.4f}) should be <15% of peak ({peak_val:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 7: GABA_B reversal potential
# ---------------------------------------------------------------------------

def test_gaba_b_reversal():
    """GABA_B current at v = E_GABA_B = -95 mV should be ~0."""
    ss = SynapseState(
        g_ampa=jnp.array([0.0]),
        g_gaba_a=jnp.array([0.0]),
        g_nmda_rise=jnp.array([0.0]),
        g_nmda_decay=jnp.array([0.0]),
        g_gaba_b_rise=jnp.array([0.0]),
        g_gaba_b_decay=jnp.array([1.0]),  # g_gaba_b = 1.0
    )

    v = jnp.array([E_GABA_B])  # -95 mV
    I_syn = compute_synaptic_current(ss, v)

    # At reversal potential, driving force is zero so current should be ~0
    # (NMDA contribution is also ~0 since g_nmda=0)
    npt.assert_allclose(float(I_syn[0]), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: Backward compatibility
# ---------------------------------------------------------------------------

def test_backward_compatible():
    """Old code using just g_ampa and g_gaba_a still works with extended
    SynapseState when NMDA/GABA_B fields are zero."""
    v = jnp.array([-65.0])

    # All new fields zeroed -- should give same result as Phase 1
    ss = SynapseState(
        g_ampa=jnp.array([1.0]),
        g_gaba_a=jnp.array([0.5]),
        g_nmda_rise=jnp.array([0.0]),
        g_nmda_decay=jnp.array([0.0]),
        g_gaba_b_rise=jnp.array([0.0]),
        g_gaba_b_decay=jnp.array([0.0]),
    )

    I_syn = compute_synaptic_current(ss, v)

    # Expected: AMPA + GABA_A only
    I_ampa = 1.0 * (E_AMPA - (-65.0))    # 65.0
    I_gaba_a = 0.5 * (E_GABA_A - (-65.0))  # 0.5 * (-10) = -5.0
    expected = I_ampa + I_gaba_a  # 60.0

    npt.assert_allclose(float(I_syn[0]), expected, atol=1e-5)

    # Also verify create_synapse_state produces correct shape
    ss_new = create_synapse_state(10)
    assert len(ss_new) == 6, "SynapseState should have 6 fields"
    assert ss_new.g_ampa.shape == (10,)
    assert ss_new.g_nmda_rise.shape == (10,)
    assert ss_new.g_nmda_decay.shape == (10,)
    assert ss_new.g_gaba_b_rise.shape == (10,)
    assert ss_new.g_gaba_b_decay.shape == (10,)
    npt.assert_allclose(np.asarray(ss_new.g_nmda_rise), 0.0)
    npt.assert_allclose(np.asarray(ss_new.g_nmda_decay), 0.0)
    npt.assert_allclose(np.asarray(ss_new.g_gaba_b_rise), 0.0)
    npt.assert_allclose(np.asarray(ss_new.g_gaba_b_decay), 0.0)


# ---------------------------------------------------------------------------
# Test 9: Excitatory co-release (AMPA + NMDA)
# ---------------------------------------------------------------------------

def test_excitatory_co_release():
    """Excitatory spikes should activate both AMPA and NMDA simultaneously.
    After one spike, both g_ampa > 0 and g_nmda > 0.

    Note: for the dual-exponential NMDA model, g_nmda = g_decay - g_rise
    is zero immediately after the spike (both components receive equal
    increments).  The conductance becomes positive after a second timestep
    when the rise component decays faster than the decay component.  We
    therefore step twice: once with the spike, once without.
    """
    n = 2
    # Neuron 1 -> Neuron 0, weight 0.05
    W_exc = jnp.array([[0.0, 0.05],
                       [0.0, 0.0]])
    dt = 0.5

    # Initial state
    g_ampa = jnp.zeros(n)
    g_nmda_rise = jnp.zeros(n)
    g_nmda_decay = jnp.zeros(n)

    # Neuron 1 spikes at step 0
    spikes = jnp.array([0.0, 1.0])
    no_spikes = jnp.array([0.0, 0.0])

    # Step 1: spike arrives
    g_ampa = ampa_step(g_ampa, spikes, W_exc, dt)
    g_nmda_rise, g_nmda_decay, _ = nmda_step(
        g_nmda_rise, g_nmda_decay, spikes, W_exc, dt,
    )

    # AMPA is already positive after one step (single-exponential)
    assert float(g_ampa[0]) > 0, "AMPA should be activated by excitatory spike"

    # Step 2: no spike -- rise decays faster than decay, so g_nmda > 0
    g_ampa = ampa_step(g_ampa, no_spikes, W_exc, dt)
    g_nmda_rise, g_nmda_decay, g_nmda = nmda_step(
        g_nmda_rise, g_nmda_decay, no_spikes, W_exc, dt,
    )

    # Now NMDA should be positive
    assert float(g_nmda[0]) > 0, "NMDA should be positive after rise/decay diverge"

    # Both receptors activated on the same postsynaptic neuron
    assert float(g_ampa[0]) > 0, "AMPA should still be positive"

    # Neuron 1 should not receive input (no self-connections)
    npt.assert_allclose(float(g_ampa[1]), 0.0, atol=1e-7)
    npt.assert_allclose(float(g_nmda[1]), 0.0, atol=1e-7)
