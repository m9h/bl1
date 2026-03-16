"""Tests for the Izhikevich neuron model (bl1.core.izhikevich)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    V_REST,
    create_population,
    izhikevich_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate(params, state, I_ext, n_steps, dt=0.5):
    """Run *n_steps* of the Izhikevich model, returning spike count per neuron."""
    spike_count = jnp.zeros_like(state.v)
    for _ in range(n_steps):
        state = izhikevich_step(state, params, I_ext, dt=dt)
        spike_count = spike_count + state.spikes.astype(jnp.float32)
    return state, spike_count


def _make_single_neuron(a, b, c, d):
    """Create params and resting state for a single neuron."""
    params = IzhikevichParams(
        a=jnp.array([a]),
        b=jnp.array([b]),
        c=jnp.array([c]),
        d=jnp.array([d]),
    )
    v = jnp.array([V_REST])
    u = jnp.array([b * V_REST])
    state = NeuronState(v=v, u=u, spikes=jnp.array([False]))
    return params, state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_population_sizes():
    """Population of 1000 with ei_ratio=0.8: 800 E, 200 I, correct array shapes."""
    key = jax.random.PRNGKey(0)
    params, state, is_excitatory = create_population(key, n_neurons=1000, ei_ratio=0.8)

    assert int(is_excitatory.sum()) == 800
    assert int((~is_excitatory).sum()) == 200

    for arr in (params.a, params.b, params.c, params.d):
        assert arr.shape == (1000,)


def test_create_population_cell_types():
    """Large population should approximate expected cell-type fractions (+-2%)."""
    key = jax.random.PRNGKey(42)
    n = 10_000
    params, _, is_excitatory = create_population(key, n_neurons=n, ei_ratio=0.8)

    # Cell types are laid out contiguously: RS, IB, CH, FS, LTS.
    # We can identify boundaries via parameter fingerprints.
    a, b, c, d = np.asarray(params.a), np.asarray(params.b), np.asarray(params.c), np.asarray(params.d)

    rs_mask = (np.isclose(a, 0.02) & np.isclose(b, 0.2) & np.isclose(c, -65.0) & np.isclose(d, 8.0))
    ib_mask = (np.isclose(a, 0.02) & np.isclose(b, 0.2) & np.isclose(c, -55.0) & np.isclose(d, 4.0))
    ch_mask = (np.isclose(a, 0.02) & np.isclose(b, 0.2) & np.isclose(c, -50.0) & np.isclose(d, 2.0))
    fs_mask = (np.isclose(a, 0.1)  & np.isclose(b, 0.2) & np.isclose(c, -65.0) & np.isclose(d, 2.0))
    lts_mask = (np.isclose(a, 0.02) & np.isclose(b, 0.25) & np.isclose(c, -65.0) & np.isclose(d, 2.0))

    expected = {"RS": 0.64, "IB": 0.08, "CH": 0.08, "FS": 0.16, "LTS": 0.04}
    actual = {
        "RS": rs_mask.sum() / n,
        "IB": ib_mask.sum() / n,
        "CH": ch_mask.sum() / n,
        "FS": fs_mask.sum() / n,
        "LTS": lts_mask.sum() / n,
    }

    for ct in expected:
        assert abs(actual[ct] - expected[ct]) < 0.02, (
            f"Cell type {ct}: expected ~{expected[ct]:.2f}, got {actual[ct]:.4f}"
        )


def test_initial_state():
    """Initial state should be v=-65 for all neurons, u=b*v, no spikes."""
    key = jax.random.PRNGKey(1)
    params, state, _ = create_population(key, n_neurons=500)

    np.testing.assert_allclose(np.asarray(state.v), -65.0)
    np.testing.assert_allclose(np.asarray(state.u), np.asarray(params.b * state.v))
    assert not np.any(np.asarray(state.spikes))


def test_rs_tonic_spiking():
    """RS neuron with constant I=10 should produce tonic spiking (>5 spikes in 1s)."""
    params, state = _make_single_neuron(a=0.02, b=0.2, c=-65.0, d=8.0)
    I_ext = jnp.array([10.0])

    n_steps = 2000  # 1000 ms at dt=0.5
    _, spike_count = _simulate(params, state, I_ext, n_steps, dt=0.5)

    assert int(spike_count[0]) > 5, f"RS neuron only fired {int(spike_count[0])} times"


def test_fs_fast_firing():
    """FS neuron should fire more frequently than RS with the same input."""
    I_ext = jnp.array([10.0])
    n_steps = 2000

    params_rs, state_rs = _make_single_neuron(a=0.02, b=0.2, c=-65.0, d=8.0)
    _, spikes_rs = _simulate(params_rs, state_rs, I_ext, n_steps, dt=0.5)

    params_fs, state_fs = _make_single_neuron(a=0.1, b=0.2, c=-65.0, d=2.0)
    _, spikes_fs = _simulate(params_fs, state_fs, I_ext, n_steps, dt=0.5)

    assert int(spikes_fs[0]) > int(spikes_rs[0]), (
        f"FS ({int(spikes_fs[0])}) should fire more than RS ({int(spikes_rs[0])})"
    )


def test_ib_bursting():
    """IB neuron should produce bursts — some ISIs should be very short (<5 ms)."""
    params, state = _make_single_neuron(a=0.02, b=0.2, c=-55.0, d=4.0)
    I_ext = jnp.array([10.0])

    dt = 0.5
    n_steps = 2000
    spike_times = []
    for step in range(n_steps):
        state = izhikevich_step(state, params, I_ext, dt=dt)
        if bool(state.spikes[0]):
            spike_times.append(step * dt)

    assert len(spike_times) > 2, "IB neuron did not spike enough to analyse bursting"

    isis = np.diff(spike_times)
    short_isis = isis[isis < 5.0]  # < 5 ms indicates within-burst firing
    assert len(short_isis) > 0, (
        f"No short ISIs found; all ISIs: {isis[:10]}... — expected bursting"
    )


def test_below_threshold():
    """With zero input, a resting neuron should not spike over 100 ms."""
    params, state = _make_single_neuron(a=0.02, b=0.2, c=-65.0, d=8.0)
    I_ext = jnp.array([0.0])

    n_steps = 200  # 100 ms at dt=0.5
    _, spike_count = _simulate(params, state, I_ext, n_steps, dt=0.5)

    assert int(spike_count[0]) == 0, "Neuron spiked with no input"


def test_spike_reset():
    """After a spike, v should be reset to c and u should increase by d."""
    params, state = _make_single_neuron(a=0.02, b=0.2, c=-65.0, d=8.0)
    # Strong current to guarantee a spike quickly
    I_ext = jnp.array([40.0])

    dt = 0.5
    for _ in range(1000):
        prev_state = state
        state = izhikevich_step(state, params, I_ext, dt=dt)
        if bool(state.spikes[0]):
            # After the spike step, v should have been reset to c
            np.testing.assert_allclose(
                float(state.v[0]), float(params.c[0]), atol=1e-5,
                err_msg="Post-spike v != c",
            )
            # u should have had d added (on top of the dynamics update)
            # We just verify u increased relative to what it was before the
            # dynamics-only update: the code does u_new = u + dt*a*(b*v - u) + d
            # so u should be strictly greater than previous u for moderate d.
            assert float(state.u[0]) > float(prev_state.u[0]), (
                "u did not increase after spike"
            )
            return  # test passes

    pytest.fail("No spike occurred despite strong input current")
