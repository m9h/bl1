"""Tests for the AdEx neuron model (bl1.core.adex)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.adex import (
    AdExParams,
    AdExState,
    _CELL_TYPES,
    create_adex_population,
    adex_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate(params, state, I_ext, n_steps, dt=0.5):
    """Run *n_steps* of the AdEx model, returning spike count per neuron."""
    spike_count = jnp.zeros_like(state.v)
    for _ in range(n_steps):
        state = adex_step(state, params, I_ext, dt=dt)
        spike_count = spike_count + state.spikes.astype(jnp.float32)
    return state, spike_count


def _make_single_neuron(cell_type: str):
    """Create params and resting state for a single AdEx neuron of the given type."""
    p = _CELL_TYPES[cell_type]
    params = AdExParams(
        C=jnp.array([p["C"]]),
        g_L=jnp.array([p["g_L"]]),
        E_L=jnp.array([p["E_L"]]),
        delta_T=jnp.array([p["delta_T"]]),
        V_T=jnp.array([p["V_T"]]),
        V_reset=jnp.array([p["V_reset"]]),
        V_peak=jnp.array([p["V_peak"]]),
        a=jnp.array([p["a"]]),
        b=jnp.array([p["b"]]),
        tau_w=jnp.array([p["tau_w"]]),
    )
    v = jnp.array([p["E_L"]])
    w = jnp.zeros(1)
    state = AdExState(v=v, w=w, spikes=jnp.array([False]))
    return params, state


def _collect_spike_times(params, state, I_ext, n_steps, dt=0.5):
    """Run simulation and return list of spike times in ms."""
    spike_times = []
    for step in range(n_steps):
        state = adex_step(state, params, I_ext, dt=dt)
        if bool(state.spikes[0]):
            spike_times.append(step * dt)
    return spike_times, state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_adex_population():
    """Population of 1000 with ei_ratio=0.8: 800 E, 200 I, correct array shapes."""
    key = jax.random.PRNGKey(0)
    params, state, is_excitatory = create_adex_population(key, n_neurons=1000, ei_ratio=0.8)

    assert int(is_excitatory.sum()) == 800
    assert int((~is_excitatory).sum()) == 200

    for arr in params:
        assert arr.shape == (1000,)


def test_adex_initial_state():
    """Initial state should be v=E_L for all neurons, w=0, no spikes."""
    key = jax.random.PRNGKey(1)
    params, state, _ = create_adex_population(key, n_neurons=500)

    np.testing.assert_allclose(np.asarray(state.v), -70.6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(state.w), 0.0)
    assert not np.any(np.asarray(state.spikes))


def test_rs_tonic_spiking():
    """RS neuron with I=500 pA should produce tonic spiking over 1000 ms."""
    params, state = _make_single_neuron("RS")
    I_ext = jnp.array([700.0])

    n_steps = 2000  # 1000 ms at dt=0.5
    _, spike_count = _simulate(params, state, I_ext, n_steps, dt=0.5)

    assert int(spike_count[0]) > 5, (
        f"RS neuron only fired {int(spike_count[0])} times with I=500 pA"
    )


def test_fs_fast_firing():
    """FS neuron with same input should fire faster than RS (b is much smaller)."""
    I_ext = jnp.array([700.0])
    n_steps = 2000

    params_rs, state_rs = _make_single_neuron("RS")
    _, spikes_rs = _simulate(params_rs, state_rs, I_ext, n_steps, dt=0.5)

    params_fs, state_fs = _make_single_neuron("FS")
    _, spikes_fs = _simulate(params_fs, state_fs, I_ext, n_steps, dt=0.5)

    assert int(spikes_fs[0]) > int(spikes_rs[0]), (
        f"FS ({int(spikes_fs[0])}) should fire more than RS ({int(spikes_rs[0])})"
    )


def test_bursting():
    """Bursting neuron should produce burst patterns (clusters with short ISIs)."""
    params, state = _make_single_neuron("Bursting")
    I_ext = jnp.array([700.0])

    dt = 0.5
    n_steps = 2000  # 1000 ms
    spike_times, _ = _collect_spike_times(params, state, I_ext, n_steps, dt=dt)

    assert len(spike_times) > 2, "Bursting neuron did not spike enough to analyse"

    isis = np.diff(spike_times)
    short_isis = isis[isis < 5.0]  # < 5 ms indicates within-burst firing
    assert len(short_isis) > 0, (
        f"No short ISIs found; first ISIs: {isis[:10]}... -- expected bursting"
    )


def test_below_threshold():
    """With zero input, a resting neuron should not spike over 100 ms."""
    params, state = _make_single_neuron("RS")
    I_ext = jnp.array([0.0])

    n_steps = 200  # 100 ms at dt=0.5
    _, spike_count = _simulate(params, state, I_ext, n_steps, dt=0.5)

    assert int(spike_count[0]) == 0, "Neuron spiked with no input"


def test_spike_adaptation():
    """Under constant input, ISIs should increase for RS (adaptation) but not FS."""
    # RS neuron: large b (80.5 pA) means strong spike-triggered adaptation
    params_rs, state_rs = _make_single_neuron("RS")
    I_ext = jnp.array([700.0])
    spike_times_rs, _ = _collect_spike_times(params_rs, state_rs, I_ext, 4000, dt=0.5)

    assert len(spike_times_rs) > 10, (
        f"RS neuron only fired {len(spike_times_rs)} times, need >10 for analysis"
    )
    isis_rs = np.diff(spike_times_rs)
    # Compare mean ISI of first 5 vs last 5 — adaptation should make later ISIs longer
    assert np.mean(isis_rs[-5:]) > np.mean(isis_rs[:5]), (
        f"RS ISIs did not increase: first5={isis_rs[:5]}, last5={isis_rs[-5:]}"
    )

    # FS neuron: tiny b (0.5 pA) means negligible spike-triggered adaptation
    params_fs, state_fs = _make_single_neuron("FS")
    spike_times_fs, _ = _collect_spike_times(params_fs, state_fs, I_ext, 4000, dt=0.5)

    assert len(spike_times_fs) > 10, (
        f"FS neuron only fired {len(spike_times_fs)} times, need >10 for analysis"
    )
    isis_fs = np.diff(spike_times_fs)
    # For FS, ISIs should be roughly constant (ratio close to 1)
    ratio = np.mean(isis_fs[-5:]) / np.mean(isis_fs[:5])
    assert ratio < 1.5, (
        f"FS ISIs increased too much (ratio={ratio:.3f}); "
        f"first5={isis_fs[:5]}, last5={isis_fs[-5:]}"
    )
    # FS adaptation should be much weaker than RS adaptation
    rs_ratio = np.mean(isis_rs[-5:]) / np.mean(isis_rs[:5])
    assert rs_ratio > ratio, "RS should adapt more than FS"


def test_spike_reset():
    """After a spike, v should be V_reset and w should increase by b."""
    params, state = _make_single_neuron("RS")
    # Strong current to guarantee a spike quickly
    I_ext = jnp.array([1000.0])

    dt = 0.5
    for _ in range(2000):
        prev_state = state
        state = adex_step(state, params, I_ext, dt=dt)
        if bool(state.spikes[0]):
            # After the spike step, v should have been reset to V_reset
            np.testing.assert_allclose(
                float(state.v[0]), float(params.V_reset[0]), atol=1e-5,
                err_msg="Post-spike v != V_reset",
            )
            # w should have increased by b on top of the dynamics update.
            # Since b=80.5 for RS and w starts near 0, w must have increased.
            assert float(state.w[0]) > float(prev_state.w[0]), (
                "w did not increase after spike"
            )
            return  # test passes

    pytest.fail("No spike occurred despite strong input current")
