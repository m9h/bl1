"""Tests for the MEA module (bl1.mea)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.mea.electrode import MEA, MEAConfig, build_neuron_electrode_map
from bl1.mea.recording import detect_spikes, compute_electrode_rates
from bl1.mea.stimulation import apply_stimulation, generate_feedback_stim


# ---------------------------------------------------------------------------
# MEA creation and layout
# ---------------------------------------------------------------------------

def test_mea_cl1_creation():
    """MEA("cl1_64ch") creates 64 electrodes with grid shape (8, 8)."""
    mea = MEA("cl1_64ch")
    assert mea.n_electrodes == 64
    assert mea.config.grid_shape == (8, 8)
    assert mea.positions.shape == (64, 2)


def test_mea_cl1_positions():
    """Electrodes are centered on the 3000x3000 substrate.

    Positions should range from approximately 800 to 2200 in both x and y
    (center 1500 +/- 3.5 * 200).
    """
    mea = MEA("cl1_64ch")
    positions = np.asarray(mea.positions)

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # Expected: center=1500, offsets from -3.5*200 to +3.5*200
    # So range is [1500 - 700, 1500 + 700] = [800, 2200]
    np.testing.assert_allclose(x_min, 800.0, atol=1.0)
    np.testing.assert_allclose(x_max, 2200.0, atol=1.0)
    np.testing.assert_allclose(y_min, 800.0, atol=1.0)
    np.testing.assert_allclose(y_max, 2200.0, atol=1.0)


# ---------------------------------------------------------------------------
# Neuron-electrode mapping
# ---------------------------------------------------------------------------

def test_build_neuron_electrode_map():
    """Place 100 neurons uniformly; map with radius=100um.

    Verify shape (64, 100).  Neurons near electrode centers should be True;
    distant neurons should be False.
    """
    mea = MEA("cl1_64ch")
    key = jax.random.PRNGKey(42)
    neuron_positions = jax.random.uniform(key, shape=(100, 2)) * 3000.0

    ne_map = build_neuron_electrode_map(
        neuron_positions, mea.positions, radius_um=100.0,
    )

    assert ne_map.shape == (64, 100)
    assert ne_map.dtype == jnp.bool_

    # Verify correctness for a specific case: place a neuron exactly at
    # electrode 0's position -- it must be mapped.
    electrode_0_pos = np.asarray(mea.positions[0])
    special_positions = jnp.array([electrode_0_pos, [0.0, 0.0]])  # (2, 2)
    ne_map2 = build_neuron_electrode_map(
        special_positions, mea.positions, radius_um=100.0,
    )
    # Electrode 0 should see neuron 0 (at its exact position)
    assert bool(ne_map2[0, 0])
    # Electrode 0 should NOT see neuron 1 (at origin, far from electrode 0 at ~800,800)
    assert not bool(ne_map2[0, 1])


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def test_detect_spikes():
    """Electrode 0 sees neurons [0, 1, 2]; neurons 0 and 2 spike.

    detect_spikes should return count=2 for electrode 0.
    """
    n_electrodes = 3
    n_neurons = 5

    # Build a simple neuron-electrode map: electrode 0 sees neurons 0, 1, 2
    ne_map = jnp.zeros((n_electrodes, n_neurons), dtype=jnp.bool_)
    ne_map = ne_map.at[0, 0].set(True)
    ne_map = ne_map.at[0, 1].set(True)
    ne_map = ne_map.at[0, 2].set(True)

    # Neurons 0 and 2 spike this timestep
    spikes = jnp.array([True, False, True, False, False])

    counts = detect_spikes(spikes, ne_map)

    assert counts.shape == (n_electrodes,)
    assert int(counts[0]) == 2  # electrode 0: neurons 0 + 2
    assert int(counts[1]) == 0  # electrode 1 sees nobody
    assert int(counts[2]) == 0  # electrode 2 sees nobody


# ---------------------------------------------------------------------------
# Electrode firing rates
# ---------------------------------------------------------------------------

def test_compute_electrode_rates():
    """With known spike history and map, verify the computed rate in Hz."""
    n_electrodes = 2
    n_neurons = 4
    dt = 0.5  # ms
    window_ms = 100.0  # 200 timesteps at dt=0.5

    # Electrode 0 sees neurons [0, 1]; electrode 1 sees neurons [2, 3]
    ne_map = jnp.zeros((n_electrodes, n_neurons), dtype=jnp.bool_)
    ne_map = ne_map.at[0, 0].set(True)
    ne_map = ne_map.at[0, 1].set(True)
    ne_map = ne_map.at[1, 2].set(True)
    ne_map = ne_map.at[1, 3].set(True)

    T = 200  # timesteps in the window
    spike_history = jnp.zeros((T, n_neurons))

    # Neuron 0 spikes 10 times, neuron 1 spikes 10 times in the window
    # Total: electrode 0 sees 20 spikes across 2 neurons over 100 ms
    # Mean spikes per neuron = 10
    # Rate = 10 / 0.1 s = 100 Hz
    for i in range(0, 20, 2):
        spike_history = spike_history.at[i, 0].set(1.0)
    for i in range(1, 21, 2):
        spike_history = spike_history.at[i, 1].set(1.0)

    rates = compute_electrode_rates(spike_history, ne_map, window_ms, dt)

    assert rates.shape == (n_electrodes,)
    # Electrode 0: each of 2 neurons has 10 spikes in 100ms -> mean 10 spikes -> 100 Hz
    np.testing.assert_allclose(float(rates[0]), 100.0, rtol=0.01)
    # Electrode 1: neurons 2 and 3 have 0 spikes -> 0 Hz
    np.testing.assert_allclose(float(rates[1]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Stimulation
# ---------------------------------------------------------------------------

def test_apply_stimulation():
    """Neuron at electrode 0 position receives full amplitude.

    A neuron at half the activation radius should receive half amplitude.
    """
    mea = MEA("cl1_64ch")
    activation_radius = mea.activation_radius_um  # 75.0
    electrode_0_pos = np.asarray(mea.positions[0])

    # Neuron 0: exactly at electrode 0 -> full amplitude
    # Neuron 1: at half the activation radius from electrode 0 -> half amplitude
    half_radius_offset = jnp.array([activation_radius / 2.0, 0.0])
    neuron_positions = jnp.array([
        electrode_0_pos,
        electrode_0_pos + half_radius_offset,
    ])

    # Only electrode 0 is active
    stim_electrodes = jnp.zeros(mea.n_electrodes, dtype=jnp.bool_)
    stim_electrodes = stim_electrodes.at[0].set(True)

    amplitude = 150.0
    I_stim = apply_stimulation(
        neuron_positions,
        mea.positions,
        stim_electrodes,
        stim_amplitude=amplitude,
        activation_radius_um=activation_radius,
    )

    assert I_stim.shape == (2,)
    # Neuron 0 at distance 0: attenuation = 1.0 -> full amplitude
    np.testing.assert_allclose(float(I_stim[0]), amplitude, atol=1.0)
    # Neuron 1 at half radius: attenuation = 1 - 0.5 = 0.5 -> half amplitude
    np.testing.assert_allclose(float(I_stim[1]), amplitude * 0.5, atol=1.0)


# ---------------------------------------------------------------------------
# Feedback stimulation patterns
# ---------------------------------------------------------------------------

def test_generate_feedback_predictable():
    """'predictable' feedback activates all sensory channels synchronously.

    All timing offsets should be zero.
    """
    n_electrodes = 64
    key = jax.random.PRNGKey(7)

    # Designate electrodes 0-7 as sensory channels
    sensory_channels = jnp.zeros(n_electrodes, dtype=jnp.bool_)
    sensory_channels = sensory_channels.at[:8].set(True)

    electrode_positions = jnp.zeros((n_electrodes, 2))

    stim_mask, timing_offsets = generate_feedback_stim(
        "predictable", sensory_channels, electrode_positions, key,
    )

    # All sensory channels should be activated
    np.testing.assert_array_equal(np.asarray(stim_mask), np.asarray(sensory_channels))

    # All timing offsets should be zero
    np.testing.assert_allclose(np.asarray(timing_offsets), 0.0)


def test_generate_feedback_unpredictable():
    """'unpredictable' feedback activates a random subset with non-zero timing offsets."""
    n_electrodes = 64
    key = jax.random.PRNGKey(123)

    sensory_channels = jnp.zeros(n_electrodes, dtype=jnp.bool_)
    sensory_channels = sensory_channels.at[:8].set(True)

    electrode_positions = jnp.zeros((n_electrodes, 2))

    stim_mask, timing_offsets = generate_feedback_stim(
        "unpredictable", sensory_channels, electrode_positions, key,
    )

    stim_np = np.asarray(stim_mask)
    timing_np = np.asarray(timing_offsets)

    # Should activate only a subset of sensory channels (not necessarily all)
    # With p=0.5 and 8 channels it is extremely unlikely to get all 8 or 0.
    # We just verify that the stimulated electrodes are a subset of sensory channels.
    assert np.all(stim_np[8:] == False), "Non-sensory electrodes should not be activated"

    # At least some stimulated electrodes should have non-zero timing offsets
    active_offsets = timing_np[stim_np]
    if len(active_offsets) > 0:
        assert np.any(active_offsets > 0.0), (
            "Unpredictable feedback should have non-zero timing offsets"
        )

    # Non-stimulated electrodes should have zero offsets
    inactive_offsets = timing_np[~stim_np]
    np.testing.assert_allclose(inactive_offsets, 0.0)
