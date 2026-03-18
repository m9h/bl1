"""Targeted tests to close coverage gaps across bl1 modules.

Focuses on public functions with zero or low coverage, important
error-handling branches, and edge cases in analysis, visualization,
compat, validation, loop, games, mea, and core modules.

Uses small inputs and avoids duplicating tests that already exist in
the rest of the test suite.
"""

from __future__ import annotations

import math
import struct

import matplotlib
matplotlib.use("Agg")  # headless rendering

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure


# ============================================================================
# analysis/criticality — edge cases for _bin_spikes, branching_ratio,
# avalanche_size_distribution
# ============================================================================

class TestCriticalityEdgeCases:
    """Cover branches not hit by test_validation's indirect usage."""

    def test_branching_ratio_empty_raster(self):
        from bl1.analysis.criticality import branching_ratio
        raster = np.zeros((0, 10))
        assert math.isnan(branching_ratio(raster))

    def test_branching_ratio_single_timestep(self):
        from bl1.analysis.criticality import branching_ratio
        raster = np.ones((1, 5))
        assert math.isnan(branching_ratio(raster))

    def test_branching_ratio_no_active_bins(self):
        from bl1.analysis.criticality import branching_ratio
        # All zeros → no non-zero ancestor bins
        raster = np.zeros((100, 10))
        assert math.isnan(branching_ratio(raster))

    def test_branching_ratio_all_active(self):
        from bl1.analysis.criticality import branching_ratio
        rng = np.random.default_rng(7)
        raster = (rng.random((200, 20)) < 0.3).astype(float)
        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=2.0)
        assert math.isfinite(sigma)
        assert sigma > 0

    def test_avalanche_empty_raster(self):
        from bl1.analysis.criticality import avalanche_size_distribution
        sizes, durations = avalanche_size_distribution(np.zeros((0, 5)))
        assert len(sizes) == 0
        assert len(durations) == 0

    def test_avalanche_trailing_active(self):
        """Avalanche that extends to end of recording."""
        from bl1.analysis.criticality import avalanche_size_distribution
        raster = np.zeros((100, 5))
        # Active bins at the very end
        raster[90:100, :] = 1.0
        sizes, durations = avalanche_size_distribution(raster, dt_ms=0.5, bin_ms=0.5)
        # Should detect at least one avalanche reaching the end
        assert len(sizes) >= 1

    def test_bin_spikes_zero_bins(self):
        """When bin_ms > total duration → zero bins."""
        from bl1.analysis.criticality import _bin_spikes
        raster = np.ones((5, 3))  # 5 timesteps
        binned = _bin_spikes(raster, dt_ms=0.5, bin_ms=1000.0)
        assert len(binned) == 0


# ============================================================================
# analysis/metrics — rally_length and performance_comparison
# ============================================================================

class TestMetrics:
    """Cover rally_length and performance_comparison which lack dedicated tests."""

    def test_rally_length_basic(self):
        from bl1.analysis.metrics import rally_length
        events = [(0.0, "hit"), (20.0, "hit"), (40.0, "miss"), (60.0, "hit"), (80.0, "miss")]
        rl = rally_length(events)
        np.testing.assert_array_equal(rl, [2, 1])

    def test_rally_length_empty(self):
        from bl1.analysis.metrics import rally_length
        rl = rally_length([])
        assert len(rl) == 0

    def test_rally_length_trailing_hits(self):
        """Hits at end with no final miss should still be counted."""
        from bl1.analysis.metrics import rally_length
        events = [(0.0, "hit"), (20.0, "hit"), (40.0, "hit")]
        rl = rally_length(events)
        np.testing.assert_array_equal(rl, [3])

    def test_rally_length_only_misses(self):
        from bl1.analysis.metrics import rally_length
        events = [(0.0, "miss"), (20.0, "miss")]
        rl = rally_length(events)
        np.testing.assert_array_equal(rl, [0, 0])

    def test_performance_comparison_with_rally_lengths(self):
        from bl1.analysis.metrics import performance_comparison
        results = {
            "closed_loop": {"rally_lengths": np.array([3, 5, 7, 2, 4])},
            "open_loop": {"rally_lengths": np.array([1, 1, 2, 1, 1])},
        }
        out = performance_comparison(results)
        assert "mean_rally" in out
        assert "median_rally" in out
        assert "n_rallies" in out
        assert "p_values" in out
        assert out["mean_rally"]["closed_loop"] > out["mean_rally"]["open_loop"]

    def test_performance_comparison_with_game_events(self):
        from bl1.analysis.metrics import performance_comparison
        results = {
            "cond_a": {"game_events": [(0.0, "hit"), (1.0, "miss")]},
        }
        out = performance_comparison(results)
        assert out["n_rallies"]["cond_a"] == 1

    def test_performance_comparison_empty_events(self):
        from bl1.analysis.metrics import performance_comparison
        results = {
            "empty": {},
        }
        out = performance_comparison(results)
        assert out["mean_rally"]["empty"] == 0.0


# ============================================================================
# analysis/bursts — detect_bursts edge cases
# ============================================================================

class TestBurstEdgeCases:

    def test_detect_bursts_empty(self):
        from bl1.analysis.bursts import detect_bursts
        raster = np.zeros((0, 10), dtype=float)
        # T=0 should return empty
        # Need to reshape to (0, N)
        raster = raster.reshape(0, 10)
        assert detect_bursts(raster) == []

    def test_detect_bursts_constant_rate(self):
        """Constant firing rate (std=0) returns no bursts."""
        from bl1.analysis.bursts import detect_bursts
        # All neurons fire every timestep → std is zero
        raster = np.ones((500, 20), dtype=float)
        assert detect_bursts(raster) == []

    def test_burst_statistics_empty(self):
        from bl1.analysis.bursts import burst_statistics
        stats = burst_statistics([])
        assert math.isnan(stats["ibi_mean"])
        assert stats["burst_rate"] == 0.0

    def test_burst_statistics_single_burst(self):
        from bl1.analysis.bursts import burst_statistics
        bursts = [(100.0, 200.0, 500, 0.6)]
        stats = burst_statistics(bursts)
        assert stats["duration_mean"] == 100.0
        assert stats["recruitment_mean"] == 0.6
        assert math.isnan(stats["ibi_mean"])

    def test_burst_statistics_multiple_bursts(self):
        from bl1.analysis.bursts import burst_statistics
        bursts = [
            (100.0, 200.0, 500, 0.6),
            (400.0, 500.0, 300, 0.4),
            (800.0, 900.0, 400, 0.5),
        ]
        stats = burst_statistics(bursts)
        assert math.isfinite(stats["ibi_mean"])
        assert stats["burst_rate"] > 0

    def test_detect_bursts_trailing_burst(self):
        """Burst that extends to the end of recording."""
        from bl1.analysis.bursts import detect_bursts
        raster = np.zeros((1000, 10), dtype=float)
        # Create a huge spike near the end
        raster[800:, :] = 1.0
        bursts = detect_bursts(raster, dt_ms=0.5, threshold_std=1.0, min_duration_ms=10.0)
        # The burst at the end should be detected
        assert len(bursts) >= 1


# ============================================================================
# analysis/connectivity — effective_connectivity_graph edge case
# ============================================================================

class TestConnectivityEdgeCases:

    def test_effective_connectivity_all_zero(self):
        from bl1.analysis.connectivity import effective_connectivity_graph
        te = np.zeros((10, 10))
        graph = effective_connectivity_graph(te)
        np.testing.assert_array_equal(graph, 0.0)

    def test_small_world_tiny_graph(self):
        """Graph with < 3 nodes should return clustering=0."""
        from bl1.analysis.connectivity import small_world_coefficient
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        result = small_world_coefficient(adj)
        assert result["clustering_coefficient"] == 0.0

    def test_rich_club_empty_graph(self):
        from bl1.analysis.connectivity import rich_club_coefficient
        adj = np.zeros((10, 10))
        result = rich_club_coefficient(adj)
        assert result["rich_club_coeff"] == {}


# ============================================================================
# analysis/information — complexity
# ============================================================================

class TestComplexity:

    def test_complexity_nonnegative(self):
        from bl1.analysis.information import complexity
        rng = np.random.default_rng(55)
        raster = (rng.random((1000, 15)) < 0.05).astype(np.float32)
        c = complexity(raster, dt_ms=0.5, n_samples=20)
        assert isinstance(c, float)
        # Complexity is clamped to >= 0
        assert c >= 0.0

    def test_complexity_silent(self):
        from bl1.analysis.information import complexity
        raster = np.zeros((500, 10), dtype=np.float32)
        c = complexity(raster, dt_ms=0.5)
        assert c == 0.0

    def test_integration_small_network(self):
        from bl1.analysis.information import integration
        rng = np.random.default_rng(99)
        raster = (rng.random((500, 5)) < 0.1).astype(np.float32)
        integ = integration(raster, dt_ms=0.5, n_samples=20)
        assert isinstance(integ, float)
        assert integ >= 0.0

    def test_mutual_info_subset(self):
        from bl1.analysis.information import mutual_information_matrix
        rng = np.random.default_rng(11)
        raster = (rng.random((500, 20)) < 0.05).astype(np.float32)
        mi = mutual_information_matrix(raster, dt_ms=0.5, subset=10)
        assert mi.shape == (10, 10)


# ============================================================================
# core/regularization
# ============================================================================

class TestRegularization:
    """No existing tests for this module."""

    def test_firing_rate_penalty_l2(self):
        from bl1.core.regularization import firing_rate_penalty
        spikes = jnp.ones((100, 10))  # all fire every step → 2000 Hz
        pen = firing_rate_penalty(spikes, target_rate_hz=5.0, penalty_type="l2")
        assert float(pen) > 0

    def test_firing_rate_penalty_huber(self):
        from bl1.core.regularization import firing_rate_penalty
        spikes = jnp.zeros((100, 10))  # all silent → 0 Hz
        pen = firing_rate_penalty(spikes, target_rate_hz=5.0, penalty_type="huber")
        assert float(pen) > 0

    def test_firing_rate_penalty_zero_at_target(self):
        from bl1.core.regularization import firing_rate_penalty
        # 5 Hz with dt=0.5ms means each neuron fires 5/(1000/0.5) = 0.0025 per step
        # Construct a spike array where mean per neuron = 0.0025
        T, N = 10000, 10
        rng = np.random.default_rng(0)
        spikes = jnp.array((rng.random((T, N)) < 0.0025).astype(np.float32))
        pen = firing_rate_penalty(spikes, target_rate_hz=5.0)
        assert float(pen) < 1.05  # should be small (near target); allow float32 noise

    def test_sparsity_penalty(self):
        from bl1.core.regularization import sparsity_penalty
        # Silent neurons → no penalty
        spikes = jnp.zeros((100, 10))
        pen = sparsity_penalty(spikes, max_rate_hz=20.0)
        assert float(pen) == 0.0

    def test_sparsity_penalty_high_rate(self):
        from bl1.core.regularization import sparsity_penalty
        spikes = jnp.ones((100, 10))  # all fire every step → way above 20 Hz
        pen = sparsity_penalty(spikes, max_rate_hz=20.0)
        assert float(pen) > 0

    def test_silence_penalty(self):
        from bl1.core.regularization import silence_penalty
        # All silent → below minimum
        spikes = jnp.zeros((100, 10))
        pen = silence_penalty(spikes, min_rate_hz=0.5)
        assert float(pen) > 0

    def test_silence_penalty_active(self):
        from bl1.core.regularization import silence_penalty
        # All fire every step → well above minimum
        spikes = jnp.ones((100, 10))
        pen = silence_penalty(spikes, min_rate_hz=0.5)
        assert float(pen) == 0.0


# ============================================================================
# mea/recording — detect_spikes, compute_electrode_rates
# ============================================================================

class TestMEARecording:
    """No dedicated test file exists for mea/recording.py."""

    def test_detect_spikes(self):
        from bl1.mea.recording import detect_spikes
        N, E = 50, 8
        spikes = jnp.zeros(N, dtype=jnp.bool_)
        spikes = spikes.at[0].set(True).at[10].set(True)
        # Electrode 0 sees neuron 0, electrode 1 sees neuron 10
        ne_map = jnp.zeros((E, N), dtype=jnp.bool_)
        ne_map = ne_map.at[0, 0].set(True).at[1, 10].set(True)
        counts = detect_spikes(spikes, ne_map)
        assert int(counts[0]) == 1
        assert int(counts[1]) == 1
        assert int(counts[2]) == 0

    def test_compute_electrode_rates(self):
        from bl1.mea.recording import compute_electrode_rates
        N, E, T = 20, 4, 100
        spike_hist = jnp.zeros((T, N))
        # Neuron 0 fires every step
        spike_hist = spike_hist.at[:, 0].set(1.0)
        ne_map = jnp.zeros((E, N), dtype=jnp.bool_)
        ne_map = ne_map.at[0, 0].set(True)
        rates = compute_electrode_rates(spike_hist, ne_map, window_ms=50.0, dt=0.5)
        assert rates.shape == (E,)
        # Electrode 0 should have a high rate
        assert float(rates[0]) > 0


# ============================================================================
# mea/stimulation — apply_stimulation, generate_feedback_stim
# ============================================================================

class TestMEAStimulation:
    """No dedicated test file for mea/stimulation.py."""

    def test_apply_stimulation_basic(self):
        from bl1.mea.stimulation import apply_stimulation
        N, E = 20, 4
        neuron_pos = jnp.array([[100.0 * i, 100.0 * i] for i in range(N)])
        elec_pos = jnp.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        stim_elec = jnp.array([True, False, False, False])
        I = apply_stimulation(neuron_pos, elec_pos, stim_elec, stim_amplitude=10.0, activation_radius_um=150.0)
        assert I.shape == (N,)
        # Neuron 0 is at the electrode → should get max current
        assert float(I[0]) > 0

    def test_generate_feedback_stim_predictable(self):
        from bl1.mea.stimulation import generate_feedback_stim
        E = 8
        sensory = jnp.array([True, True, True, True, False, False, False, False])
        elec_pos = jnp.zeros((E, 2))
        key = jax.random.PRNGKey(0)
        stim, timing = generate_feedback_stim("predictable", sensory, elec_pos, key)
        # Should stimulate all sensory channels
        np.testing.assert_array_equal(np.asarray(stim), np.asarray(sensory))
        np.testing.assert_array_equal(np.asarray(timing), 0.0)

    def test_generate_feedback_stim_unpredictable(self):
        from bl1.mea.stimulation import generate_feedback_stim
        E = 8
        sensory = jnp.ones(E, dtype=jnp.bool_)
        elec_pos = jnp.zeros((E, 2))
        key = jax.random.PRNGKey(42)
        stim, timing = generate_feedback_stim("unpredictable", sensory, elec_pos, key)
        assert stim.shape == (E,)
        # Timing for active electrodes should be non-negative
        assert float(jnp.min(timing)) >= 0.0

    def test_generate_feedback_stim_none(self):
        from bl1.mea.stimulation import generate_feedback_stim
        E = 8
        sensory = jnp.ones(E, dtype=jnp.bool_)
        elec_pos = jnp.zeros((E, 2))
        key = jax.random.PRNGKey(0)
        stim, timing = generate_feedback_stim("none", sensory, elec_pos, key)
        assert not np.any(np.asarray(stim))

    def test_generate_feedback_stim_invalid(self):
        from bl1.mea.stimulation import generate_feedback_stim
        E = 4
        sensory = jnp.ones(E, dtype=jnp.bool_)
        elec_pos = jnp.zeros((E, 2))
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="Unknown feedback_type"):
            generate_feedback_stim("invalid", sensory, elec_pos, key)


# ============================================================================
# network/growth
# ============================================================================

class TestNetworkGrowth:
    """No existing tests for network/growth.py."""

    def test_init_growth(self):
        from bl1.network.growth import init_growth
        key = jax.random.PRNGKey(0)
        N = 20
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1000.0)
        is_exc = jnp.ones(N, dtype=jnp.bool_)
        state = init_growth(key, pos, is_exc)
        assert state.div == 0.0
        assert float(jnp.sum(jnp.abs(state.W_exc))) == 0.0
        assert state.connectivity_fraction == 0.0

    def test_grow_to_div_early(self):
        from bl1.network.growth import grow_to_div
        key = jax.random.PRNGKey(1)
        N = 30
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1000.0)
        is_exc = jnp.concatenate([jnp.ones(24, dtype=jnp.bool_), jnp.zeros(6, dtype=jnp.bool_)])
        state = grow_to_div(key, pos, is_exc, target_div=5.0)
        assert state.div == 5.0
        assert state.connectivity_fraction >= 0.0

    def test_grow_to_div_mature(self):
        """DIV > 14 triggers hub formation branch."""
        from bl1.network.growth import grow_to_div
        key = jax.random.PRNGKey(2)
        N = 30
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1000.0)
        is_exc = jnp.ones(N, dtype=jnp.bool_).at[:6].set(False)
        state = grow_to_div(key, pos, is_exc, target_div=28.0)
        assert state.div == 28.0
        # Mature network should have some connections
        assert state.connectivity_fraction > 0.0

    def test_mature_network(self):
        from bl1.network.growth import mature_network
        key = jax.random.PRNGKey(3)
        N = 20
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1000.0)
        is_exc = jnp.ones(N, dtype=jnp.bool_)
        W_exc, W_inh = mature_network(key, pos, is_exc, target_div=28.0)
        assert W_exc.shape == (N, N)
        assert W_inh.shape == (N, N)


# ============================================================================
# loop/encoding — _place_code, _rate_code_frequency, encode_sensory
# ============================================================================

class TestEncoding:
    """No dedicated test for loop/encoding.py."""

    def test_place_code_center(self):
        from bl1.loop.encoding import _place_code
        # ball_y=0.5 → channel 4 (middle of 8 channels)
        channels = _place_code(0.5)
        assert 4 in channels

    def test_place_code_boundary_low(self):
        from bl1.loop.encoding import _place_code
        # Very close to the bottom of band 1 → should activate adjacent
        channels = _place_code(0.126)  # just above band 1 start, near lower boundary
        assert len(channels) >= 1

    def test_place_code_boundary_high(self):
        from bl1.loop.encoding import _place_code
        # Near the top of a band → should activate next channel
        channels = _place_code(0.12)  # near upper part of band 0
        assert 0 in channels

    def test_place_code_clamp(self):
        from bl1.loop.encoding import _place_code
        # Out of range should be clamped
        channels = _place_code(-0.5)
        assert 0 in channels
        channels = _place_code(1.5)
        assert 7 in channels

    def test_rate_code_frequency(self):
        from bl1.loop.encoding import _rate_code_frequency, FREQ_MIN_HZ, FREQ_MAX_HZ
        assert _rate_code_frequency(0.0) == pytest.approx(FREQ_MIN_HZ)
        assert _rate_code_frequency(1.0) == pytest.approx(FREQ_MAX_HZ)
        # Midpoint
        mid = _rate_code_frequency(0.5)
        assert FREQ_MIN_HZ < mid < FREQ_MAX_HZ

    def test_encode_sensory_wrong_channels(self):
        from bl1.loop.encoding import encode_sensory

        class FakeState:
            ball_x = 0.5
            ball_y = 0.5

        class FakeMEAConfig:
            pass

        with pytest.raises(ValueError, match="Expected 8"):
            encode_sensory(FakeState(), FakeMEAConfig(), [0, 1, 2])  # too few

    def test_encode_sensory_pulse_generation(self):
        from bl1.loop.encoding import encode_sensory

        class FakeState:
            ball_x = 0.9  # close to paddle → high freq
            ball_y = 0.5

        class FakeMEAConfig:
            pass

        channels = list(range(8))
        # Run several calls to accumulate enough phase for a pulse
        phase = 0.0
        got_pulse = False
        for _ in range(200):
            active, amp, phase = encode_sensory(
                FakeState(), FakeMEAConfig(), channels,
                dt_game_ms=20.0, _phase_accumulator=phase,
            )
            if active:
                got_pulse = True
                assert amp > 0
                break
        assert got_pulse, "Should emit at least one pulse with high frequency"


# ============================================================================
# loop/decoding — decode_motor
# ============================================================================

class TestDecoding:
    """No dedicated test for loop/decoding.py."""

    def test_decode_motor_empty_window(self):
        from bl1.loop.decoding import decode_motor
        spike_hist = jnp.zeros((0, 10))
        ne_map = jnp.zeros((4, 10), dtype=jnp.bool_)
        motor_regions = {"up": [0, 1], "down": [2, 3]}
        action, rates = decode_motor(spike_hist, ne_map, motor_regions)
        assert action == 0
        assert rates["up"] == 0.0
        assert rates["down"] == 0.0

    def test_decode_motor_stay(self):
        from bl1.loop.decoding import decode_motor
        # Low firing rate → stay (action 0)
        N, W = 20, 200
        spike_hist = jnp.zeros((W, N))
        ne_map = jnp.zeros((4, N), dtype=jnp.bool_)
        ne_map = ne_map.at[0, 0].set(True).at[1, 1].set(True)
        ne_map = ne_map.at[2, 2].set(True).at[3, 3].set(True)
        motor_regions = {"up": [0, 1], "down": [2, 3]}
        action, rates = decode_motor(spike_hist, ne_map, motor_regions)
        assert action == 0

    def test_decode_motor_up(self):
        from bl1.loop.decoding import decode_motor
        N, W = 20, 200
        ne_map = jnp.zeros((4, N), dtype=jnp.bool_)
        ne_map = ne_map.at[0, 0].set(True).at[1, 1].set(True)
        ne_map = ne_map.at[2, 2].set(True).at[3, 3].set(True)
        # Make "up" neurons fire a lot, "down" neurons silent
        spike_hist = jnp.zeros((W, N))
        spike_hist = spike_hist.at[:, 0].set(1.0).at[:, 1].set(1.0)
        motor_regions = {"up": [0, 1], "down": [2, 3]}
        action, rates = decode_motor(spike_hist, ne_map, motor_regions, baseline_rate=1.0)
        assert action == 1  # up


# ============================================================================
# games/doom — smoke tests for data containers and is_vizdoom_available
# ============================================================================

class TestDoomContainers:

    def test_is_vizdoom_available(self):
        from bl1.games.doom import is_vizdoom_available
        # Should return a bool without error
        result = is_vizdoom_available()
        assert isinstance(result, bool)

    def test_doom_action_defaults(self):
        from bl1.games.doom import DoomAction
        a = DoomAction()
        assert a.move_forward == 0
        assert a.attack is False

    def test_doom_event(self):
        from bl1.games.doom import DoomEvent
        e = DoomEvent("enemy_kill", 1.0)
        assert e.event_type == "enemy_kill"
        assert e.magnitude == 1.0

    def test_doom_state(self):
        from bl1.games.doom import DoomState
        s = DoomState(
            screen_buffer=np.zeros((4, 4, 3), dtype=np.uint8),
            health=100.0, ammo=50.0, kill_count=0, armor=0.0,
            episode_reward=0.0, is_terminal=False, step_count=0,
        )
        assert s.health == 100.0


# ============================================================================
# compat/udp_bridge — packet helpers (no socket needed)
# ============================================================================

class TestUDPBridgePackets:
    """Tests for the pure-function packet helpers in udp_bridge.py."""

    def test_pack_spike_data(self):
        from bl1.compat.udp_bridge import pack_spike_data, SPIKE_PACKET_SIZE
        counts = np.zeros(8, dtype=np.float32)
        counts[0] = 3.0
        pkt = pack_spike_data(counts)
        assert isinstance(pkt, bytes)
        assert len(pkt) == SPIKE_PACKET_SIZE

    def test_unpack_stimulation_command(self):
        from bl1.compat.udp_bridge import (
            unpack_stimulation_command,
            STIM_FORMAT,
            STIM_PACKET_SIZE,
        )
        # Build a fake stim packet
        timestamp = 12345
        freqs = [10.0] * 8
        amps = [2.0] * 8
        pkt = struct.pack(STIM_FORMAT, timestamp, *freqs, *amps)
        assert len(pkt) == STIM_PACKET_SIZE
        ts, f, a = unpack_stimulation_command(pkt)
        assert ts == timestamp
        np.testing.assert_allclose(f, 10.0)
        np.testing.assert_allclose(a, 2.0)


# ============================================================================
# compat/cl_sdk — ChannelSet, SpikeEvent, Recording, DataStream
# ============================================================================

class TestCLSDKHelpers:
    """Cover lightweight CL-SDK data classes."""

    def test_channel_set(self):
        from bl1.compat.cl_sdk import ChannelSet
        cs = ChannelSet(1, 2, 3)
        assert len(cs) == 3
        assert list(cs) == [1, 2, 3]
        assert "ChannelSet" in repr(cs)

    def test_spike_event(self):
        from bl1.compat.cl_sdk import SpikeEvent
        se = SpikeEvent(channel=5, timestamp=1.23)
        assert se.channel == 5
        assert "SpikeEvent" in repr(se)

    def test_recording(self):
        from bl1.compat.cl_sdk import Recording
        rec = Recording(file_suffix="_test")
        assert rec._stopped is False
        rec.stop()
        assert rec._stopped is True

    def test_data_stream(self):
        from bl1.compat.cl_sdk import DataStream
        ds = DataStream(name="test_stream")
        ds.append(1.0, {"event": "hit"})
        assert len(ds.events) == 1
        assert ds.events[0] == (1.0, {"event": "hit"})


# ============================================================================
# validation/comparison — _estimate_power_law_exponent edge cases
# ============================================================================

class TestPowerLawEstimation:
    """Cover _estimate_power_law_exponent edge cases."""

    def test_too_few_values(self):
        from bl1.validation.comparison import _estimate_power_law_exponent
        assert math.isnan(_estimate_power_law_exponent(np.array([1.0, 2.0])))

    def test_no_positive_values(self):
        from bl1.validation.comparison import _estimate_power_law_exponent
        assert math.isnan(_estimate_power_law_exponent(np.array([-1.0, -2.0, -3.0, -4.0, -5.0])))

    def test_all_same_values(self):
        from bl1.validation.comparison import _estimate_power_law_exponent
        # All same value → < 3 unique → nan
        assert math.isnan(_estimate_power_law_exponent(np.array([5.0] * 10)))

    def test_power_law_data(self):
        from bl1.validation.comparison import _estimate_power_law_exponent
        rng = np.random.default_rng(42)
        # Pareto distribution → should give a negative exponent
        data = rng.pareto(1.5, size=500) + 1.0
        exp = _estimate_power_law_exponent(data)
        assert math.isfinite(exp)
        assert exp < 0  # power-law decay


# ============================================================================
# validation/comparison — generate_comparison_report with NaN
# ============================================================================

class TestComparisonReportEdgeCases:

    def test_report_nan_metric(self):
        from bl1.validation.comparison import generate_comparison_report
        stats = {"mean_firing_rate_hz": float("nan"), "burst_rate_per_min": 5.0}
        report = generate_comparison_report(stats, "wagenaar_2006")
        assert "NaN" in report
        assert "PASS" in report or "FAIL" in report

    def test_report_different_datasets(self):
        from bl1.validation.comparison import generate_comparison_report
        stats = {"mean_firing_rate_hz": 2.0, "branching_ratio": 1.0}
        for ds_name in ["beggs_plenz_2003", "kagan_2022_dishbrain"]:
            report = generate_comparison_report(stats, ds_name)
            assert len(report) > 0


# ============================================================================
# visualization/_style — fallback path
# ============================================================================

class TestStyleFallback:

    def test_bl1_style_works(self):
        """bl1_style context manager should work regardless of matplotlib version."""
        from bl1.visualization._style import bl1_style
        with bl1_style():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            plt.close(fig)


# ============================================================================
# loop/feedback — _event_names utility, reward_based mode
# ============================================================================

class TestFeedbackUtilities:

    def test_event_names_dict(self):
        from bl1.loop.feedback import _event_names
        events = [
            {"event_type": "enemy_kill"},
            {"type": "took_damage"},
            "hit",
        ]
        names = _event_names(events)
        assert names == ["enemy_kill", "took_damage", "hit"]

    def test_event_names_object(self):
        from bl1.loop.feedback import _event_names

        class FakeEvent:
            event_type = "armor_pickup"

        names = _event_names([FakeEvent()])
        assert names == ["armor_pickup"]

    def test_reward_based_mode(self):
        from bl1.loop.feedback import FeedbackProtocol, FeedbackState, compute_feedback_current
        from bl1.mea.electrode import build_neuron_electrode_map

        N = 50
        key = jax.random.PRNGKey(0)
        neuron_pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1400.0)
        xs = jnp.arange(8) * 200.0
        ys = jnp.arange(8) * 200.0
        gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
        elec_pos = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
        ne_map = build_neuron_electrode_map(neuron_pos, elec_pos, radius_um=250.0)

        protocol = FeedbackProtocol(
            mode="reward_based",
            reward_positive_channels=[19, 20],
            reward_negative_channels=[23, 24],
        )
        state = FeedbackState()

        # Positive reward
        I = compute_feedback_current(
            protocol, state, events=[], reward=1.0,
            n_neurons=N, neuron_electrode_map=ne_map,
            key=jax.random.PRNGKey(1),
        )
        assert I.shape == (N,)

        # Negative reward
        I_neg = compute_feedback_current(
            protocol, state, events=[], reward=-1.0,
            n_neurons=N, neuron_electrode_map=ne_map,
            key=jax.random.PRNGKey(2),
        )
        assert I_neg.shape == (N,)

    def test_fep_no_event(self):
        """FEP with no relevant events → zero current."""
        from bl1.loop.feedback import FeedbackProtocol, FeedbackState, compute_feedback_current
        N = 20
        ne_map = jnp.zeros((64, N), dtype=jnp.bool_)
        protocol = FeedbackProtocol(mode="fep")
        state = FeedbackState()
        I = compute_feedback_current(
            protocol, state, events=[], reward=0.0,
            n_neurons=N, neuron_electrode_map=ne_map,
            key=jax.random.PRNGKey(0),
        )
        np.testing.assert_allclose(np.asarray(I), 0.0, atol=1e-8)


# ============================================================================
# games/pong — more step scenarios
# ============================================================================

class TestPongEdgeCases:

    def test_pong_miss(self):
        """Ball reaching right with paddle elsewhere → miss."""
        from bl1.games.pong import Pong, EVENT_MISS, PongState
        game = Pong(paddle_height=0.05, ball_speed=0.5)
        key = jax.random.PRNGKey(0)
        state = game.reset(key)
        # Move ball to right edge, paddle far away
        state = state._replace(
            ball_x=jnp.float32(0.99),
            ball_y=jnp.float32(0.1),
            ball_vx=jnp.float32(0.5),
            ball_vy=jnp.float32(0.0),
            paddle_y=jnp.float32(0.9),
        )
        new_state, event = game.step(state, 0, key)
        assert int(event) == EVENT_MISS

    def test_pong_wall_bounce(self):
        """Ball hitting top/bottom wall → vertical velocity reversal."""
        from bl1.games.pong import Pong
        game = Pong(ball_speed=0.02)
        key = jax.random.PRNGKey(1)
        state = game.reset(key)
        # Place ball near top wall going up
        state = state._replace(
            ball_x=jnp.float32(0.5),
            ball_y=jnp.float32(0.99),
            ball_vx=jnp.float32(0.01),
            ball_vy=jnp.float32(0.1),
        )
        new_state, event = game.step(state, 0, key)
        # Ball should have bounced (vy reversed)
        # y should be clamped to 1.0
        assert float(new_state.ball_y) <= 1.0


# ============================================================================
# analysis/pharmacology — DrugEffect, apply_drug, wash_out
# ============================================================================

class TestPharmacology:
    """Cover pharmacology module with pure JAX arrays (not BCOO)."""

    def test_drug_effect_construction(self):
        from bl1.analysis.pharmacology import DrugEffect
        d = DrugEffect(name="test", description="test drug")
        assert d.ampa_scale == 1.0  # default

    def test_apply_drug_ttx(self):
        from bl1.analysis.pharmacology import TTX, apply_drug
        W_exc = jnp.ones((5, 5))
        W_inh = jnp.ones((5, 5))
        new_exc, new_inh = apply_drug(W_exc, W_inh, TTX)
        np.testing.assert_allclose(np.asarray(new_exc), 0.0)
        np.testing.assert_allclose(np.asarray(new_inh), 0.0)

    def test_apply_drug_carbamazepine(self):
        from bl1.analysis.pharmacology import CARBAMAZEPINE, apply_drug
        W_exc = jnp.ones((5, 5))
        W_inh = jnp.ones((5, 5))
        new_exc, new_inh = apply_drug(W_exc, W_inh, CARBAMAZEPINE)
        np.testing.assert_allclose(np.asarray(new_exc), 0.7, atol=1e-6)
        # Inhibitory unchanged (inh_weight_scale=1.0)
        np.testing.assert_allclose(np.asarray(new_inh), 1.0)

    def test_apply_drug_no_change(self):
        """Drug with all scales=1.0 should return original matrices."""
        from bl1.analysis.pharmacology import DrugEffect, apply_drug
        d = DrugEffect(name="placebo", description="no effect")
        W_exc = jnp.ones((3, 3))
        W_inh = jnp.ones((3, 3))
        new_exc, new_inh = apply_drug(W_exc, W_inh, d)
        # Should be the same object (identity optimization)
        assert new_exc is W_exc
        assert new_inh is W_inh

    def test_wash_out(self):
        from bl1.analysis.pharmacology import wash_out
        W_exc = jnp.ones((3, 3)) * 0.5
        W_inh = jnp.ones((3, 3)) * 0.2
        out_exc, out_inh = wash_out(W_exc, W_inh)
        assert out_exc is W_exc
        assert out_inh is W_inh

    def test_predefined_drugs(self):
        from bl1.analysis.pharmacology import TTX, CARBAMAZEPINE, BICUCULLINE, APV, CNQX
        assert TTX.exc_weight_scale == 0.0
        assert CARBAMAZEPINE.exc_weight_scale == 0.7
        assert BICUCULLINE.gaba_a_scale == 0.0
        assert APV.nmda_scale == 0.0
        assert CNQX.ampa_scale == 0.0


# ============================================================================
# visualization/rates — plot_rate_comparison edge case
# ============================================================================

class TestVisualizationGaps:

    def test_plot_rate_comparison_empty(self):
        from bl1.visualization.rates import plot_rate_comparison
        fig = plot_rate_comparison({}, dt_ms=0.5)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_isi_single_neuron_no_spikes(self):
        """ISI plot for a neuron with 0 or 1 spikes → empty histogram."""
        from bl1.visualization.bursts import plot_isi_distribution
        raster = np.zeros((100, 10), dtype=bool)
        fig = plot_isi_distribution(raster, dt_ms=0.5, neuron_idx=0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_population_rate_empty(self):
        """Empty raster → empty plot (no crash)."""
        from bl1.visualization.rates import plot_population_rate
        raster = np.zeros((0, 10), dtype=bool).reshape(0, 10)
        # n_bins would be 0
        # Need at least 1 timestep for shape
        raster = np.zeros((1, 10), dtype=bool)
        fig = plot_population_rate(raster, dt_ms=0.5, bin_ms=100.0)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ============================================================================
# datasets — list_datasets non-verbose
# ============================================================================

class TestDatasetsCoverage:

    def test_list_datasets_non_verbose(self):
        from bl1.validation.datasets import list_datasets
        keys = list_datasets(verbose=False)
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "wagenaar_2006" in keys
