"""Integration and edge-case tests for BL-1 cortical culture simulator.

Covers gaps in the test suite for:
- core.integrator (simulate function)
- loop.encoding (sensory encoding)
- loop.decoding (motor decoding)
- analysis.criticality (branching ratio, avalanches)
- analysis.bursts (burst detection and statistics)
- analysis.metrics (rally lengths, performance comparison)
- Edge cases (empty rasters, single-neuron networks, large dt)
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    create_population,
    izhikevich_step,
)
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
)
from bl1.core.integrator import simulate, SimulationResult
from bl1.loop.encoding import (
    _place_code,
    _rate_code_frequency,
    encode_sensory,
    FREQ_MIN_HZ,
    FREQ_MAX_HZ,
    N_SENSORY_CHANNELS,
    STIM_AMPLITUDE_MV,
)
from bl1.loop.decoding import decode_motor
from bl1.analysis.criticality import branching_ratio, avalanche_size_distribution
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.analysis.metrics import rally_length, performance_comparison
from bl1.plasticity.stdp import STDPParams, init_stdp_state, stdp_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_network(n: int = 50, seed: int = 0):
    """Build a small dense-weight network for integration tests.

    Returns params, init_state, W_exc, W_inh, is_excitatory.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    params, state, is_excitatory = create_population(k1, n)

    # Simple dense random weight matrices (small values)
    n_exc = int(is_excitatory.sum())
    W = jax.random.uniform(k2, (n, n), minval=0.0, maxval=0.03)
    # Zero diagonal (no self-connections)
    W = W * (1 - jnp.eye(n))

    # Split into E and I based on pre-synaptic identity
    exc_mask = is_excitatory[None, :]  # (1, N) broadcast over rows
    W_exc = W * exc_mask
    W_inh = W * (~is_excitatory)[None, :]

    return params, state, W_exc, W_inh, is_excitatory


class _FakeGameState:
    """Minimal stand-in for PongState with ball_x and ball_y."""

    def __init__(self, ball_x: float, ball_y: float):
        self.ball_x = jnp.float32(ball_x)
        self.ball_y = jnp.float32(ball_y)


class _FakeMEAConfig:
    """Minimal MEA config stand-in for encoding tests."""
    pass


# =========================================================================
# 1. Integrator tests
# =========================================================================


class TestSimulateProducesSpikes:
    def test_simulate_produces_spikes(self):
        """simulate() on a 50-neuron network for 500 steps with external
        current should produce at least some spikes."""
        n = 50
        T = 500
        params, state, W_exc, W_inh, _ = _small_network(n)
        syn_state = create_synapse_state(n)

        # Constant external current strong enough to elicit spikes
        I_ext = jnp.full((T, n), 10.0)

        result = simulate(
            params, state, syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=0.5,
            plasticity_fn=None,
        )

        assert isinstance(result, SimulationResult)
        total_spikes = int(jnp.sum(result.spike_history))
        assert total_spikes > 0, (
            f"Expected spikes with I_ext=10, got {total_spikes} total spikes"
        )


class TestSimulateWithPlasticity:
    def test_simulate_with_plasticity(self):
        """simulate() with a plasticity function should change weights."""
        n = 50
        T = 500
        params, state, W_exc, W_inh, is_excitatory = _small_network(n)
        syn_state = create_synapse_state(n)
        stdp_params = STDPParams()
        stdp_state = init_stdp_state(n)

        # External current to drive spiking
        I_ext = jnp.full((T, n), 12.0)

        W_exc_before = W_exc.copy()

        def plasticity_fn(s_stdp, spikes, w_exc):
            return stdp_update(s_stdp, stdp_params, spikes, w_exc, is_excitatory)

        result = simulate(
            params, state, syn_state,
            stdp_state=stdp_state,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=0.5,
            plasticity_fn=plasticity_fn,
        )

        # Weights should have changed if there were spikes
        total_spikes = int(jnp.sum(result.spike_history))
        if total_spikes > 0:
            weight_diff = float(jnp.sum(jnp.abs(result.final_W_exc - W_exc_before)))
            assert weight_diff > 0, (
                "Weights did not change despite spikes and plasticity enabled"
            )


class TestSimulateShapes:
    def test_simulate_shapes(self):
        """spike_history should be (T, N) boolean."""
        n = 30
        T = 200
        params, state, W_exc, W_inh, _ = _small_network(n, seed=1)
        syn_state = create_synapse_state(n)

        I_ext = jnp.full((T, n), 5.0)

        result = simulate(
            params, state, syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=0.5,
            plasticity_fn=None,
        )

        assert result.spike_history.shape == (T, n), (
            f"Expected shape ({T}, {n}), got {result.spike_history.shape}"
        )
        assert result.spike_history.dtype == jnp.bool_, (
            f"Expected bool dtype, got {result.spike_history.dtype}"
        )


class TestSimulateNoPlasticity:
    def test_simulate_no_plasticity(self):
        """When plasticity_fn=None, simulate should still work without error."""
        n = 20
        T = 100
        params, state, W_exc, W_inh, _ = _small_network(n, seed=2)
        syn_state = create_synapse_state(n)

        I_ext = jnp.zeros((T, n))

        result = simulate(
            params, state, syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=0.5,
            plasticity_fn=None,
        )

        assert result.spike_history.shape == (T, n)
        # With zero input, expect no or very few spikes
        total_spikes = int(jnp.sum(result.spike_history))
        assert total_spikes == 0, (
            f"Expected no spikes with zero input, got {total_spikes}"
        )


# =========================================================================
# 2. Encoding tests
# =========================================================================


class TestEncodeSensoryPlaceCoding:
    def test_ball_at_bottom_activates_channel_0(self):
        """Ball at y=0.1 should activate channel 0."""
        active = _place_code(0.1, N_SENSORY_CHANNELS)
        assert 0 in active, f"Expected channel 0 to be active, got {active}"

    def test_ball_at_top_activates_channel_7(self):
        """Ball at y=0.9 should activate channel 7."""
        active = _place_code(0.9, N_SENSORY_CHANNELS)
        assert 7 in active, f"Expected channel 7 to be active, got {active}"

    def test_ball_at_middle_activates_channel_4(self):
        """Ball at y=0.5 should activate channel 4."""
        active = _place_code(0.5, N_SENSORY_CHANNELS)
        assert 4 in active, f"Expected channel 4 to be active, got {active}"

    def test_channels_span_full_range(self):
        """Sweeping ball_y from 0 to 1 should activate all 8 channels."""
        activated = set()
        for y in np.linspace(0.0, 0.99, 100):
            channels = _place_code(float(y), N_SENSORY_CHANNELS)
            activated.update(channels)
        assert activated == set(range(8)), (
            f"Not all channels activated, got {activated}"
        )


class TestEncodeSensoryRateCoding:
    def test_far_ball_low_frequency(self):
        """Ball at x=0.0 (far) should have low frequency."""
        freq = _rate_code_frequency(0.0)
        npt.assert_allclose(freq, FREQ_MIN_HZ, atol=0.1)

    def test_close_ball_high_frequency(self):
        """Ball at x=1.0 (close) should have high frequency."""
        freq = _rate_code_frequency(1.0)
        npt.assert_allclose(freq, FREQ_MAX_HZ, atol=0.1)

    def test_frequency_monotonic(self):
        """Frequency should increase monotonically with ball_x."""
        xs = np.linspace(0.0, 1.0, 20)
        freqs = [_rate_code_frequency(float(x)) for x in xs]
        for i in range(1, len(freqs)):
            assert freqs[i] >= freqs[i - 1], (
                f"Frequency decreased: f({xs[i]})={freqs[i]} < f({xs[i-1]})={freqs[i-1]}"
            )


class TestEncodeSensoryPhaseContinuity:
    def test_phase_accumulator_carries(self):
        """Phase accumulator should carry between calls."""
        game_state = _FakeGameState(ball_x=0.5, ball_y=0.5)
        sensory_channels = list(range(8))
        mea_config = _FakeMEAConfig()

        # First call: initial phase = 0
        _, _, phase1 = encode_sensory(
            game_state, mea_config, sensory_channels,
            dt_game_ms=20.0, _phase_accumulator=0.0,
        )
        assert phase1 > 0.0, "Phase should advance after a call"

        # Second call: pass phase1 back
        _, _, phase2 = encode_sensory(
            game_state, mea_config, sensory_channels,
            dt_game_ms=20.0, _phase_accumulator=phase1,
        )
        assert phase2 > phase1, "Phase should continue advancing"

    def test_pulse_eventually_fires(self):
        """Calling encode_sensory repeatedly should eventually produce a pulse."""
        game_state = _FakeGameState(ball_x=1.0, ball_y=0.5)
        sensory_channels = list(range(8))
        mea_config = _FakeMEAConfig()

        phase = 0.0
        pulse_fired = False
        for _ in range(100):
            active, amp, phase = encode_sensory(
                game_state, mea_config, sensory_channels,
                dt_game_ms=20.0, _phase_accumulator=phase,
            )
            if amp > 0.0:
                pulse_fired = True
                assert len(active) > 0
                npt.assert_allclose(amp, STIM_AMPLITUDE_MV)
                break

        assert pulse_fired, "No pulse fired in 100 game steps"


# =========================================================================
# 3. Decoding tests
# =========================================================================


def _make_decoding_fixtures(n_neurons=20, n_electrodes=16, window_steps=200):
    """Build fixtures for motor decoding tests.

    Returns spike_window, neuron_electrode_map, motor_regions.
    """
    # Simple neuron-electrode map: each electrode sees 1-2 nearby neurons
    ne_map = jnp.zeros((n_electrodes, n_neurons), dtype=jnp.bool_)
    # Electrodes 8-11 ("up") see neurons 0-3
    ne_map = ne_map.at[8, 0].set(True)
    ne_map = ne_map.at[9, 1].set(True)
    ne_map = ne_map.at[10, 2].set(True)
    ne_map = ne_map.at[11, 3].set(True)
    # Electrodes 12-15 ("down") see neurons 4-7
    ne_map = ne_map.at[12, 4].set(True)
    ne_map = ne_map.at[13, 5].set(True)
    ne_map = ne_map.at[14, 6].set(True)
    ne_map = ne_map.at[15, 7].set(True)

    motor_regions = {
        "up": [8, 9, 10, 11],
        "down": [12, 13, 14, 15],
    }

    return ne_map, motor_regions


class TestDecodeMotorUp:
    def test_more_spikes_in_up_region(self):
        """More spikes in 'up' region should produce action=1."""
        n_neurons = 20
        window = 200
        ne_map, motor_regions = _make_decoding_fixtures(n_neurons, window_steps=window)

        # Create spike history: "up" neurons (0-3) fire heavily
        spikes = jnp.zeros((window, n_neurons), dtype=jnp.float32)
        # Neurons 0-3 spike every other step (very high rate)
        spikes = spikes.at[::2, 0].set(1.0)
        spikes = spikes.at[::2, 1].set(1.0)
        spikes = spikes.at[::2, 2].set(1.0)
        spikes = spikes.at[::2, 3].set(1.0)

        action, rates = decode_motor(
            spikes, ne_map, motor_regions, baseline_rate=5.0,
        )

        assert action == 1, f"Expected action=1 (up), got {action}"
        assert rates["up"] > rates["down"], (
            f"Up rate ({rates['up']}) should exceed down rate ({rates['down']})"
        )


class TestDecodeMotorDown:
    def test_more_spikes_in_down_region(self):
        """More spikes in 'down' region should produce action=2."""
        n_neurons = 20
        window = 200
        ne_map, motor_regions = _make_decoding_fixtures(n_neurons, window_steps=window)

        # Create spike history: "down" neurons (4-7) fire heavily
        spikes = jnp.zeros((window, n_neurons), dtype=jnp.float32)
        spikes = spikes.at[::2, 4].set(1.0)
        spikes = spikes.at[::2, 5].set(1.0)
        spikes = spikes.at[::2, 6].set(1.0)
        spikes = spikes.at[::2, 7].set(1.0)

        action, rates = decode_motor(
            spikes, ne_map, motor_regions, baseline_rate=5.0,
        )

        assert action == 2, f"Expected action=2 (down), got {action}"
        assert rates["down"] > rates["up"], (
            f"Down rate ({rates['down']}) should exceed up rate ({rates['up']})"
        )


class TestDecodeMotorStay:
    def test_no_spikes_returns_stay(self):
        """No spikes should produce action=0 (stay)."""
        n_neurons = 20
        window = 200
        ne_map, motor_regions = _make_decoding_fixtures(n_neurons, window_steps=window)

        spikes = jnp.zeros((window, n_neurons), dtype=jnp.float32)

        action, rates = decode_motor(
            spikes, ne_map, motor_regions, baseline_rate=20.0,
        )

        assert action == 0, f"Expected action=0 (stay), got {action}"

    def test_empty_window_returns_stay(self):
        """Zero-length spike window should return action=0."""
        n_neurons = 20
        ne_map, motor_regions = _make_decoding_fixtures(n_neurons)

        spikes = jnp.zeros((0, n_neurons), dtype=jnp.float32)

        action, rates = decode_motor(
            spikes, ne_map, motor_regions, baseline_rate=20.0,
        )

        assert action == 0, f"Expected action=0 with empty window, got {action}"


# =========================================================================
# 4. Analysis tests
# =========================================================================


class TestBranchingRatioCritical:
    def test_branching_ratio_near_one(self):
        """Synthetic raster with constant activity should yield sigma ~ 1.0."""
        # Every bin has approximately the same number of spikes -> sigma ~ 1
        T = 2000
        N = 50
        rng = np.random.RandomState(42)
        # Each neuron fires with ~5% probability each step
        raster = rng.rand(T, N) < 0.05

        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=4.0)
        assert not np.isnan(sigma), "Branching ratio should not be NaN"
        # For a stationary Poisson process sigma should be close to 1.0
        assert 0.7 < sigma < 1.3, (
            f"Expected sigma near 1.0 for stationary activity, got {sigma:.3f}"
        )


class TestBranchingRatioSubcritical:
    def test_branching_ratio_below_one(self):
        """Raster with long silent periods should yield sigma < 1."""
        T = 2000
        N = 50
        raster = np.zeros((T, N), dtype=bool)
        # Short bursts of activity separated by long silence
        for start in range(0, T, 200):
            raster[start:start + 5, :10] = True  # 5 steps of activity, then 195 silent

        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=4.0)
        # Many bins go from active to silent, so descendants/ancestors < 1
        if not np.isnan(sigma):
            assert sigma < 1.0, (
                f"Expected sigma < 1 for bursty-then-silent pattern, got {sigma:.3f}"
            )


class TestAvalancheDetection:
    def test_known_avalanche_pattern(self):
        """Known avalanche pattern: two bursts separated by silence."""
        T = 400
        N = 20
        raster = np.zeros((T, N), dtype=bool)

        # Avalanche 1: steps 0-31 (4 bins of 8 steps each), 3 neurons active
        # Total spikes: 4 bins * 8 steps * 3 neurons = 96
        raster[0:32, :3] = True

        # Silent gap: steps 32-127 (12 silent bins)

        # Avalanche 2: steps 128-191 (8 bins of 8 steps each), 10 neurons active
        # Total spikes: 8 bins * 8 steps * 10 neurons = 640
        raster[128:192, :10] = True

        sizes, durations = avalanche_size_distribution(raster, dt_ms=0.5, bin_ms=4.0)

        assert len(sizes) >= 2, f"Expected at least 2 avalanches, got {len(sizes)}"
        assert len(durations) >= 2
        assert all(s > 0 for s in sizes), "All avalanche sizes should be positive"
        assert all(d > 0 for d in durations), "All durations should be positive"

        # First avalanche should be smaller than second (fewer neurons, shorter)
        assert sizes[0] < sizes[1], (
            f"First avalanche size ({sizes[0]}) should be smaller than second ({sizes[1]})"
        )
        # First duration should be shorter than second
        assert durations[0] < durations[1], (
            f"First duration ({durations[0]}) should be shorter than second ({durations[1]})"
        )


class TestBurstDetection:
    def test_synthetic_burst_raster(self):
        """Synthetic burst raster should detect bursts in correct time ranges."""
        T = 4000  # 2000 ms at dt=0.5
        N = 100
        raster = np.zeros((T, N), dtype=bool)

        # Background: low random activity
        rng = np.random.RandomState(7)
        raster = rng.rand(T, N) < 0.005  # ~0.5% per step

        # Burst 1: steps 400-600 (200-300 ms), all neurons fire at high rate
        raster[400:600, :] = rng.rand(200, N) < 0.5

        # Burst 2: steps 2000-2200 (1000-1100 ms)
        raster[2000:2200, :] = rng.rand(200, N) < 0.5

        bursts = detect_bursts(raster, dt_ms=0.5, threshold_std=2.0, min_duration_ms=30.0)

        assert len(bursts) >= 2, f"Expected at least 2 bursts, got {len(bursts)}"

        # Check that burst times are in the expected ranges
        burst_starts = [b[0] for b in bursts]
        burst_ends = [b[1] for b in bursts]

        # At least one burst should start near 200 ms
        found_burst1 = any(150 < s < 350 for s in burst_starts)
        # At least one burst should start near 1000 ms
        found_burst2 = any(900 < s < 1150 for s in burst_starts)

        assert found_burst1, f"Expected a burst near 200ms, burst starts: {burst_starts}"
        assert found_burst2, f"Expected a burst near 1000ms, burst starts: {burst_starts}"


class TestBurstStatisticsValues:
    def test_burst_statistics_computation(self):
        """Check IBI, duration, recruitment computations from known bursts."""
        # Manually construct bursts: (start_ms, end_ms, n_spikes, frac_recruited)
        bursts = [
            (100.0, 200.0, 500, 0.8),
            (500.0, 600.0, 600, 0.9),
            (1000.0, 1150.0, 700, 0.85),
        ]

        stats = burst_statistics(bursts)

        # Duration mean: (100 + 100 + 150) / 3 = 116.67
        npt.assert_allclose(stats["duration_mean"], 116.667, atol=0.1)

        # Recruitment mean: (0.8 + 0.9 + 0.85) / 3 = 0.85
        npt.assert_allclose(stats["recruitment_mean"], 0.85, atol=1e-5)

        # IBI (onset-to-onset): [500-100, 1000-500] = [400, 500]
        # Mean IBI = 450
        npt.assert_allclose(stats["ibi_mean"], 450.0, atol=1e-5)

        # IBI CV = std([400, 500]) / mean([400, 500]) = 50/450
        expected_cv = np.std([400, 500]) / np.mean([400, 500])
        npt.assert_allclose(stats["ibi_cv"], expected_cv, atol=1e-4)

        # Burst rate: (3 - 1) / ((1000 - 100) / 1000) = 2 / 0.9 = 2.222
        npt.assert_allclose(stats["burst_rate"], 2.222, atol=0.01)

    def test_empty_burst_list(self):
        """Empty burst list should return NaN/zero defaults."""
        stats = burst_statistics([])
        assert np.isnan(stats["ibi_mean"])
        assert np.isnan(stats["duration_mean"])
        assert stats["burst_rate"] == 0.0

    def test_single_burst(self):
        """Single burst should compute duration but IBI should be NaN."""
        bursts = [(100.0, 250.0, 300, 0.7)]
        stats = burst_statistics(bursts)

        npt.assert_allclose(stats["duration_mean"], 150.0)
        npt.assert_allclose(stats["recruitment_mean"], 0.7)
        assert np.isnan(stats["ibi_mean"])
        assert stats["burst_rate"] == 0.0


class TestRallyLengthExtraction:
    def test_known_event_sequence(self):
        """Known event sequence should produce correct rally lengths."""
        events = [
            (100.0, "hit"),
            (200.0, "hit"),
            (300.0, "hit"),
            (400.0, "miss"),
            (500.0, "hit"),
            (600.0, "miss"),
            (700.0, "miss"),
        ]

        rallies = rally_length(events)
        expected = np.array([3, 1, 0], dtype=np.int32)
        npt.assert_array_equal(rallies, expected)

    def test_empty_events(self):
        """Empty event list should return empty array."""
        rallies = rally_length([])
        assert len(rallies) == 0

    def test_only_hits_no_miss(self):
        """Hits without a terminating miss should still record the rally."""
        events = [
            (100.0, "hit"),
            (200.0, "hit"),
            (300.0, "hit"),
        ]
        rallies = rally_length(events)
        assert len(rallies) == 1
        assert rallies[0] == 3

    def test_only_misses(self):
        """Only misses should produce zero-length rallies."""
        events = [
            (100.0, "miss"),
            (200.0, "miss"),
        ]
        rallies = rally_length(events)
        npt.assert_array_equal(rallies, np.array([0, 0], dtype=np.int32))


class TestPerformanceComparison:
    def test_three_conditions(self):
        """Three conditions should produce correct means and comparisons."""
        results = {
            "closed_loop": {
                "game_events": [
                    (100, "hit"), (200, "hit"), (300, "miss"),
                    (400, "hit"), (500, "hit"), (600, "hit"), (700, "miss"),
                ],
            },
            "open_loop": {
                "game_events": [
                    (100, "hit"), (200, "miss"),
                    (300, "hit"), (400, "miss"),
                ],
            },
            "silent": {
                "game_events": [
                    (100, "miss"),
                    (200, "miss"),
                    (300, "miss"),
                ],
            },
        }

        comparison = performance_comparison(results)

        # closed_loop rallies: [2, 3], mean=2.5
        npt.assert_allclose(comparison["mean_rally"]["closed_loop"], 2.5)
        # open_loop rallies: [1, 1], mean=1.0
        npt.assert_allclose(comparison["mean_rally"]["open_loop"], 1.0)
        # silent rallies: [0, 0, 0], mean=0.0
        npt.assert_allclose(comparison["mean_rally"]["silent"], 0.0)

        assert comparison["n_rallies"]["closed_loop"] == 2
        assert comparison["n_rallies"]["open_loop"] == 2
        assert comparison["n_rallies"]["silent"] == 3

    def test_rally_lengths_input(self):
        """Should also accept pre-computed rally_lengths arrays."""
        results = {
            "a": {"rally_lengths": np.array([3, 5, 2])},
            "b": {"rally_lengths": np.array([1, 1, 1])},
        }

        comparison = performance_comparison(results)

        npt.assert_allclose(comparison["mean_rally"]["a"], 10.0 / 3)
        npt.assert_allclose(comparison["mean_rally"]["b"], 1.0)


# =========================================================================
# 5. Edge cases
# =========================================================================


class TestEmptySpikeRaster:
    def test_branching_ratio_empty(self):
        """Zero-length raster should return NaN, not crash."""
        raster = np.zeros((0, 10), dtype=bool)
        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=4.0)
        assert np.isnan(sigma)

    def test_avalanche_empty(self):
        """Zero-length raster should return empty arrays."""
        raster = np.zeros((0, 10), dtype=bool)
        sizes, durations = avalanche_size_distribution(raster, dt_ms=0.5, bin_ms=4.0)
        assert len(sizes) == 0
        assert len(durations) == 0

    def test_burst_detection_empty(self):
        """Zero-length raster should return empty burst list."""
        raster = np.zeros((0, 10), dtype=bool)
        bursts = detect_bursts(raster, dt_ms=0.5)
        assert len(bursts) == 0

    def test_silent_raster(self):
        """All-zero raster (no spikes at all) should not crash analysis."""
        raster = np.zeros((1000, 50), dtype=bool)

        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=4.0)
        assert np.isnan(sigma), "Branching ratio of silent raster should be NaN"

        sizes, durations = avalanche_size_distribution(raster, dt_ms=0.5, bin_ms=4.0)
        assert len(sizes) == 0

        bursts = detect_bursts(raster, dt_ms=0.5)
        assert len(bursts) == 0


class TestSingleNeuronNetwork:
    def test_single_neuron_simulate(self):
        """1-neuron network should work in simulate()."""
        n = 1
        T = 200
        params = IzhikevichParams(
            a=jnp.array([0.02]),
            b=jnp.array([0.2]),
            c=jnp.array([-65.0]),
            d=jnp.array([8.0]),
        )
        state = NeuronState(
            v=jnp.array([-65.0]),
            u=jnp.array([0.2 * -65.0]),
            spikes=jnp.array([False]),
        )
        syn_state = create_synapse_state(n)
        W_exc = jnp.zeros((1, 1))
        W_inh = jnp.zeros((1, 1))

        I_ext = jnp.full((T, 1), 15.0)

        result = simulate(
            params, state, syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=0.5,
            plasticity_fn=None,
        )

        assert result.spike_history.shape == (T, 1)
        total_spikes = int(jnp.sum(result.spike_history))
        assert total_spikes > 0, "Single neuron with I=15 should spike"

    def test_single_neuron_analysis(self):
        """Analysis functions should work with 1-neuron rasters."""
        raster = np.zeros((500, 1), dtype=bool)
        # Neuron fires every 20 steps
        raster[::20, 0] = True

        sigma = branching_ratio(raster, dt_ms=0.5, bin_ms=4.0)
        assert not np.isnan(sigma), "Branching ratio should be computable"

        sizes, durations = avalanche_size_distribution(raster, dt_ms=0.5, bin_ms=4.0)
        assert len(sizes) > 0, "Should detect avalanches in periodic spiking"


class TestLargeDt:
    def test_large_dt_simulation(self):
        """dt=1.0 instead of 0.5 should still produce a valid simulation."""
        n = 30
        T = 200
        params, state, W_exc, W_inh, _ = _small_network(n, seed=5)
        syn_state = create_synapse_state(n)

        I_ext = jnp.full((T, n), 10.0)

        result = simulate(
            params, state, syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_ext,
            dt=1.0,
            plasticity_fn=None,
        )

        assert result.spike_history.shape == (T, n)
        # Should still produce spikes (though timing will differ from dt=0.5)
        total_spikes = int(jnp.sum(result.spike_history))
        assert total_spikes > 0, (
            f"Expected spikes with dt=1.0 and I=10, got {total_spikes}"
        )

    def test_large_dt_branching_ratio(self):
        """Branching ratio computation should work with dt=1.0."""
        rng = np.random.RandomState(99)
        raster = rng.rand(1000, 20) < 0.05

        sigma = branching_ratio(raster, dt_ms=1.0, bin_ms=4.0)
        assert not np.isnan(sigma)
        assert 0.5 < sigma < 2.0, f"Sigma should be reasonable, got {sigma}"
