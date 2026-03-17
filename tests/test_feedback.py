"""Tests for the rich feedback protocol (bl1.loop.feedback)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.loop.feedback import (
    EventFeedbackConfig,
    FeedbackProtocol,
    FeedbackState,
    compute_feedback_current,
    create_dishbrain_pong_protocol,
    create_doom_feedback_protocol,
)
from bl1.mea.electrode import build_neuron_electrode_map


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NEURONS = 100
N_ELECTRODES = 64
KEY = jax.random.PRNGKey(42)


def _make_electrode_map():
    """Build a small (64, 100) neuron-electrode map for testing.

    Places 100 neurons uniformly on [0, 1400] x [0, 1400] and 64
    electrodes on an 8x8 grid with 200 um spacing starting at (0, 0).
    Uses a generous detection radius so every electrode sees some neurons.
    """
    key = jax.random.PRNGKey(0)
    neuron_positions = jax.random.uniform(key, (N_NEURONS, 2), minval=0.0, maxval=1400.0)

    # 8x8 grid, 200 um spacing
    xs = jnp.arange(8) * 200.0
    ys = jnp.arange(8) * 200.0
    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    electrode_positions = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

    return build_neuron_electrode_map(neuron_positions, electrode_positions, radius_um=250.0)


ELECTRODE_MAP = _make_electrode_map()


# ---------------------------------------------------------------------------
# FEP mode tests
# ---------------------------------------------------------------------------


def test_fep_predictable():
    """Positive event with FEP protocol produces uniform current."""
    protocol = FeedbackProtocol(mode="fep", fep_predictable_amplitude=2.0)
    state = FeedbackState()

    I = compute_feedback_current(
        protocol, state,
        events=["hit"],
        reward=1.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
    )

    assert I.shape == (N_NEURONS,)
    # All values should be exactly the predictable amplitude
    np.testing.assert_allclose(I, 2.0, atol=1e-6)


def test_fep_unpredictable():
    """Negative event with FEP protocol produces random current in range."""
    protocol = FeedbackProtocol(
        mode="fep",
        fep_unpredictable_amplitude_range=(-5.0, 15.0),
    )
    state = FeedbackState()

    I = compute_feedback_current(
        protocol, state,
        events=["miss"],
        reward=-1.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
    )

    assert I.shape == (N_NEURONS,)
    # Values should lie within the unpredictable range
    assert float(jnp.min(I)) >= -5.0
    assert float(jnp.max(I)) <= 15.0
    # Should NOT be uniform (with overwhelming probability)
    assert float(jnp.std(I)) > 0.1, "Unpredictable current should have variance"


def test_fep_silent():
    """Silent mode always returns zero current."""
    protocol = FeedbackProtocol(mode="silent")
    state = FeedbackState()

    I = compute_feedback_current(
        protocol, state,
        events=["hit", "miss"],
        reward=10.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
    )

    assert I.shape == (N_NEURONS,)
    np.testing.assert_allclose(I, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Event-based mode tests
# ---------------------------------------------------------------------------


def _make_event_protocol():
    """Small event-based protocol using channels within our 64-electrode map."""
    return FeedbackProtocol(
        mode="event_based",
        event_configs={
            "enemy_kill": EventFeedbackConfig(
                channels=[2, 3, 4],
                base_frequency=20.0,
                base_amplitude=2.5,
                base_pulses=40,
                td_sign="positive",
            ),
            "took_damage": EventFeedbackConfig(
                channels=[10, 11, 12],
                base_frequency=90.0,
                base_amplitude=2.2,
                base_pulses=50,
                td_sign="negative",
                unpredictable=True,
            ),
        },
        reward_positive_channels=[20, 21],
        reward_negative_channels=[30, 31],
        surprise_scaling_enabled=True,
        surprise_ema_beta=0.99,
    )


def test_event_based_positive():
    """'enemy_kill' event injects current on channels [2, 3, 4] only."""
    protocol = _make_event_protocol()
    state = FeedbackState()
    state.surprise_ema = 1.0  # pre-seed so normalisation is stable

    I = compute_feedback_current(
        protocol, state,
        events=["enemy_kill"],
        reward=0.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=1.0,
    )

    assert I.shape == (N_NEURONS,)

    # Neurons near channels 2, 3, 4 should receive current
    target_mask = (
        ELECTRODE_MAP[2].astype(jnp.float32)
        + ELECTRODE_MAP[3].astype(jnp.float32)
        + ELECTRODE_MAP[4].astype(jnp.float32)
    )
    has_target = target_mask > 0
    # At least some target neurons got nonzero current
    assert float(jnp.sum(jnp.where(has_target, jnp.abs(I), 0.0))) > 0.0

    # Neurons with no coverage from channels 2/3/4 AND no reward channels
    # should be zero
    reward_mask = (
        ELECTRODE_MAP[20].astype(jnp.float32)
        + ELECTRODE_MAP[21].astype(jnp.float32)
    )
    any_mask = (target_mask + reward_mask) > 0
    off_target = ~any_mask
    np.testing.assert_allclose(
        np.asarray(I[off_target]),
        0.0,
        atol=1e-6,
        err_msg="Off-target neurons should receive zero current",
    )


def test_event_based_negative_unpredictable():
    """'took_damage' event produces noisy current on channels [10, 11, 12]."""
    protocol = _make_event_protocol()
    state = FeedbackState()
    state.surprise_ema = 1.0

    I = compute_feedback_current(
        protocol, state,
        events=["took_damage"],
        reward=0.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=1.0,
    )

    assert I.shape == (N_NEURONS,)

    # Neurons on channels 10/11/12 should get nonzero, noisy current
    target_mask = (
        ELECTRODE_MAP[10].astype(jnp.float32)
        + ELECTRODE_MAP[11].astype(jnp.float32)
        + ELECTRODE_MAP[12].astype(jnp.float32)
    )
    has_target = target_mask > 0
    target_values = I[has_target]

    if target_values.size > 1:
        # Noise means values should not all be identical
        assert float(jnp.std(target_values)) > 0.01, (
            "Unpredictable feedback should produce non-uniform current"
        )


def test_surprise_scaling():
    """Higher TD-error magnitude produces larger stimulation amplitude."""
    protocol = _make_event_protocol()

    # Low surprise run
    state_lo = FeedbackState()
    state_lo.surprise_ema = 1.0
    I_lo = compute_feedback_current(
        protocol, state_lo,
        events=["enemy_kill"],
        reward=0.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=0.5,
    )

    # High surprise run
    state_hi = FeedbackState()
    state_hi.surprise_ema = 1.0
    I_hi = compute_feedback_current(
        protocol, state_hi,
        events=["enemy_kill"],
        reward=0.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=5.0,
    )

    # High-surprise run should have equal or larger total current
    assert float(jnp.sum(jnp.abs(I_hi))) >= float(jnp.sum(jnp.abs(I_lo))), (
        "Higher TD-error should produce equal or larger total feedback current"
    )


# ---------------------------------------------------------------------------
# Reward-based mode tests
# ---------------------------------------------------------------------------


def test_reward_feedback_positive():
    """Positive reward in event_based mode injects on positive channels."""
    protocol = _make_event_protocol()
    state = FeedbackState()
    state.surprise_ema = 1.0

    I = compute_feedback_current(
        protocol, state,
        events=[],  # no named events, just reward
        reward=1.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=0.0,
    )

    # Positive reward channels [20, 21] should have current
    pos_mask = (
        ELECTRODE_MAP[20].astype(jnp.float32)
        + ELECTRODE_MAP[21].astype(jnp.float32)
    )
    has_pos = pos_mask > 0
    assert float(jnp.sum(jnp.where(has_pos, jnp.abs(I), 0.0))) > 0.0


def test_reward_feedback_negative():
    """Negative reward in event_based mode injects noisy current on negative channels."""
    protocol = _make_event_protocol()
    state = FeedbackState()
    state.surprise_ema = 1.0

    I = compute_feedback_current(
        protocol, state,
        events=[],
        reward=-1.0,
        n_neurons=N_NEURONS,
        neuron_electrode_map=ELECTRODE_MAP,
        key=KEY,
        td_error=0.0,
    )

    # Negative reward channels [30, 31] should have current
    neg_mask = (
        ELECTRODE_MAP[30].astype(jnp.float32)
        + ELECTRODE_MAP[31].astype(jnp.float32)
    )
    has_neg = neg_mask > 0
    neg_values = I[has_neg]

    if neg_values.size > 1:
        assert float(jnp.sum(jnp.abs(neg_values))) > 0.0, (
            "Negative reward should produce nonzero current on negative channels"
        )


# ---------------------------------------------------------------------------
# Factory / protocol tests
# ---------------------------------------------------------------------------


def test_create_doom_protocol():
    """Verify doom protocol has correct event configs and mode."""
    protocol = create_doom_feedback_protocol()

    assert protocol.mode == "event_based"
    assert "enemy_kill" in protocol.event_configs
    assert "armor_pickup" in protocol.event_configs
    assert "took_damage" in protocol.event_configs
    assert "ammo_waste" in protocol.event_configs

    # Check enemy_kill config details
    ek = protocol.event_configs["enemy_kill"]
    assert ek.channels == [35, 36, 38]
    assert ek.base_amplitude == 2.5
    assert ek.td_sign == "positive"
    assert ek.unpredictable is False

    # took_damage should be unpredictable
    td = protocol.event_configs["took_damage"]
    assert td.unpredictable is True
    assert td.channels == [44, 47, 48]

    # Episode feedback should be enabled
    assert protocol.episode_feedback_enabled is True


def test_create_dishbrain_protocol():
    """Verify DishBrain protocol is FEP mode with canonical values."""
    protocol = create_dishbrain_pong_protocol()

    assert protocol.mode == "fep"
    assert protocol.fep_predictable_amplitude == 2.0
    assert protocol.fep_unpredictable_amplitude_range == (-5.0, 15.0)


# ---------------------------------------------------------------------------
# Feedback state tests
# ---------------------------------------------------------------------------


def test_feedback_state_surprise_ema():
    """Surprise EMA updates correctly with new TD-error observations."""
    state = FeedbackState()
    assert state.surprise_ema == 0.0

    # First update: EMA = 0.99 * 0.0 + 0.01 * |5.0| = 0.05
    state.update_surprise(5.0, beta=0.99)
    np.testing.assert_allclose(state.surprise_ema, 0.05, atol=1e-10)

    # Second update: EMA = 0.99 * 0.05 + 0.01 * |3.0| = 0.0495 + 0.03 = 0.0795
    state.update_surprise(3.0, beta=0.99)
    np.testing.assert_allclose(state.surprise_ema, 0.0795, atol=1e-10)

    # Negative TD-error: uses abs value
    state.update_surprise(-10.0, beta=0.99)
    expected = 0.99 * 0.0795 + 0.01 * 10.0  # 0.078705 + 0.1 = 0.178705
    np.testing.assert_allclose(state.surprise_ema, expected, atol=1e-10)

    # Reset episode should clear reward sum but keep surprise EMA
    state.episode_reward_sum = 42.0
    state.reset_episode()
    assert state.episode_reward_sum == 0.0
    assert state.surprise_ema > 0.0  # EMA persists across episodes
