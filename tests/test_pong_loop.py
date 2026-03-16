"""Tests for Pong game and closed-loop integration (bl1.games.pong, bl1.loop.controller)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.games.pong import Pong, PongState, EVENT_HIT, EVENT_MISS


# ---------------------------------------------------------------------------
# Pong game tests
# ---------------------------------------------------------------------------

def test_pong_reset():
    """After reset, ball should be at x~0, paddle at y=0.5."""
    game = Pong()
    key = jax.random.PRNGKey(0)
    state = game.reset(key)

    np.testing.assert_allclose(float(state.ball_x), 0.0, atol=1e-5)
    np.testing.assert_allclose(float(state.paddle_y), 0.5, atol=1e-5)


def test_pong_ball_moves():
    """After step with action=0, ball_x should increase (ball moves right)."""
    game = Pong()
    key = jax.random.PRNGKey(1)
    state = game.reset(key)

    initial_x = float(state.ball_x)
    new_state, _ = game.step(state, action=0, key=key)

    assert float(new_state.ball_x) > initial_x, (
        "Ball should move rightward after a step"
    )


def test_pong_paddle_up():
    """Action=1 should move paddle up (increase paddle_y)."""
    game = Pong()
    key = jax.random.PRNGKey(2)
    state = game.reset(key)

    initial_paddle_y = float(state.paddle_y)
    new_state, _ = game.step(state, action=1, key=key)

    assert float(new_state.paddle_y) > initial_paddle_y, (
        "Paddle should move up (increase paddle_y) with action=1"
    )


def test_pong_paddle_down():
    """Action=2 should move paddle down (decrease paddle_y)."""
    game = Pong()
    key = jax.random.PRNGKey(3)
    state = game.reset(key)

    initial_paddle_y = float(state.paddle_y)
    new_state, _ = game.step(state, action=2, key=key)

    assert float(new_state.paddle_y) < initial_paddle_y, (
        "Paddle should move down (decrease paddle_y) with action=2"
    )


def test_pong_hit_detection():
    """Track the ball with the paddle and verify a hit event.

    Event should be "hit" (EVENT_HIT = 1).
    """
    game = Pong(ball_speed=0.01, paddle_speed=0.03)
    key = jax.random.PRNGKey(10)
    state = game.reset(key)

    # Actively track the ball with the paddle
    event = 0
    for i in range(200):
        key, subkey = jax.random.split(key)
        # Move paddle toward ball
        if float(state.ball_y) > float(state.paddle_y) + 0.01:
            action = 1  # up
        elif float(state.ball_y) < float(state.paddle_y) - 0.01:
            action = 2  # down
        else:
            action = 0  # stay
        state, event = game.step(state, action, key=subkey)
        if int(event) != 0:
            break

    assert int(event) == EVENT_HIT, (
        f"Expected hit (event={EVENT_HIT}), got event={int(event)}"
    )


def test_pong_miss_detection():
    """Move paddle far from ball's y position and advance until ball reaches right side.

    Event should be "miss" (EVENT_MISS = 2).
    """
    game = Pong(ball_speed=0.01)
    key = jax.random.PRNGKey(20)
    state = game.reset(key)

    # Move paddle to y=0.0 (far from ball at y=0.5) by stepping with action=2
    for _ in range(50):
        key, subkey = jax.random.split(key)
        state, _ = game.step(state, action=2, key=subkey)

    # Now advance the ball without moving the paddle until it reaches the right side
    event = 0
    for i in range(200):
        key, subkey = jax.random.split(key)
        state, event = game.step(state, action=0, key=subkey)
        if int(event) != 0:
            break

    assert int(event) == EVENT_MISS, (
        f"Expected miss (event={EVENT_MISS}), got event={int(event)}"
    )


def test_pong_wall_bounce():
    """Ball should bounce off top/bottom walls (ball_vy reverses sign)."""
    game = Pong(ball_speed=0.05)
    key = jax.random.PRNGKey(30)

    # Create a state where ball is heading toward a wall with strong vy
    state = PongState(
        ball_x=jnp.float32(0.2),
        ball_y=jnp.float32(0.98),   # near top wall
        ball_vx=jnp.float32(0.005),
        ball_vy=jnp.float32(0.05),  # heading up toward top wall
        paddle_y=jnp.float32(0.5),
        score_hits=jnp.int32(0),
        score_misses=jnp.int32(0),
        rally_length=jnp.int32(0),
        rally_history=jnp.zeros(game.max_rallies, dtype=jnp.int32),
    )

    new_state, _ = game.step(state, action=0, key=key)

    # After stepping, ball_vy should have reversed (become negative)
    assert float(new_state.ball_vy) < 0, (
        f"Ball vy should reverse after hitting top wall, got vy={float(new_state.ball_vy)}"
    )


# ---------------------------------------------------------------------------
# Closed-loop integration tests (lightweight, small network)
# ---------------------------------------------------------------------------

def test_closed_loop_creation():
    """Create a ClosedLoop with 100 neurons and verify it instantiates without error."""
    from bl1.network.topology import place_neurons, build_connectivity
    from bl1.core.izhikevich import create_population, IzhikevichParams
    from bl1.mea.electrode import MEA
    from bl1.games.pong import Pong
    from bl1.loop.controller import ClosedLoop
    from bl1.network.types import NetworkParams
    from bl1.plasticity.stdp import STDPParams

    key = jax.random.PRNGKey(0)
    n_neurons = 100

    # Build a small 100-neuron culture
    k1, k2, k3 = jax.random.split(key, 3)
    positions = place_neurons(k1, n_neurons)
    izh_params, _, is_excitatory = create_population(k2, n_neurons)
    W_exc, W_inh, delays = build_connectivity(k3, positions, is_excitatory)

    net_params = NetworkParams(
        positions=positions,
        is_excitatory=is_excitatory,
        W_exc=W_exc,
        W_inh=W_inh,
        delays=delays,
    )

    mea = MEA("cl1_64ch")
    game = Pong()
    stdp_params = STDPParams()

    # Sensory channels: first 8 electrodes
    sensory_channels = list(range(8))
    # Motor regions: 4 electrodes each for up/down
    motor_regions = {"up": list(range(8, 12)), "down": list(range(12, 16))}

    cl = ClosedLoop(
        network_params=net_params,
        neuron_params=izh_params,
        mea=mea,
        sensory_channels=sensory_channels,
        motor_regions=motor_regions,
        game=game,
        stdp_params=stdp_params,
    )

    assert cl is not None
    assert cl.network_params.positions.shape == (100, 2)


def test_closed_loop_short_run():
    """Run ClosedLoop for 1 second (not 300s) with 100 neurons.

    Verify it returns results dict with the expected keys.
    This tests the full pipeline end-to-end.
    """
    from bl1.network.topology import place_neurons, build_connectivity
    from bl1.core.izhikevich import create_population, IzhikevichParams
    from bl1.mea.electrode import MEA
    from bl1.games.pong import Pong
    from bl1.loop.controller import ClosedLoop
    from bl1.network.types import NetworkParams
    from bl1.plasticity.stdp import STDPParams

    key = jax.random.PRNGKey(99)
    n_neurons = 100

    k1, k2, k3, k_run = jax.random.split(key, 4)
    positions = place_neurons(k1, n_neurons)
    izh_params, _, is_excitatory = create_population(k2, n_neurons)
    W_exc, W_inh, delays = build_connectivity(k3, positions, is_excitatory)

    net_params = NetworkParams(
        positions=positions,
        is_excitatory=is_excitatory,
        W_exc=W_exc,
        W_inh=W_inh,
        delays=delays,
    )

    mea = MEA("cl1_64ch")
    game = Pong()
    stdp_params = STDPParams()

    sensory_channels = list(range(8))
    motor_regions = {"up": list(range(8, 12)), "down": list(range(12, 16))}

    cl = ClosedLoop(
        network_params=net_params,
        neuron_params=izh_params,
        mea=mea,
        sensory_channels=sensory_channels,
        motor_regions=motor_regions,
        game=game,
        stdp_params=stdp_params,
    )

    results = cl.run(
        key=k_run,
        duration_s=1.0,
        dt_ms=0.5,
        feedback="fep",
        game_dt_ms=20.0,
    )

    expected_keys = {
        "spike_history",
        "game_events",
        "rally_lengths",
        "final_neuron_state",
        "final_syn_state",
        "final_game_state",
        "population_rates",
    }
    assert set(results.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(results.keys())}, "
        f"extra keys: {set(results.keys()) - expected_keys}"
    )

    # population_rates should have one entry per game step
    # 1 s / 20 ms = 50 game steps
    assert results["population_rates"].shape[0] == 50
