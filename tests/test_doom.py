"""Tests for the VizDoom game environment wrapper.

All tests handle the case where VizDoom is not installed.  The first four
tests always run; the remaining tests are skipped when vizdoom is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional-dependency detection
# ---------------------------------------------------------------------------

try:
    import vizdoom

    HAS_VIZDOOM = True
except ImportError:
    HAS_VIZDOOM = False

skip_no_vizdoom = pytest.mark.skipif(
    not HAS_VIZDOOM,
    reason="VizDoom not installed (pip install vizdoom)",
)


# ===================================================================
# Tests that always run (whether or not vizdoom is installed)
# ===================================================================


def test_is_vizdoom_available():
    """is_vizdoom_available() returns a bool matching importability."""
    from bl1.games.doom import is_vizdoom_available

    result = is_vizdoom_available()
    assert isinstance(result, bool)
    assert result == HAS_VIZDOOM


def test_import_without_vizdoom():
    """The doom module is importable even when vizdoom is absent."""
    # The module itself should always import without error, regardless
    # of whether vizdoom is installed -- import guards handle it.
    import bl1.games.doom  # noqa: F401


def test_doom_state_types():
    """DoomState, DoomEvent, DoomAction are importable and constructable."""
    from bl1.games.doom import DoomAction, DoomEvent, DoomState

    state = DoomState(
        screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
        health=100.0,
        ammo=50.0,
        kill_count=0,
        armor=0.0,
        episode_reward=0.0,
        is_terminal=False,
        step_count=0,
    )
    assert state.health == 100.0
    assert state.is_terminal is False

    event = DoomEvent(event_type="enemy_kill", magnitude=1.0)
    assert event.event_type == "enemy_kill"

    action = DoomAction(move_forward=1, strafe=0, turn=2, attack=True)
    assert action.move_forward == 1
    assert action.attack is True

    # Default action
    default_action = DoomAction()
    assert default_action.move_forward == 0
    assert default_action.attack is False


@pytest.mark.skipif(HAS_VIZDOOM, reason="Only runs when vizdoom is NOT installed")
def test_doom_requires_vizdoom():
    """Doom() raises ImportError when vizdoom is not installed."""
    from bl1.games.doom import Doom

    with pytest.raises(ImportError, match="VizDoom is not installed"):
        Doom()


# ===================================================================
# Tests that require VizDoom
# ===================================================================


@skip_no_vizdoom
def test_doom_init(tmp_path):
    """Doom can be initialised with a basic VizDoom config."""
    from bl1.games.doom import Doom

    # Write a minimal config that uses a built-in scenario
    cfg = tmp_path / "basic.cfg"
    cfg.write_text(
        "doom_scenario_path = basic.wad\n"
        "doom_map = map01\n"
        "episode_timeout = 300\n"
        "available_buttons = { MOVE_FORWARD MOVE_BACKWARD "
        "MOVE_LEFT MOVE_RIGHT TURN_LEFT TURN_RIGHT ATTACK }\n"
        "available_game_variables = { HEALTH AMMO2 KILLCOUNT ARMOR }\n"
        "screen_resolution = RES_320X240\n"
        "screen_format = CRCGCB\n"
        "render_hud = false\n"
    )
    game = Doom(config_path=str(cfg), show_window=False)
    assert game._game is not None
    game.close()


@skip_no_vizdoom
def test_doom_reset(tmp_path):
    """Doom.reset() returns a valid DoomState."""
    from bl1.games.doom import Doom, DoomState

    cfg = tmp_path / "basic.cfg"
    cfg.write_text(
        "doom_scenario_path = basic.wad\n"
        "doom_map = map01\n"
        "episode_timeout = 300\n"
        "available_buttons = { MOVE_FORWARD MOVE_BACKWARD "
        "MOVE_LEFT MOVE_RIGHT TURN_LEFT TURN_RIGHT ATTACK }\n"
        "available_game_variables = { HEALTH AMMO2 KILLCOUNT ARMOR }\n"
        "screen_resolution = RES_320X240\n"
        "screen_format = CRCGCB\n"
        "render_hud = false\n"
    )
    game = Doom(config_path=str(cfg), show_window=False)
    state = game.reset()
    assert isinstance(state, DoomState)
    assert state.is_terminal is False
    assert state.step_count == 0
    assert isinstance(state.screen_buffer, np.ndarray)
    game.close()


@skip_no_vizdoom
def test_doom_step(tmp_path):
    """Doom.step() returns state, events, reward."""
    from bl1.games.doom import Doom, DoomAction, DoomEvent, DoomState

    cfg = tmp_path / "basic.cfg"
    cfg.write_text(
        "doom_scenario_path = basic.wad\n"
        "doom_map = map01\n"
        "episode_timeout = 300\n"
        "available_buttons = { MOVE_FORWARD MOVE_BACKWARD "
        "MOVE_LEFT MOVE_RIGHT TURN_LEFT TURN_RIGHT ATTACK }\n"
        "available_game_variables = { HEALTH AMMO2 KILLCOUNT ARMOR }\n"
        "screen_resolution = RES_320X240\n"
        "screen_format = CRCGCB\n"
        "render_hud = false\n"
    )
    game = Doom(config_path=str(cfg), show_window=False)
    game.reset()

    action = DoomAction(move_forward=1, attack=False)
    state, events, reward = game.step(action)

    assert isinstance(state, DoomState)
    assert isinstance(events, list)
    assert all(isinstance(e, DoomEvent) for e in events)
    assert isinstance(reward, (int, float))
    assert state.step_count == 1
    game.close()


@skip_no_vizdoom
def test_event_detection(tmp_path):
    """Event detection correctly identifies state transitions."""
    from bl1.games.doom import Doom, DoomAction, DoomState

    cfg = tmp_path / "basic.cfg"
    cfg.write_text(
        "doom_scenario_path = basic.wad\n"
        "doom_map = map01\n"
        "episode_timeout = 300\n"
        "available_buttons = { MOVE_FORWARD MOVE_BACKWARD "
        "MOVE_LEFT MOVE_RIGHT TURN_LEFT TURN_RIGHT ATTACK }\n"
        "available_game_variables = { HEALTH AMMO2 KILLCOUNT ARMOR }\n"
        "screen_resolution = RES_320X240\n"
        "screen_format = CRCGCB\n"
        "render_hud = false\n"
    )
    game = Doom(config_path=str(cfg), show_window=False)
    game.reset()

    # Manually test _detect_events by crafting states
    # Simulate a kill event
    game._prev_kill_count = 0
    game._prev_health = 100.0
    game._prev_ammo = 50.0
    game._prev_armor = 0.0

    fake_state = DoomState(
        screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
        health=80.0,
        ammo=49.0,
        kill_count=1,
        armor=0.0,
        episode_reward=1.0,
        is_terminal=False,
        step_count=1,
    )
    events = game._detect_events(fake_state)
    event_types = [e.event_type for e in events]

    assert "enemy_kill" in event_types
    assert "took_damage" in event_types
    # Ammo decreased but there WAS a kill, so no ammo_waste
    assert "ammo_waste" not in event_types

    # Now simulate ammo waste (fire, miss)
    game._prev_kill_count = 1
    game._prev_health = 80.0
    game._prev_ammo = 49.0
    game._prev_armor = 0.0

    waste_state = DoomState(
        screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
        health=80.0,
        ammo=48.0,
        kill_count=1,
        armor=0.0,
        episode_reward=1.0,
        is_terminal=False,
        step_count=2,
    )
    events2 = game._detect_events(waste_state)
    event_types2 = [e.event_type for e in events2]
    assert "ammo_waste" in event_types2
    assert "enemy_kill" not in event_types2

    # Armor pickup
    game._prev_kill_count = 1
    game._prev_health = 80.0
    game._prev_ammo = 48.0
    game._prev_armor = 0.0

    armor_state = DoomState(
        screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
        health=80.0,
        ammo=48.0,
        kill_count=1,
        armor=50.0,
        episode_reward=1.0,
        is_terminal=False,
        step_count=3,
    )
    events3 = game._detect_events(armor_state)
    event_types3 = [e.event_type for e in events3]
    assert "armor_pickup" in event_types3

    # Terminal state returns no events
    terminal_state = DoomState(
        screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
        health=0,
        ammo=0,
        kill_count=0,
        armor=0,
        episode_reward=0.0,
        is_terminal=True,
        step_count=4,
    )
    events4 = game._detect_events(terminal_state)
    assert events4 == []

    game.close()
