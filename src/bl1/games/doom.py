"""VizDoom game environment for closed-loop cortical culture experiments.

Wraps VizDoom scenarios (progressive deathmatch, survival, deadly corridor)
with an interface compatible with BL-1's closed-loop controller.

VizDoom is an optional dependency. When not installed, the module provides
stub classes that raise ImportError on instantiation.

The action space matches doom-neuron's channel group structure:
- forward/backward (3 states)
- strafe left/right (3 states)
- turn left/right (3 states or continuous delta)
- attack (binary)

Scenarios:
- progressive_deathmatch: Kill enemies, manage ammo (default)
- survival: Survive against enemies
- deadly_corridor_1-5: Curriculum with increasing difficulty
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_VIZDOOM_AVAILABLE = False
try:
    import vizdoom
    from vizdoom import Button, DoomGame, GameVariable, Mode

    _VIZDOOM_AVAILABLE = True
except ImportError:
    pass


def is_vizdoom_available() -> bool:
    """Return True if the vizdoom package is importable."""
    return _VIZDOOM_AVAILABLE


# ---------------------------------------------------------------------------
# State / event / action containers
# ---------------------------------------------------------------------------


class DoomState(NamedTuple):
    """Game state for Doom environment."""

    screen_buffer: np.ndarray  # (H, W, C) or (C, H, W) screen pixels
    health: float
    ammo: float
    kill_count: int
    armor: float
    episode_reward: float
    is_terminal: bool
    step_count: int


class DoomEvent(NamedTuple):
    """Event detected during a game step."""

    event_type: str  # "enemy_kill", "took_damage", "armor_pickup", "ammo_waste", etc.
    magnitude: float  # Event magnitude (e.g., damage amount)


class DoomAction(NamedTuple):
    """Multi-axis action for Doom.

    The action space mirrors doom-neuron's channel group structure so that
    cortical culture output channels can map directly to game controls.
    """

    move_forward: int = 0  # 0=none, 1=forward, 2=backward
    strafe: int = 0  # 0=none, 1=left, 2=right
    turn: int = 0  # 0=none, 1=left, 2=right (or continuous delta)
    attack: bool = False  # True=fire


# ---------------------------------------------------------------------------
# Doom environment
# ---------------------------------------------------------------------------


class Doom:
    """VizDoom game environment for BL-1 closed-loop experiments.

    Provides a higher-level interface than raw VizDoom, with:
    - Event detection (kills, damage, pickups)
    - Multi-axis discrete action space matching doom-neuron's channel groups
    - Screen buffer + scalar observation extraction
    - Configurable scenarios and difficulty

    Example::

        game = Doom(config_path="progressive_deathmatch.cfg")
        state = game.reset()
        action = DoomAction(move_forward=1, attack=True)
        state, events, reward = game.step(action)
        game.close()
    """

    def __init__(
        self,
        config_path: str = "progressive_deathmatch.cfg",
        wad_path: str | None = None,
        screen_resolution: str = "RES_320X240",
        show_window: bool = False,
        tick_frequency_hz: int = 10,
    ):
        if not _VIZDOOM_AVAILABLE:
            raise ImportError(
                "VizDoom is not installed. Install with: pip install vizdoom"
            )

        self.config_path = config_path
        self.tick_frequency_hz = tick_frequency_hz
        self._game: DoomGame | None = None
        self._step_count = 0
        self._prev_health = 100.0
        self._prev_ammo = 50.0
        self._prev_kill_count = 0
        self._prev_armor = 0.0

        # Setup game
        self._game = DoomGame()
        self._game.load_config(config_path)
        if wad_path:
            self._game.set_doom_scenario_path(wad_path)

        # Resolution
        res_map = {
            "RES_160X120": vizdoom.ScreenResolution.RES_160X120,
            "RES_320X240": vizdoom.ScreenResolution.RES_320X240,
            "RES_640X480": vizdoom.ScreenResolution.RES_640X480,
        }
        self._game.set_screen_resolution(
            res_map.get(screen_resolution, vizdoom.ScreenResolution.RES_320X240)
        )
        self._game.set_window_visible(show_window)
        self._game.set_mode(Mode.PLAYER)
        self._game.init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> DoomState:
        """Start a new episode."""
        self._game.new_episode()
        self._step_count = 0
        self._prev_health = 100.0
        self._prev_ammo = 50.0
        self._prev_kill_count = 0
        self._prev_armor = 0.0
        return self._get_state()

    def step(self, action: DoomAction) -> tuple[DoomState, list[DoomEvent], float]:
        """Step the game with a multi-axis action.

        Args:
            action: DoomAction with movement, strafe, turn, attack.

        Returns:
            Tuple of (state, events, reward).
        """
        # Convert DoomAction to VizDoom button vector
        doom_action = self._convert_action(action)

        # Execute action
        reward = self._game.make_action(doom_action)
        self._step_count += 1

        # Get new state
        state = self._get_state()

        # Detect events
        events = self._detect_events(state)

        return state, events, reward

    def close(self) -> None:
        """Clean up VizDoom instance."""
        if self._game is not None:
            self._game.close()
            self._game = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> DoomState:
        """Extract current game state."""
        if self._game.is_episode_finished():
            return DoomState(
                screen_buffer=np.zeros((240, 320, 3), dtype=np.uint8),
                health=0,
                ammo=0,
                kill_count=0,
                armor=0,
                episode_reward=self._game.get_total_reward(),
                is_terminal=True,
                step_count=self._step_count,
            )

        game_state = self._game.get_state()
        screen = (
            game_state.screen_buffer
            if game_state.screen_buffer is not None
            else np.zeros((240, 320, 3), dtype=np.uint8)
        )

        health = self._game.get_game_variable(GameVariable.HEALTH)
        ammo = self._game.get_game_variable(GameVariable.AMMO2)  # Selected weapon ammo
        kill_count = int(self._game.get_game_variable(GameVariable.KILLCOUNT))
        armor = self._game.get_game_variable(GameVariable.ARMOR)

        return DoomState(
            screen_buffer=screen,
            health=health,
            ammo=ammo,
            kill_count=kill_count,
            armor=armor,
            episode_reward=self._game.get_total_reward(),
            is_terminal=False,
            step_count=self._step_count,
        )

    def _detect_events(self, state: DoomState) -> list[DoomEvent]:
        """Detect game events by comparing current and previous state."""
        events: list[DoomEvent] = []

        if state.is_terminal:
            return events

        # Kill
        if state.kill_count > self._prev_kill_count:
            events.append(
                DoomEvent(
                    "enemy_kill",
                    float(state.kill_count - self._prev_kill_count),
                )
            )

        # Damage taken
        if state.health < self._prev_health:
            events.append(
                DoomEvent("took_damage", self._prev_health - state.health)
            )

        # Armor pickup
        if state.armor > self._prev_armor:
            events.append(
                DoomEvent("armor_pickup", state.armor - self._prev_armor)
            )

        # Ammo waste (fired but no kill)
        if state.ammo < self._prev_ammo and state.kill_count == self._prev_kill_count:
            events.append(
                DoomEvent("ammo_waste", self._prev_ammo - state.ammo)
            )

        # Update previous state
        self._prev_health = state.health
        self._prev_ammo = state.ammo
        self._prev_kill_count = state.kill_count
        self._prev_armor = state.armor

        return events

    def _convert_action(self, action: DoomAction) -> list:
        """Convert DoomAction to VizDoom button list.

        Maps the multi-axis DoomAction (forward/backward, strafe, turn,
        attack) onto the VizDoom button vector, respecting whichever
        buttons are configured in the active scenario.
        """
        buttons = [0] * self._game.get_available_buttons_size()

        # Map based on available buttons (varies by config)
        available = self._game.get_available_buttons()

        for i, btn in enumerate(available):
            if btn == Button.MOVE_FORWARD and action.move_forward == 1 or btn == Button.MOVE_BACKWARD and action.move_forward == 2 or btn == Button.MOVE_LEFT and action.strafe == 1 or btn == Button.MOVE_RIGHT and action.strafe == 2 or btn == Button.TURN_LEFT and action.turn == 1 or btn == Button.TURN_RIGHT and action.turn == 2 or btn == Button.ATTACK and action.attack:
                buttons[i] = 1

        return buttons

    def __del__(self) -> None:
        if hasattr(self, "_game"):
            self.close()
