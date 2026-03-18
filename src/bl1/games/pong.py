"""Simple Pong game environment for closed-loop cortical culture experiments.

A minimal Pong implementation in pure JAX.  The ball moves from the left
side toward a paddle on the right.  The culture's decoded motor output
controls the paddle.  Hits and misses produce distinct feedback signals
that drive learning in the culture.

All state is stored in a JAX-compatible ``PongState`` NamedTuple, and
the step function uses ``jax.lax`` primitives so it can be JIT-compiled
and embedded in ``jax.lax.scan`` loops.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


class PongState(NamedTuple):
    """Complete game state for a single Pong instance."""

    ball_x: Array  # scalar, 0-1 normalised horizontal position
    ball_y: Array  # scalar, 0-1 normalised vertical position
    ball_vx: Array  # scalar, horizontal velocity (positive = rightward)
    ball_vy: Array  # scalar, vertical velocity
    paddle_y: Array  # scalar, 0-1 normalised paddle centre
    score_hits: Array  # scalar int, cumulative hits
    score_misses: Array  # scalar int, cumulative misses
    rally_length: Array  # scalar int, current consecutive hits
    rally_history: Array  # (max_rallies,) completed rally lengths


# ---------------------------------------------------------------------------
# Event encoding (used as return alongside state)
# ---------------------------------------------------------------------------
# We encode events as integers so the step function stays JIT-friendly:
#   0 = "none"
#   1 = "hit"
#   2 = "miss"

EVENT_NONE: int = 0
EVENT_HIT: int = 1
EVENT_MISS: int = 2


# ---------------------------------------------------------------------------
# Pong environment
# ---------------------------------------------------------------------------


class Pong:
    """Simple Pong game environment.

    The playing field spans [0, 1] in both x and y.  The ball starts on
    the left side and moves rightward.  The paddle occupies a vertical
    strip on the right edge (x = 1).

    Example::

        game = Pong()
        key = jax.random.PRNGKey(0)
        state = game.reset(key)
        state, event = game.step(state, action=1, key=key)
    """

    paddle_height: float = 0.2  # fraction of field height
    paddle_speed: float = 0.03  # paddle movement per step
    ball_speed: float = 0.01  # base ball speed per step
    max_rallies: int = 1024  # size of rally_history buffer

    def __init__(
        self,
        paddle_height: float = 0.2,
        paddle_speed: float = 0.03,
        ball_speed: float = 0.01,
        max_rallies: int = 1024,
    ) -> None:
        self.paddle_height = paddle_height
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed
        self.max_rallies = max_rallies

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, key: Array) -> PongState:
        """Initialise a fresh game state with random ball direction.

        Args:
            key: JAX PRNG key.

        Returns:
            A ``PongState`` with the ball at the left edge heading
            rightward and the paddle centred.
        """
        key_angle, _ = jax.random.split(key)
        state = self._reset_ball(
            PongState(
                ball_x=jnp.float32(0.0),
                ball_y=jnp.float32(0.5),
                ball_vx=jnp.float32(self.ball_speed),
                ball_vy=jnp.float32(0.0),
                paddle_y=jnp.float32(0.5),
                score_hits=jnp.int32(0),
                score_misses=jnp.int32(0),
                rally_length=jnp.int32(0),
                rally_history=jnp.zeros(self.max_rallies, dtype=jnp.int32),
            ),
            key_angle,
        )
        return state

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: PongState,
        action: int | Array,
        key: Array,
    ) -> tuple[PongState, Array]:
        """Advance the game by one timestep.

        Args:
            state: Current PongState.
            action: Paddle action — 0 = stay, 1 = up, 2 = down.
            key: JAX PRNG key (used to randomise ball angle on reset).

        Returns:
            new_state: Updated PongState.
            event: Integer event code (0 = none, 1 = hit, 2 = miss).
        """
        action = jnp.asarray(action, dtype=jnp.int32)

        # --- Paddle movement ---
        paddle_dy = jnp.where(
            action == 1,
            self.paddle_speed,
            jnp.where(action == 2, -self.paddle_speed, 0.0),
        )
        new_paddle_y = jnp.clip(state.paddle_y + paddle_dy, 0.0, 1.0)

        # --- Ball movement ---
        new_ball_x = state.ball_x + state.ball_vx
        new_ball_y = state.ball_y + state.ball_vy

        # Bounce off top/bottom walls
        # Reflect if out of [0, 1]
        new_ball_vy = jnp.where(
            (new_ball_y <= 0.0) | (new_ball_y >= 1.0),
            -state.ball_vy,
            state.ball_vy,
        )
        new_ball_y = jnp.clip(new_ball_y, 0.0, 1.0)

        # Bounce off left wall (safety — ball should always head right,
        # but handle it gracefully)
        new_ball_vx = jnp.where(new_ball_x <= 0.0, jnp.abs(state.ball_vx), state.ball_vx)
        new_ball_x = jnp.maximum(new_ball_x, 0.0)

        # --- Check if ball reached right side ---
        reached_right = new_ball_x >= 1.0

        half_paddle = self.paddle_height / 2.0
        paddle_top = new_paddle_y + half_paddle
        paddle_bottom = new_paddle_y - half_paddle
        ball_in_paddle = (new_ball_y >= paddle_bottom) & (new_ball_y <= paddle_top)

        is_hit = reached_right & ball_in_paddle
        is_miss = reached_right & ~ball_in_paddle

        # Event code
        event = jnp.where(is_hit, EVENT_HIT, jnp.where(is_miss, EVENT_MISS, EVENT_NONE))

        # --- Update scores ---
        new_hits = state.score_hits + is_hit.astype(jnp.int32)
        new_misses = state.score_misses + is_miss.astype(jnp.int32)

        # --- Rally tracking ---
        new_rally_length = jnp.where(
            is_hit,
            state.rally_length + 1,
            jnp.where(is_miss, jnp.int32(0), state.rally_length),
        )

        # On miss, record the completed rally in history
        total_completed = state.score_misses  # index for next slot
        history_idx = total_completed % self.max_rallies
        new_rally_history = jnp.where(
            is_miss,
            state.rally_history.at[history_idx].set(state.rally_length),
            state.rally_history,
        )

        # --- Ball reset on hit or miss ---
        intermediate_state = PongState(
            ball_x=new_ball_x,
            ball_y=new_ball_y,
            ball_vx=new_ball_vx,
            ball_vy=new_ball_vy,
            paddle_y=new_paddle_y,
            score_hits=new_hits,
            score_misses=new_misses,
            rally_length=new_rally_length,
            rally_history=new_rally_history,
        )

        # Reset ball position and velocity when it reached the right side
        reset_state = self._reset_ball(intermediate_state, key)
        new_state = jax.lax.cond(
            reached_right,
            lambda _: reset_state,
            lambda _: intermediate_state,
            None,
        )

        return new_state, event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_ball(self, state: PongState, key: Array) -> PongState:
        """Place the ball at the left edge with a random rightward angle.

        The launch angle is sampled uniformly from [-pi/4, +pi/4] so the
        ball always heads rightward with some vertical spread.
        """
        angle = jax.random.uniform(key, minval=-jnp.pi / 4, maxval=jnp.pi / 4)
        vx = self.ball_speed * jnp.cos(angle)
        vy = self.ball_speed * jnp.sin(angle)

        return state._replace(
            ball_x=jnp.float32(0.0),
            ball_y=jnp.float32(0.5),
            ball_vx=vx,
            ball_vy=vy,
        )
