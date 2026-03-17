"""Game environments for closed-loop cortical culture experiments.

Currently provides a minimal Pong implementation in pure JAX, compatible
with ``jax.lax.scan`` for JIT-compiled experiment loops, and a VizDoom
wrapper for 3-D deathmatch / survival scenarios (optional dependency).
"""

from bl1.games.pong import Pong, PongState
from bl1.games.doom import (
    Doom,
    DoomAction,
    DoomEvent,
    DoomState,
    is_vizdoom_available,
)

__all__ = [
    "Pong",
    "PongState",
    "Doom",
    "DoomAction",
    "DoomEvent",
    "DoomState",
    "is_vizdoom_available",
]
