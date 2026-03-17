"""Game environments for closed-loop cortical culture experiments.

Currently provides a minimal Pong implementation in pure JAX, compatible
with ``jax.lax.scan`` for JIT-compiled experiment loops.
"""

from bl1.games.pong import Pong, PongState

__all__ = ["Pong", "PongState"]
