"""Closed-loop controller connecting a cortical culture to a game environment.

Implements the full DishBrain-style experiment loop: sensory encoding of
game state onto MEA electrodes, motor decoding from population activity,
and feedback stimulation conditioned on game events.  Three feedback modes
are supported: ``"fep"`` (Free Energy Principle), ``"open_loop"``, and
``"silent"``.
"""

from bl1.loop.controller import ClosedLoop
from bl1.loop.encoding import encode_sensory
from bl1.loop.decoding import decode_motor

__all__ = ["ClosedLoop", "encode_sensory", "decode_motor"]
