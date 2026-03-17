"""Closed-loop controller connecting a cortical culture to a game environment.

Implements the full DishBrain-style experiment loop: sensory encoding of
game state onto MEA electrodes, motor decoding from population activity,
and feedback stimulation conditioned on game events.

Feedback modes range from the original DishBrain FEP binary feedback
(``"fep"``, ``"open_loop"``, ``"silent"``) to the richer doom-neuron-style
event-based and reward-based protocols provided by
:mod:`bl1.loop.feedback`.
"""

from bl1.loop.controller import ClosedLoop
from bl1.loop.encoding import encode_sensory
from bl1.loop.decoding import decode_motor
from bl1.loop.feedback import (
    EventFeedbackConfig,
    FeedbackProtocol,
    FeedbackState,
    compute_feedback_current,
    create_dishbrain_pong_protocol,
    create_doom_feedback_protocol,
)

__all__ = [
    "ClosedLoop",
    "encode_sensory",
    "decode_motor",
    "EventFeedbackConfig",
    "FeedbackProtocol",
    "FeedbackState",
    "compute_feedback_current",
    "create_dishbrain_pong_protocol",
    "create_doom_feedback_protocol",
]
