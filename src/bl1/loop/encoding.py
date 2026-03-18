"""Sensory encoding: game state to electrode stimulation patterns.

Implements the mixed place + rate coding scheme from DishBrain (Kagan et al.
2022).  Ball position on the game field is converted to a spatiotemporal
stimulation pattern across 8 sensory electrode channels:

* **Place coding** -- The ball's Y position selects which of 8 sensory
  channels is active.  Two adjacent channels are activated when the ball
  falls near a band boundary, providing smoother spatial encoding.

* **Rate coding** -- The ball's X proximity to the paddle modulates the
  stimulation *frequency*.  When far from the paddle the frequency is low
  (~4 Hz); when close it is high (~40 Hz).  A phase accumulator is used
  to generate pulse timing consistent with the desired frequency at
  arbitrary simulation timesteps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from bl1.mea.electrode import MEAConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STIM_AMPLITUDE_MV: float = 150.0  # Standard stimulation amplitude (mV)
FREQ_MIN_HZ: float = 4.0  # Stimulation freq when ball is far
FREQ_MAX_HZ: float = 40.0  # Stimulation freq when ball is close
N_SENSORY_CHANNELS: int = 8  # Number of sensory electrode channels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _place_code(ball_y: float, n_channels: int = N_SENSORY_CHANNELS) -> list[int]:
    """Map ball Y position (0-1 normalised) to active channel indices.

    The field is divided into *n_channels* horizontal bands.  The band
    containing the ball determines the primary active channel.  If the ball
    is within the outer 25 % of a band boundary, the adjacent channel is
    also activated for smoother encoding.

    Args:
        ball_y: Normalised ball Y position in [0, 1].
        n_channels: Number of sensory channels.

    Returns:
        List of 1 or 2 active channel indices (0-based).
    """
    # Clamp to valid range
    ball_y = float(jnp.clip(ball_y, 0.0, 1.0))

    band_width = 1.0 / n_channels
    # Primary channel index
    primary = int(ball_y / band_width)
    primary = min(primary, n_channels - 1)

    # Position within the band [0, 1)
    pos_in_band = (ball_y - primary * band_width) / band_width

    active = [primary]

    # Activate adjacent channel when near a band boundary (outer 25 %)
    boundary_threshold = 0.25
    if pos_in_band < boundary_threshold and primary > 0:
        active.append(primary - 1)
    elif pos_in_band > (1.0 - boundary_threshold) and primary < n_channels - 1:
        active.append(primary + 1)

    return active


def _rate_code_frequency(ball_x: float) -> float:
    """Map ball X proximity to stimulation frequency in Hz.

    Args:
        ball_x: Normalised ball X position in [0, 1], where 0 is far from
            the paddle and 1 is close.

    Returns:
        Desired stimulation frequency in Hz.
    """
    ball_x = float(jnp.clip(ball_x, 0.0, 1.0))
    return FREQ_MIN_HZ + ball_x * (FREQ_MAX_HZ - FREQ_MIN_HZ)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_sensory(
    game_state,
    mea_config: MEAConfig,
    sensory_channels: list[int],
    dt_game_ms: float = 20.0,
    *,
    _phase_accumulator: float = 0.0,
) -> tuple[list[int], float, float]:
    """Convert game state to electrode stimulation pattern.

    Place coding: Ball Y position mapped to which of the 8 sensory channels
    is active.  Rate coding: Ball X proximity to paddle encoded as
    stimulation frequency.

    Args:
        game_state: PongState (or any object) with ``.ball_x`` and
            ``.ball_y`` attributes in [0, 1].
        mea_config: MEAConfig with electrode positions (unused in the
            current implementation but reserved for future spatial mapping).
        sensory_channels: List of 8 electrode indices used for sensory
            input.  ``sensory_channels[i]`` is the electrode index for
            sensory channel *i*.
        dt_game_ms: Game timestep in ms (default 20 ms = 50 Hz game
            update rate).
        _phase_accumulator: Internal accumulated phase (radians-equivalent
            in period fraction).  Callers should pass the returned
            ``new_phase`` back on the next call to maintain pulse timing
            continuity.

    Returns:
        active_electrodes: List of electrode indices to stimulate this
            timestep (may be empty if no pulse is due).
        stim_amplitude: Stimulation amplitude in mV (150 mV standard);
            0.0 when no pulse is due.
        new_phase: Updated phase accumulator for the next call.
    """
    if len(sensory_channels) != N_SENSORY_CHANNELS:
        raise ValueError(
            f"Expected {N_SENSORY_CHANNELS} sensory channels, got {len(sensory_channels)}"
        )

    # --- Place coding: which channels are active? --------------------------
    ball_y = float(game_state.ball_y)
    active_channel_ids = _place_code(ball_y, N_SENSORY_CHANNELS)

    # Map channel indices to electrode indices
    candidate_electrodes = [sensory_channels[ch] for ch in active_channel_ids]

    # --- Rate coding: should we emit a pulse this timestep? ----------------
    ball_x = float(game_state.ball_x)
    freq_hz = _rate_code_frequency(ball_x)

    # Phase accumulation: each game timestep advances phase by
    # freq_hz * dt_game_ms / 1000.  A pulse fires whenever accumulated
    # phase crosses an integer boundary.
    dt_game_s = dt_game_ms / 1000.0
    phase_increment = freq_hz * dt_game_s
    new_phase = _phase_accumulator + phase_increment

    # Determine if a pulse should occur: phase crossed an integer boundary
    emit_pulse = int(new_phase) > int(_phase_accumulator)

    if emit_pulse:
        return candidate_electrodes, STIM_AMPLITUDE_MV, new_phase
    else:
        return [], 0.0, new_phase
