"""Rich feedback protocol for closed-loop experiments.

Supports three feedback paradigms:

1. **FEP (Free Energy Principle)** -- DishBrain-style binary feedback:
   - Hit -> predictable stimulation (low surprise)
   - Miss -> unpredictable stimulation (high surprise)

2. **Event-based** -- doom-neuron-style per-event feedback:
   - Each game event (kill, damage, pickup) maps to specific electrode channels
   - Positive events -> structured stimulation
   - Negative events -> unpredictable stimulation
   - Feedback scaled by TD-error surprise magnitude

3. **Reward-based** -- general RL-style feedback:
   - Positive reward -> positive feedback channels
   - Negative reward -> negative feedback channels
   - Amplitude/frequency proportional to reward magnitude

Reference: Kagan et al. (2022), Watmuff et al. (2025), doom-neuron
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Event feedback configuration (mirrors doom-neuron EventFeedbackConfig)
# ---------------------------------------------------------------------------


@dataclass
class EventFeedbackConfig:
    """Configuration for a single event type's feedback stimulation.

    Matches doom-neuron's ``EventFeedbackConfig``.  Each event (e.g.
    ``"enemy_kill"``, ``"took_damage"``) gets its own set of electrode
    channels, base stimulation parameters, and surprise-scaling gains.
    """

    channels: list[int]
    """Electrode channels that receive stimulation for this event."""

    base_frequency: float
    """Base stimulation frequency in Hz."""

    base_amplitude: float
    """Base stimulation amplitude (uA or model units)."""

    base_pulses: int
    """Base number of stimulation pulses."""

    td_sign: str = "positive"
    """Which TD-error sign triggers this event: ``"positive"``,
    ``"negative"``, or ``"absolute"``."""

    # Surprise-scaling gains
    freq_gain: float = 0.2
    """How much normalised surprise scales frequency."""

    freq_max_scale: float = 2.0
    """Maximum frequency multiplier from surprise scaling."""

    amp_gain: float = 0.2
    """How much normalised surprise scales amplitude."""

    amp_max_scale: float = 1.5
    """Maximum amplitude multiplier from surprise scaling."""

    # Unpredictable mode (for negative events)
    unpredictable: bool = False
    """If ``True``, stimulation on these channels is noisy
    (DishBrain-style unpredictable feedback)."""

    unpredictable_frequency: float = 5.0
    """Frequency of unpredictable stimulation bursts (Hz)."""

    unpredictable_duration_ms: float = 1000.0
    """Duration of unpredictable stimulation burst (ms)."""

    unpredictable_rest_ms: float = 1000.0
    """Rest period between unpredictable bursts (ms)."""


# ---------------------------------------------------------------------------
# Complete feedback protocol
# ---------------------------------------------------------------------------


@dataclass
class FeedbackProtocol:
    """Complete feedback protocol configuration.

    Can represent FEP, event-based, or reward-based feedback.  The *mode*
    field selects the paradigm; only the settings relevant to that mode
    are used at runtime.
    """

    mode: str = "fep"
    """Feedback paradigm: ``"fep"``, ``"event_based"``,
    ``"reward_based"``, or ``"silent"``."""

    # FEP mode settings
    fep_predictable_amplitude: float = 2.0
    fep_unpredictable_amplitude_range: tuple[float, float] = (-5.0, 15.0)

    # Event-based settings (doom-neuron style)
    event_configs: dict[str, EventFeedbackConfig] = field(default_factory=dict)

    # Reward-based settings
    reward_positive_channels: list[int] = field(
        default_factory=lambda: [19, 20, 22],
    )
    reward_negative_channels: list[int] = field(
        default_factory=lambda: [23, 24, 26],
    )
    reward_positive_frequency: float = 20.0
    reward_negative_frequency: float = 60.0
    reward_positive_amplitude: float = 2.0
    reward_negative_amplitude: float = 2.0

    # Episode-level feedback
    episode_feedback_enabled: bool = False
    episode_positive_frequency: float = 40.0
    episode_negative_frequency: float = 120.0
    episode_positive_pulses: int = 80
    episode_negative_pulses: int = 160

    # Surprise scaling
    surprise_scaling_enabled: bool = True
    surprise_ema_beta: float = 0.99
    """EMA smoothing coefficient for baseline surprise."""


# ---------------------------------------------------------------------------
# Mutable runtime state
# ---------------------------------------------------------------------------


class FeedbackState:
    """Mutable state for the feedback system.

    Tracks a running exponential moving average of the absolute TD-error
    (used to normalise surprise), per-episode reward accumulator, and a
    step counter.
    """

    def __init__(self) -> None:
        self.surprise_ema: float = 0.0
        self.episode_reward_sum: float = 0.0
        self.step_count: int = 0

    def update_surprise(self, td_error: float, beta: float = 0.99) -> None:
        """Update surprise EMA with a new TD-error observation."""
        self.surprise_ema = beta * self.surprise_ema + (1 - beta) * abs(td_error)

    def reset_episode(self) -> None:
        """Reset per-episode accumulators."""
        self.episode_reward_sum = 0.0


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_doom_feedback_protocol() -> FeedbackProtocol:
    """Create a feedback protocol matching doom-neuron's defaults.

    Returns a fully populated :class:`FeedbackProtocol` in
    ``"event_based"`` mode with four event types (enemy_kill,
    armor_pickup, took_damage, ammo_waste) and episode-level feedback
    enabled.
    """
    return FeedbackProtocol(
        mode="event_based",
        event_configs={
            "enemy_kill": EventFeedbackConfig(
                channels=[35, 36, 38],
                base_frequency=20.0,
                base_amplitude=2.5,
                base_pulses=40,
                td_sign="positive",
            ),
            "armor_pickup": EventFeedbackConfig(
                channels=[39, 40, 43],
                base_frequency=20.0,
                base_amplitude=2.0,
                base_pulses=35,
                td_sign="positive",
            ),
            "took_damage": EventFeedbackConfig(
                channels=[44, 47, 48],
                base_frequency=90.0,
                base_amplitude=2.2,
                base_pulses=50,
                td_sign="negative",
                unpredictable=True,
                unpredictable_frequency=5.0,
                unpredictable_duration_ms=4000.0,
                unpredictable_rest_ms=4000.0,
            ),
            "ammo_waste": EventFeedbackConfig(
                channels=[52, 54, 55],
                base_frequency=60.0,
                base_amplitude=1.8,
                base_pulses=25,
                td_sign="negative",
            ),
        },
        reward_positive_channels=[19, 20, 22],
        reward_negative_channels=[23, 24, 26],
        episode_feedback_enabled=True,
    )


def create_dishbrain_pong_protocol() -> FeedbackProtocol:
    """Create the original DishBrain Pong feedback protocol.

    Returns a :class:`FeedbackProtocol` in ``"fep"`` mode with the
    canonical predictable amplitude (2.0) and unpredictable range
    (-5.0, 15.0) from Kagan et al. (2022).
    """
    return FeedbackProtocol(
        mode="fep",
        fep_predictable_amplitude=2.0,
        fep_unpredictable_amplitude_range=(-5.0, 15.0),
    )


# ---------------------------------------------------------------------------
# Main feedback computation
# ---------------------------------------------------------------------------


def compute_feedback_current(
    protocol: FeedbackProtocol,
    feedback_state: FeedbackState,
    events: list,
    reward: float,
    n_neurons: int,
    neuron_electrode_map: Array,
    key: Array,
    td_error: float = 0.0,
) -> Array:
    """Compute per-neuron feedback stimulation current.

    This is the main entry-point called by the closed-loop controller
    each game step when events occur.

    Args:
        protocol: Feedback protocol configuration.
        feedback_state: Mutable feedback state (surprise EMA, etc.).
        events: List of event names (strings) or event dicts with an
            ``"event_type"`` or ``"type"`` key.
        reward: Scalar reward this step.
        n_neurons: Total neuron count.
        neuron_electrode_map: Boolean mask of shape ``(E, N)`` mapping
            electrodes to neurons (from
            :func:`bl1.mea.electrode.build_neuron_electrode_map`).
        key: JAX PRNG key.
        td_error: TD-error for surprise scaling (default 0.0).

    Returns:
        Per-neuron current injection vector, shape ``(N,)``.
    """
    if protocol.mode == "silent":
        return jnp.zeros(n_neurons)

    I_feedback = jnp.zeros(n_neurons)

    if protocol.mode == "fep":
        I_feedback = _compute_fep(protocol, events, reward, n_neurons, key)

    elif protocol.mode == "event_based":
        I_feedback = _compute_event_based(
            protocol, feedback_state, events, reward,
            n_neurons, neuron_electrode_map, key, td_error,
        )

    elif protocol.mode == "reward_based":
        I_feedback = _compute_reward_based(
            protocol, reward, n_neurons, neuron_electrode_map, key,
        )

    feedback_state.step_count += 1
    feedback_state.episode_reward_sum += reward

    return I_feedback


# ---------------------------------------------------------------------------
# Per-mode implementations
# ---------------------------------------------------------------------------

# Names that count as "positive" or "negative" in FEP mode.
_FEP_POSITIVE = frozenset({"hit", "enemy_kill", "armor_pickup"})
_FEP_NEGATIVE = frozenset({"miss", "took_damage", "ammo_waste"})


def _compute_fep(
    protocol: FeedbackProtocol,
    events: list,
    reward: float,
    n_neurons: int,
    key: Array,
) -> Array:
    """DishBrain-style FEP feedback: predictable vs unpredictable."""
    names = _event_names(events)
    has_positive = reward > 0 or any(n in _FEP_POSITIVE for n in names)
    has_negative = reward < 0 or any(n in _FEP_NEGATIVE for n in names)

    if has_positive:
        return jnp.full(n_neurons, protocol.fep_predictable_amplitude)
    if has_negative:
        lo, hi = protocol.fep_unpredictable_amplitude_range
        return jax.random.uniform(key, (n_neurons,), minval=lo, maxval=hi)
    return jnp.zeros(n_neurons)


def _compute_event_based(
    protocol: FeedbackProtocol,
    feedback_state: FeedbackState,
    events: list,
    reward: float,
    n_neurons: int,
    neuron_electrode_map: Array,
    key: Array,
    td_error: float,
) -> Array:
    """doom-neuron-style per-event channel feedback with surprise scaling."""
    feedback_state.update_surprise(td_error, protocol.surprise_ema_beta)
    surprise_magnitude = abs(td_error) / max(feedback_state.surprise_ema, 1e-8)

    I_feedback = jnp.zeros(n_neurons)

    for event_name in _event_names(events):
        if event_name not in protocol.event_configs:
            continue
        cfg = protocol.event_configs[event_name]

        # Surprise-scaled amplitude
        if protocol.surprise_scaling_enabled:
            amp_scale = min(
                1.0 + cfg.amp_gain * surprise_magnitude,
                cfg.amp_max_scale,
            )
        else:
            amp_scale = 1.0

        amplitude = cfg.base_amplitude * amp_scale

        # Apply to each channel
        for ch in cfg.channels:
            if ch < neuron_electrode_map.shape[0]:
                neuron_mask = neuron_electrode_map[ch].astype(jnp.float32)
                if cfg.unpredictable:
                    key, subkey = jax.random.split(key)
                    noise = jax.random.uniform(
                        subkey, (n_neurons,), minval=0.5, maxval=1.5,
                    )
                    I_feedback = I_feedback + neuron_mask * amplitude * noise
                else:
                    I_feedback = I_feedback + neuron_mask * amplitude

    # Reward-channel overlay (positive -> structured, negative -> noisy)
    if reward > 0 and protocol.reward_positive_channels:
        for ch in protocol.reward_positive_channels:
            if ch < neuron_electrode_map.shape[0]:
                mask = neuron_electrode_map[ch].astype(jnp.float32)
                I_feedback = I_feedback + mask * protocol.reward_positive_amplitude
    elif reward < 0 and protocol.reward_negative_channels:
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, (n_neurons,), minval=0.5, maxval=1.5)
        for ch in protocol.reward_negative_channels:
            if ch < neuron_electrode_map.shape[0]:
                mask = neuron_electrode_map[ch].astype(jnp.float32)
                I_feedback = I_feedback + mask * protocol.reward_negative_amplitude * noise

    return I_feedback


def _compute_reward_based(
    protocol: FeedbackProtocol,
    reward: float,
    n_neurons: int,
    neuron_electrode_map: Array,
    key: Array,
) -> Array:
    """General RL-style reward-proportional feedback."""
    I_feedback = jnp.zeros(n_neurons)

    if reward > 0:
        for ch in protocol.reward_positive_channels:
            if ch < neuron_electrode_map.shape[0]:
                mask = neuron_electrode_map[ch].astype(jnp.float32)
                I_feedback = I_feedback + (
                    mask * protocol.reward_positive_amplitude * abs(reward)
                )
    elif reward < 0:
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, (n_neurons,), minval=0.5, maxval=1.5)
        for ch in protocol.reward_negative_channels:
            if ch < neuron_electrode_map.shape[0]:
                mask = neuron_electrode_map[ch].astype(jnp.float32)
                I_feedback = I_feedback + (
                    mask * protocol.reward_negative_amplitude * abs(reward) * noise
                )

    return I_feedback


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _event_names(events: list) -> list[str]:
    """Extract event name strings from heterogeneous event formats.

    Handles bare strings, dicts with ``"event_type"`` or ``"type"``
    keys, and objects with an ``event_type`` attribute.
    """
    names: list[str] = []
    for e in events:
        if isinstance(e, str):
            names.append(e)
        elif isinstance(e, dict):
            names.append(e.get("event_type", e.get("type", "")))
        elif hasattr(e, "event_type"):
            names.append(e.event_type)
    return names
