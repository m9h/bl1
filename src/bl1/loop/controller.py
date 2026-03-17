"""Closed-loop controller connecting a cortical culture to a game environment.

Orchestrates the full experiment cycle:

1. **Decode** motor actions from recent neural activity.
2. **Step** the game environment with the decoded action.
3. **Encode** new sensory input and apply feedback stimulation.
4. **Simulate** the neural culture for the inter-game-update interval,
   using ``jax.lax.scan`` for GPU-accelerated inner steps.
5. **Record** spike statistics and game events for post-hoc analysis.

Three feedback modes are supported (following DishBrain):

- ``"fep"`` -- Free Energy Principle feedback.  On a *hit*, deliver
  predictable (low-entropy) stimulation; on a *miss*, deliver
  unpredictable (high-entropy) stimulation.
- ``"open_loop"`` -- Random stimulation uncorrelated with performance.
- ``"silent"`` -- No stimulation at all (spontaneous activity only).
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
    gaba_b_step,
    nmda_step,
)
from bl1.games.pong import EVENT_HIT, EVENT_MISS
from bl1.loop.decoding import decode_motor
from bl1.loop.encoding import STIM_AMPLITUDE_MV, encode_sensory
from bl1.plasticity.stdp import STDPParams, STDPState, init_stdp_state, stdp_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feedback stimulation helpers
# ---------------------------------------------------------------------------

def _feedback_fep(
    key: Array,
    event_code: int,
    n_neurons: int,
) -> Array:
    """Generate FEP-style feedback stimulation current.

    - **Hit** (predictable): uniform low-amplitude current across all neurons.
    - **Miss** (unpredictable): random high-amplitude current.
    """
    is_hit = event_code == EVENT_HIT
    predictable = jnp.full(n_neurons, 2.0)
    unpredictable = jax.random.uniform(key, (n_neurons,), minval=-5.0, maxval=15.0)
    return jnp.where(is_hit, predictable, unpredictable)


def _feedback_open_loop(key: Array, n_neurons: int) -> Array:
    """Random stimulation uncorrelated with game events."""
    return jax.random.uniform(key, (n_neurons,), minval=-2.0, maxval=8.0)


# ---------------------------------------------------------------------------
# Inner simulation step (designed for jax.lax.scan)
# ---------------------------------------------------------------------------

def _make_scan_step(
    params: IzhikevichParams,
    W_inh: Array,
    dt: float,
    has_plasticity: bool,
    stdp_params: STDPParams | None,
    is_excitatory: Array,
):
    """Build a single-step function suitable for ``jax.lax.scan``."""

    def step_fn(carry, I_ext_t):
        neuron_state, syn_state, stdp_state, W_exc = carry

        # 1. Synaptic current
        I_syn = compute_synaptic_current(syn_state, neuron_state.v)

        # 2. Total input
        I_total = I_syn + I_ext_t

        # 3. Neuron update
        neuron_state = izhikevich_step(neuron_state, params, I_total, dt)

        # 4. Synapse conductance update
        spikes_f = neuron_state.spikes.astype(jnp.float32)

        # Phase 1: single-exponential receptors
        new_g_ampa = ampa_step(syn_state.g_ampa, spikes_f, W_exc, dt)
        new_g_gaba_a = gaba_a_step(syn_state.g_gaba_a, spikes_f, W_inh, dt)

        # Phase 2: dual-exponential receptors
        new_nmda_rise, new_nmda_decay, _ = nmda_step(
            syn_state.g_nmda_rise, syn_state.g_nmda_decay,
            spikes_f, W_exc, dt,
        )
        new_gaba_b_rise, new_gaba_b_decay, _ = gaba_b_step(
            syn_state.g_gaba_b_rise, syn_state.g_gaba_b_decay,
            spikes_f, W_inh, dt,
        )

        syn_state = SynapseState(
            g_ampa=new_g_ampa,
            g_gaba_a=new_g_gaba_a,
            g_nmda_rise=new_nmda_rise,
            g_nmda_decay=new_nmda_decay,
            g_gaba_b_rise=new_gaba_b_rise,
            g_gaba_b_decay=new_gaba_b_decay,
        )

        # 5. STDP (optional)
        if has_plasticity and stdp_params is not None:
            stdp_state, W_exc = stdp_update(
                stdp_state, stdp_params, neuron_state.spikes,
                W_exc, is_excitatory, dt,
            )

        new_carry = (neuron_state, syn_state, stdp_state, W_exc)
        return new_carry, neuron_state.spikes

    return step_fn


# ---------------------------------------------------------------------------
# Main ClosedLoop class
# ---------------------------------------------------------------------------

class ClosedLoop:
    """Closed-loop controller connecting a cortical culture to a game.

    Wires together the neural simulation, MEA interface, sensory encoder,
    motor decoder, and game environment into a single experiment loop.
    Call :meth:`run` to execute the full experiment and collect results.

    Example::

        loop = ClosedLoop(
            network_params, izh_params, mea,
            sensory_channels=[0, 1, 2, 3, 4, 5, 6, 7],
            motor_regions={"up": [56, 57], "down": [62, 63]},
            game=Pong(),
            stdp_params=STDPParams(),
        )
        results = loop.run(key, duration_s=300, feedback="fep")
    """

    def __init__(
        self,
        network_params,
        neuron_params: IzhikevichParams,
        mea,
        sensory_channels: list[int],
        motor_regions: dict[str, list[int]],
        game,
        stdp_params: STDPParams | None = None,
    ) -> None:
        """Initialise the closed-loop controller.

        Parameters
        ----------
        network_params : NetworkParams
            Static network description (positions, is_excitatory, W_exc,
            W_inh, delays).
        neuron_params : IzhikevichParams
            Per-neuron Izhikevich model parameters.
        mea : MEA
            Virtual multi-electrode array instance.
        sensory_channels : list of int
            Eight electrode indices that serve as sensory input channels.
        motor_regions : dict
            ``{"up": [...], "down": [...]}`` mapping action names to lists
            of electrode indices that define the motor readout regions.
        game : Pong
            Game environment instance.
        stdp_params : STDPParams or None
            STDP hyperparameters.  Pass ``None`` to disable online
            plasticity during the experiment.
        """
        self.network_params = network_params
        self.neuron_params = neuron_params
        self.mea = mea
        self.sensory_channels = sensory_channels
        self.motor_regions = motor_regions
        self.game = game
        self.stdp_params = stdp_params

    def run(
        self,
        key: Array,
        duration_s: float = 300.0,
        dt_ms: float = 0.5,
        feedback: str = "fep",
        game_dt_ms: float = 20.0,
        decode_window_ms: float = 100.0,
    ) -> dict[str, Any]:
        """Run a closed-loop experiment.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        duration_s : float
            Total experiment duration in seconds (default 300).
        dt_ms : float
            Neural simulation timestep in ms (default 0.5).
        feedback : str
            Feedback mode -- ``"fep"``, ``"open_loop"``, or ``"silent"``.
        game_dt_ms : float
            Game update interval in ms (default 20).
        decode_window_ms : float
            Width of the spike-history window for motor decoding in ms
            (default 100).

        Returns
        -------
        dict
            - ``spike_history`` -- ``(T_ds, N)`` boolean numpy array
              (downsampled spike raster; empty if memory limit exceeded).
            - ``game_events`` -- list of ``(time_ms, "hit"|"miss")`` tuples.
            - ``rally_lengths`` -- ``(n_rallies,)`` int32 numpy array.
            - ``final_neuron_state`` -- NeuronState at experiment end.
            - ``final_syn_state`` -- SynapseState at experiment end.
            - ``final_game_state`` -- PongState at experiment end.
            - ``population_rates`` -- ``(n_game_steps, 2)`` numpy array of
              ``(total_spikes, step_index)`` per game update.
        """
        if feedback not in ("fep", "open_loop", "silent"):
            raise ValueError(f"Unknown feedback mode '{feedback}'.")

        # --- Unpack network params ---
        positions = self.network_params.positions
        n_neurons = positions.shape[0]
        W_exc = self.network_params.W_exc
        W_inh = self.network_params.W_inh
        is_excitatory = self.network_params.is_excitatory

        # Initialise neuron + synapse state
        b = self.neuron_params.b
        v0 = jnp.full(n_neurons, -65.0)
        neuron_state = NeuronState(v=v0, u=b * v0, spikes=jnp.zeros(n_neurons, dtype=jnp.bool_))
        syn_state = create_synapse_state(n_neurons)

        # Neuron-electrode map for motor decoding
        from bl1.mea.electrode import build_neuron_electrode_map
        neuron_electrode_map = build_neuron_electrode_map(
            positions, self.mea.positions, self.mea.detection_radius_um,
        )

        # --- Timing ---
        total_steps = int(duration_s * 1000.0 / dt_ms)
        steps_per_game = int(game_dt_ms / dt_ms)
        n_game_steps = total_steps // steps_per_game
        decode_window_steps = int(decode_window_ms / dt_ms)

        logger.info(
            "ClosedLoop.run: %d total steps, %d game updates, feedback=%s",
            total_steps, n_game_steps, feedback,
        )

        # --- STDP state ---
        has_plasticity = self.stdp_params is not None
        stdp_state = init_stdp_state(n_neurons)

        # --- Build scan step ---
        scan_step = _make_scan_step(
            self.neuron_params, W_inh, dt_ms,
            has_plasticity, self.stdp_params, is_excitatory,
        )

        # --- Recording buffers ---
        spike_window = np.zeros((decode_window_steps, n_neurons), dtype=np.float32)

        # Downsampled spike history
        ds_factor = max(int(1.0 / dt_ms), 1)
        max_raster_rows = 10_000_000
        total_ds_rows = n_game_steps * (steps_per_game // ds_factor)
        record_full_raster = (total_ds_rows * n_neurons) < max_raster_rows
        spike_raster_list: list[np.ndarray] = []

        game_events: list[tuple[float, str]] = []
        population_rates: list[tuple[int, int]] = []
        rally_counter: int = 0
        rally_lengths_list: list[int] = []
        phase_acc: float = 0.0

        # Initialise game state
        key, key_game = jax.random.split(key)
        game_state = self.game.reset(key_game)

        # --- Main loop ---
        for g_step in range(n_game_steps):
            time_ms = g_step * game_dt_ms
            key, key_fb, key_game_step = jax.random.split(key, 3)

            # 1. Decode motor action
            spike_window_jax = jnp.array(spike_window)
            action, _rates = decode_motor(
                spike_window_jax, neuron_electrode_map, self.motor_regions,
            )

            # 2. Step the game
            game_state, event_code = self.game.step(game_state, action, key_game_step)

            # 3. Handle events and feedback
            feedback_current = jnp.zeros(n_neurons)
            is_event = (event_code == EVENT_HIT) | (event_code == EVENT_MISS)

            if bool(event_code == EVENT_HIT):
                game_events.append((time_ms, "hit"))
                rally_counter += 1
            elif bool(event_code == EVENT_MISS):
                game_events.append((time_ms, "miss"))
                rally_lengths_list.append(rally_counter)
                rally_counter = 0

            if bool(is_event):
                if feedback == "fep":
                    feedback_current = _feedback_fep(key_fb, int(event_code), n_neurons)
                elif feedback == "open_loop":
                    feedback_current = _feedback_open_loop(key_fb, n_neurons)

            # 4. Encode sensory input
            active_electrodes, stim_amp, phase_acc = encode_sensory(
                game_state, self.mea.config, self.sensory_channels,
                dt_game_ms=game_dt_ms, _phase_accumulator=phase_acc,
            )

            # Build sensory stimulation current
            sensory_current = jnp.zeros(n_neurons)
            if active_electrodes and stim_amp > 0.0:
                for e_idx in active_electrodes:
                    neuron_mask = neuron_electrode_map[e_idx]
                    sensory_current = sensory_current + neuron_mask.astype(jnp.float32) * stim_amp

            # 5. External current for inner sim block
            I_external = jnp.tile(sensory_current[None, :], (steps_per_game, 1))
            I_external = I_external.at[0].add(feedback_current)

            # 6. Inner simulation (jax.lax.scan)
            carry = (neuron_state, syn_state, stdp_state, W_exc)
            carry, spikes_block = jax.lax.scan(scan_step, carry, I_external)
            neuron_state, syn_state, stdp_state, W_exc = carry

            # 7. Update rolling spike window
            spikes_block_np = np.asarray(spikes_block, dtype=np.float32)
            if steps_per_game >= decode_window_steps:
                spike_window[:] = spikes_block_np[-decode_window_steps:]
            else:
                spike_window = np.roll(spike_window, -steps_per_game, axis=0)
                spike_window[-steps_per_game:] = spikes_block_np

            # 8. Record
            total_spikes = int(jnp.sum(spikes_block))
            population_rates.append((total_spikes, g_step * steps_per_game))

            if record_full_raster:
                spike_raster_list.append(np.asarray(spikes_block[::ds_factor], dtype=np.bool_))

            if g_step > 0 and g_step % 1000 == 0:
                logger.info("  t=%.1f s  step=%d  events=%d  spikes=%d",
                            time_ms / 1000, g_step, len(game_events), total_spikes)

        # --- Finalise ---
        if rally_counter > 0:
            rally_lengths_list.append(rally_counter)

        spike_history = (
            np.concatenate(spike_raster_list, axis=0)
            if spike_raster_list
            else np.empty((0, n_neurons), dtype=np.bool_)
        )

        return {
            "spike_history": spike_history,
            "game_events": game_events,
            "rally_lengths": np.array(rally_lengths_list, dtype=np.int32),
            "final_neuron_state": neuron_state,
            "final_syn_state": syn_state,
            "final_game_state": game_state,
            "population_rates": np.array(population_rates),
        }
