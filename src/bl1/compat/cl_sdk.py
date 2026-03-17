"""CL-SDK compatible interface backed by BL-1 simulation.

Provides a drop-in replacement for Cortical Labs' ``cl`` Python SDK so
that code written for the real CL1 hardware can run unchanged against
BL-1's Izhikevich-based cortical culture simulator.

Usage (mirrors real CL-SDK exactly)::

    import bl1.compat.cl_sdk as cl

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=10):
            neurons.stim(channel_set, stim_design, burst_design)
            for spike in tick.analysis.spikes:
                print(spike.channel)
"""

from __future__ import annotations

import contextlib
import time
from typing import Generator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    izhikevich_step,
    create_population,
)
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.mea.electrode import MEA, build_neuron_electrode_map


# -----------------------------------------------------------------------
# Data classes mirroring the CL-SDK types
# -----------------------------------------------------------------------


class ChannelSet:
    """Mirrors ``cl.ChannelSet`` -- a set of electrode channel indices."""

    def __init__(self, *channels: int) -> None:
        self.channels: list[int] = list(channels)

    def __repr__(self) -> str:
        return f"ChannelSet({self.channels})"

    def __iter__(self):
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)


@dataclass
class StimDesign:
    """Mirrors ``cl.StimDesign`` -- biphasic pulse specification.

    Parameters follow CL-SDK conventions:
        phase1 is typically cathodic-first (negative amplitude),
        phase2 is the charge-balancing anodic phase (positive amplitude).
    """

    phase1_duration_us: float  # microseconds
    phase1_amplitude_uA: float  # microamps (typically negative)
    phase2_duration_us: float  # microseconds
    phase2_amplitude_uA: float  # microamps (typically positive)


@dataclass
class BurstDesign:
    """Mirrors ``cl.BurstDesign`` -- burst pattern specification."""

    burst_count: int  # number of pulses in the burst
    frequency_hz: float  # repetition frequency in Hz


class SpikeEvent:
    """Mirrors a spike event from ``cl.Tick.analysis.spikes``."""

    __slots__ = ("channel", "timestamp")

    def __init__(self, channel: int, timestamp: float = 0.0) -> None:
        self.channel = channel
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"SpikeEvent(channel={self.channel}, timestamp={self.timestamp:.3f})"


class TickAnalysis:
    """Analysis results attached to a single :class:`Tick`."""

    __slots__ = ("spikes",)

    def __init__(self, spikes: list[SpikeEvent]) -> None:
        self.spikes = spikes


class Tick:
    """Mirrors ``cl.Tick`` -- one timestep of neural data."""

    __slots__ = ("timestamp", "analysis")

    def __init__(self, timestamp: float, analysis: TickAnalysis) -> None:
        self.timestamp = timestamp
        self.analysis = analysis


class Recording:
    """Mirrors ``cl.Recording`` -- records neural data to file."""

    def __init__(
        self,
        file_suffix: str = "",
        file_location: str = ".",
        attributes: Optional[dict] = None,
    ) -> None:
        self.file_suffix = file_suffix
        self.file_location = file_location
        self.attributes = attributes or {}
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True


class DataStream:
    """Mirrors ``cl.DataStream`` -- event logging."""

    def __init__(
        self,
        name: str = "",
        attributes: Optional[dict] = None,
    ) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.events: list[tuple] = []

    def append(self, timestamp: float, data: object) -> None:
        self.events.append((timestamp, data))


# -----------------------------------------------------------------------
# Main hardware-equivalent: Neurons
# -----------------------------------------------------------------------


class Neurons:
    """Mirrors ``cl.Neurons`` -- the main hardware interface.

    Internally runs a BL-1 Izhikevich network on a 64-channel MEA,
    stepping the full simulation at each tick.

    Parameters
    ----------
    n_neurons : int
        Number of Izhikevich neurons in the simulated culture.
    seed : int
        PRNG seed for reproducibility.
    substrate_um : tuple
        (width, height) of the simulated culture dish in micrometres.
    """

    # Forbidden (corner) channels matching the real CL1 hardware
    forbidden_channels: frozenset[int] = frozenset({0, 4, 7, 56, 63})

    def __init__(
        self,
        n_neurons: int = 100_000,
        seed: int = 42,
        substrate_um: tuple[float, float] = (3000.0, 3000.0),
    ) -> None:
        self.n_neurons = n_neurons
        self.seed = seed

        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)

        # -- neuron population -----------------------------------------------
        self._izh_params, self._neuron_state, self._is_excitatory = (
            create_population(k1, n_neurons, ei_ratio=0.8)
        )

        # -- spatial layout --------------------------------------------------
        self._positions = place_neurons(k2, n_neurons, substrate_um)

        # -- connectivity (BCOO sparse) --------------------------------------
        self._W_exc, self._W_inh, self._delays = build_connectivity(
            k3,
            self._positions,
            self._is_excitatory,
            lambda_um=200.0,
            p_max=0.21,
            g_exc=0.05,
            g_inh=0.20,
        )

        # -- synaptic state --------------------------------------------------
        self._syn_state = create_synapse_state(n_neurons)

        # -- MEA -------------------------------------------------------------
        self._mea = MEA("cl1_64ch")
        self._neuron_electrode_map = build_neuron_electrode_map(
            self._positions,
            self._mea.positions,
            self._mea.detection_radius_um,
        )

        # -- stimulation bookkeeping -----------------------------------------
        # channel -> (StimDesign, BurstDesign, start_time_ms)
        self._active_stim: dict[int, tuple[StimDesign, BurstDesign, float]] = {}
        self._current_time: float = 0.0

        # External current accumulator (for manual injection if needed)
        self._I_ext = jnp.zeros(n_neurons)

    # -------------------------------------------------------------------
    # CL-SDK API: stimulation
    # -------------------------------------------------------------------

    def stim(
        self,
        channel_set: ChannelSet,
        stim_design: StimDesign,
        burst_design: BurstDesign,
    ) -> None:
        """Apply stimulation to the specified channels."""
        for ch in channel_set.channels:
            if ch not in self.forbidden_channels:
                self._active_stim[ch] = (stim_design, burst_design, self._current_time)

    def interrupt(self, channel_set: ChannelSet) -> None:
        """Stop stimulation on the specified channels."""
        for ch in channel_set.channels:
            self._active_stim.pop(ch, None)

    # -------------------------------------------------------------------
    # CL-SDK API: recording / data stream
    # -------------------------------------------------------------------

    def record(
        self,
        file_suffix: str = "",
        file_location: str = ".",
        attributes: Optional[dict] = None,
    ) -> Recording:
        """Start a recording (stub -- data is not persisted yet)."""
        return Recording(file_suffix, file_location, attributes or {})

    def create_data_stream(
        self,
        name: str = "",
        attributes: Optional[dict] = None,
    ) -> DataStream:
        """Create a named data stream for event logging."""
        return DataStream(name, attributes or {})

    # -------------------------------------------------------------------
    # Internal: stimulation -> current conversion
    # -------------------------------------------------------------------

    def _compute_stimulation_current(self) -> jnp.ndarray:
        """Convert active stimulation designs into a per-neuron current vector.

        The real CL1 hardware delivers biphasic current pulses.  We
        approximate this by injecting a scaled DC current to neurons near
        each stimulating electrode for as long as the stimulation remains
        active.

        The amplitude is derived from the anodic (depolarising) phase of
        the :class:`StimDesign` and scaled into Izhikevich current units.
        """
        I_stim = jnp.zeros(self.n_neurons)

        for ch, (stim_design, burst_design, _start_time) in self._active_stim.items():
            if ch >= 64 or ch in self.forbidden_channels:
                continue

            # Use the absolute anodic amplitude, scaled to model units
            amplitude = abs(stim_design.phase2_amplitude_uA)
            freq = burst_design.frequency_hz

            if freq > 0:
                # Scale micro-amps to Izhikevich current units.
                # A typical CL1 pulse of ~1-5 uA maps to ~5-25 Izh units.
                scale = min(amplitude * 5.0, 20.0)
            else:
                scale = 0.0

            # Inject into neurons within this electrode's detection radius
            neuron_mask = self._neuron_electrode_map[ch]
            I_stim = I_stim + neuron_mask.astype(jnp.float32) * scale

        return I_stim

    # -------------------------------------------------------------------
    # Internal: one simulation step
    # -------------------------------------------------------------------

    def _step_simulation(self, dt_ms: float = 0.5) -> list[SpikeEvent]:
        """Run one Izhikevich timestep and return per-electrode spike events."""
        # Currents
        I_stim = self._compute_stimulation_current()
        I_syn = compute_synaptic_current(self._syn_state, self._neuron_state.v)
        I_total = I_syn + I_stim + self._I_ext

        # Step neurons
        self._neuron_state = izhikevich_step(
            self._neuron_state, self._izh_params, I_total, dt_ms
        )

        # Update fast synapses (AMPA / GABA_A)
        spikes_f = self._neuron_state.spikes.astype(jnp.float32)
        new_g_ampa = ampa_step(self._syn_state.g_ampa, spikes_f, self._W_exc, dt_ms)
        new_g_gaba_a = gaba_a_step(
            self._syn_state.g_gaba_a, spikes_f, self._W_inh, dt_ms
        )
        self._syn_state = SynapseState(
            g_ampa=new_g_ampa,
            g_gaba_a=new_g_gaba_a,
            g_nmda_rise=self._syn_state.g_nmda_rise,
            g_nmda_decay=self._syn_state.g_nmda_decay,
            g_gaba_b_rise=self._syn_state.g_gaba_b_rise,
            g_gaba_b_decay=self._syn_state.g_gaba_b_decay,
        )

        self._current_time += dt_ms

        # Detect per-electrode spikes
        spike_events: list[SpikeEvent] = []
        spikes_np = np.asarray(self._neuron_state.spikes)
        ne_map_np = np.asarray(self._neuron_electrode_map)

        for electrode_idx in range(64):
            if electrode_idx in self.forbidden_channels:
                continue
            neurons_near = ne_map_np[electrode_idx]
            n_spikes = int(np.sum(spikes_np & neurons_near))
            for _ in range(n_spikes):
                spike_events.append(
                    SpikeEvent(channel=electrode_idx, timestamp=self._current_time)
                )

        return spike_events

    # -------------------------------------------------------------------
    # CL-SDK API: main loop
    # -------------------------------------------------------------------

    def loop(self, ticks_per_second: int = 10) -> Generator[Tick, None, None]:
        """Generate ticks at the specified rate, like real hardware.

        Each tick runs multiple simulation steps (dt = 0.5 ms) to cover
        the inter-tick interval, then yields a :class:`Tick` containing
        aggregated spike data.

        Parameters
        ----------
        ticks_per_second : int
            Number of ticks per simulated second.  Each tick advances
            the simulation by ``1000 / ticks_per_second`` ms.
        """
        tick_interval_ms = 1000.0 / ticks_per_second
        dt_ms = 0.5
        steps_per_tick = int(tick_interval_ms / dt_ms)

        while True:
            all_spike_events: list[SpikeEvent] = []
            for _ in range(steps_per_tick):
                events = self._step_simulation(dt_ms)
                all_spike_events.extend(events)

            yield Tick(
                timestamp=self._current_time,
                analysis=TickAnalysis(spikes=all_spike_events),
            )


# -----------------------------------------------------------------------
# Context manager (top-level entry point)
# -----------------------------------------------------------------------


@contextlib.contextmanager
def open(
    n_neurons: int = 100_000,
    seed: int = 42,
) -> Generator[Neurons, None, None]:
    """Context manager mimicking ``cl.open()``.

    Usage::

        import bl1.compat.cl_sdk as cl

        with cl.open() as neurons:
            for tick in neurons.loop(ticks_per_second=10):
                ...

    Parameters
    ----------
    n_neurons : int
        Number of simulated neurons (default 100 000).
    seed : int
        PRNG seed for reproducibility.
    """
    neurons = Neurons(n_neurons=n_neurons, seed=seed)
    try:
        yield neurons
    finally:
        # Cleanup hook (reserved for future resource management)
        pass
