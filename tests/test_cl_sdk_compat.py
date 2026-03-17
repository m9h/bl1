"""Tests for the CL-SDK compatibility layer (bl1.compat.cl_sdk).

All tests use small networks (n_neurons=1000) for speed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import bl1.compat.cl_sdk as cl
from bl1.compat.cl_sdk import (
    BurstDesign,
    ChannelSet,
    DataStream,
    Neurons,
    Recording,
    SpikeEvent,
    StimDesign,
    Tick,
    TickAnalysis,
)

# Small network for fast tests
N_NEURONS = 1000


# ---------------------------------------------------------------------------
# 1. Context manager
# ---------------------------------------------------------------------------


def test_open_context_manager():
    """``cl.open(n_neurons=1000)`` should yield a Neurons instance."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        assert isinstance(neurons, Neurons)
        assert neurons.n_neurons == N_NEURONS


# ---------------------------------------------------------------------------
# 2. ChannelSet
# ---------------------------------------------------------------------------


def test_channel_set():
    """ChannelSet(1, 2, 3).channels should equal [1, 2, 3]."""
    cs = ChannelSet(1, 2, 3)
    assert cs.channels == [1, 2, 3]


def test_channel_set_iteration():
    """ChannelSet should be iterable."""
    cs = ChannelSet(10, 20)
    assert list(cs) == [10, 20]
    assert len(cs) == 2


# ---------------------------------------------------------------------------
# 3. StimDesign
# ---------------------------------------------------------------------------


def test_stim_design():
    """StimDesign fields should be accessible by name."""
    sd = StimDesign(
        phase1_duration_us=200.0,
        phase1_amplitude_uA=-3.0,
        phase2_duration_us=200.0,
        phase2_amplitude_uA=3.0,
    )
    assert sd.phase1_duration_us == 200.0
    assert sd.phase1_amplitude_uA == -3.0
    assert sd.phase2_duration_us == 200.0
    assert sd.phase2_amplitude_uA == 3.0


# ---------------------------------------------------------------------------
# 4. BurstDesign
# ---------------------------------------------------------------------------


def test_burst_design():
    """BurstDesign fields should be accessible by name."""
    bd = BurstDesign(burst_count=5, frequency_hz=100.0)
    assert bd.burst_count == 5
    assert bd.frequency_hz == 100.0


# ---------------------------------------------------------------------------
# 5. Loop yields Tick objects
# ---------------------------------------------------------------------------


def test_loop_yields_ticks():
    """``neurons.loop(ticks_per_second=10)`` should yield Tick objects."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        ticks = list(itertools.islice(neurons.loop(ticks_per_second=10), 3))

    assert len(ticks) == 3
    for t in ticks:
        assert isinstance(t, Tick)
        assert isinstance(t.timestamp, float)
        assert isinstance(t.analysis, TickAnalysis)


# ---------------------------------------------------------------------------
# 6. Tick has spikes list
# ---------------------------------------------------------------------------


def test_tick_has_spikes():
    """Each tick should have a tick.analysis.spikes list."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        tick = next(neurons.loop(ticks_per_second=10))

    assert isinstance(tick.analysis.spikes, list)
    # Each element should be a SpikeEvent (list may be empty on the very first tick)
    for spike in tick.analysis.spikes:
        assert isinstance(spike, SpikeEvent)


# ---------------------------------------------------------------------------
# 7. Stim and collect spikes
# ---------------------------------------------------------------------------


def test_stim_and_collect():
    """Apply stimulation, run a few ticks, and collect spike events.

    With stimulation driving the culture we expect at least some spikes
    across a handful of ticks.
    """
    with cl.open(n_neurons=N_NEURONS, seed=99) as neurons:
        # Pick several non-forbidden channels for stimulation
        stim_channels = ChannelSet(
            *[ch for ch in range(8, 24) if ch not in neurons.forbidden_channels]
        )
        stim_design = StimDesign(
            phase1_duration_us=200.0,
            phase1_amplitude_uA=-4.0,
            phase2_duration_us=200.0,
            phase2_amplitude_uA=4.0,
        )
        burst_design = BurstDesign(burst_count=5, frequency_hz=200.0)

        neurons.stim(stim_channels, stim_design, burst_design)

        all_spikes: list[SpikeEvent] = []
        for tick in itertools.islice(neurons.loop(ticks_per_second=10), 5):
            all_spikes.extend(tick.analysis.spikes)

    # With stimulation on 16 channels over 5 ticks (500 ms simulated)
    # we expect at least a few spikes from a 1000-neuron network.
    assert len(all_spikes) > 0, (
        "Expected at least some spikes after stimulation"
    )


# ---------------------------------------------------------------------------
# 8. Interrupt stops stimulation
# ---------------------------------------------------------------------------


def test_interrupt_stops_stim():
    """After calling interrupt(), the stimulation dict should be empty."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        channels = ChannelSet(10, 11, 12)
        stim_design = StimDesign(200.0, -3.0, 200.0, 3.0)
        burst_design = BurstDesign(3, 100.0)

        neurons.stim(channels, stim_design, burst_design)
        assert len(neurons._active_stim) > 0

        neurons.interrupt(channels)
        # All three channels should have been removed
        for ch in channels.channels:
            assert ch not in neurons._active_stim


# ---------------------------------------------------------------------------
# 9. Forbidden channels
# ---------------------------------------------------------------------------


def test_forbidden_channels():
    """Stimulation on forbidden channels should be silently ignored."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        # Channel 0 is forbidden on the real CL1 hardware
        forbidden = ChannelSet(0, 7, 63)
        stim_design = StimDesign(200.0, -3.0, 200.0, 3.0)
        burst_design = BurstDesign(3, 100.0)

        neurons.stim(forbidden, stim_design, burst_design)
        # None of the forbidden channels should appear in active_stim
        for ch in forbidden.channels:
            assert ch not in neurons._active_stim


# ---------------------------------------------------------------------------
# 10. Record and DataStream
# ---------------------------------------------------------------------------


def test_record_and_datastream():
    """record() and create_data_stream() should return valid objects."""
    with cl.open(n_neurons=N_NEURONS) as neurons:
        rec = neurons.record(
            file_suffix="test",
            file_location="/tmp",
            attributes={"experiment": "unit_test"},
        )
        assert isinstance(rec, Recording)
        assert rec.file_suffix == "test"
        assert rec._stopped is False
        rec.stop()
        assert rec._stopped is True

        ds = neurons.create_data_stream(
            name="actions",
            attributes={"type": "motor"},
        )
        assert isinstance(ds, DataStream)
        assert ds.name == "actions"
        ds.append(0.0, {"action": "left"})
        assert len(ds.events) == 1


# ---------------------------------------------------------------------------
# 11. SpikeEvent has .channel attribute
# ---------------------------------------------------------------------------


def test_spike_event_has_channel():
    """Each SpikeEvent should have a .channel int attribute."""
    se = SpikeEvent(channel=42, timestamp=1.23)
    assert se.channel == 42
    assert se.timestamp == 1.23


def test_spike_event_from_simulation():
    """Spike events generated by the simulation should have valid channels."""
    with cl.open(n_neurons=N_NEURONS, seed=77) as neurons:
        # Drive the network to produce spikes
        channels = ChannelSet(
            *[ch for ch in range(16, 48) if ch not in neurons.forbidden_channels]
        )
        sd = StimDesign(200.0, -4.0, 200.0, 4.0)
        bd = BurstDesign(5, 200.0)
        neurons.stim(channels, sd, bd)

        all_spikes: list[SpikeEvent] = []
        for tick in itertools.islice(neurons.loop(ticks_per_second=10), 5):
            all_spikes.extend(tick.analysis.spikes)

    for spike in all_spikes:
        assert hasattr(spike, "channel")
        assert isinstance(spike.channel, int)
        # Channel should be a valid electrode index, not in forbidden set
        assert 0 <= spike.channel < 64
        assert spike.channel not in neurons.forbidden_channels
