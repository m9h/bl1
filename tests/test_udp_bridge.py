"""Tests for the UDP bridge protocol helpers and VirtualCL1Server init."""

import struct
import numpy as np
import pytest

from bl1.compat.udp_bridge import (
    NUM_CHANNEL_SETS,
    STIM_PACKET_SIZE,
    SPIKE_PACKET_SIZE,
    STIM_FORMAT,
    SPIKE_FORMAT,
    CHANNEL_GROUPS,
    pack_spike_data,
    unpack_stimulation_command,
    unpack_feedback_command,
    VirtualCL1Server,
)


# ---------------------------------------------------------------------------
# Packet round-trip tests
# ---------------------------------------------------------------------------


def _make_stim_packet(freqs, amps):
    """Helper: build a stim packet from arrays."""
    import time

    ts = int(time.time() * 1_000_000)
    return struct.pack(STIM_FORMAT, ts, *freqs.tolist(), *amps.tolist())


class TestPackUnpack:
    def test_spike_roundtrip(self):
        counts = np.array([1, 0, 3, 2, 0, 5, 1, 4], dtype=np.float32)
        pkt = pack_spike_data(counts)
        assert len(pkt) == SPIKE_PACKET_SIZE
        vals = struct.unpack(SPIKE_FORMAT, pkt)
        recovered = np.array(vals[1:], dtype=np.float32)
        np.testing.assert_allclose(recovered, counts)

    def test_stim_roundtrip(self):
        freqs = np.arange(1, 9, dtype=np.float32) * 5.0
        amps = np.arange(1, 9, dtype=np.float32) * 0.3
        pkt = _make_stim_packet(freqs, amps)
        assert len(pkt) == STIM_PACKET_SIZE
        ts, f, a = unpack_stimulation_command(pkt)
        assert ts > 0
        np.testing.assert_allclose(f, freqs)
        np.testing.assert_allclose(a, amps)

    def test_feedback_roundtrip(self):
        """Build a minimal feedback packet and unpack it."""
        channels = [8, 9, 10]
        channels_arr = [0xFF] * 64
        for i, ch in enumerate(channels):
            channels_arr[i] = ch
        event_name = b"enemy_kill" + b"\x00" * 22  # 32 bytes
        import time

        ts = int(time.time() * 1_000_000)
        pkt = struct.pack(
            "<QBB64BIfIB32sx",
            ts,
            1,  # type = event
            len(channels),
            *channels_arr,
            20,  # freq
            2.5,  # amp
            40,  # pulses
            0,  # not unpredictable
            event_name,
        )
        assert len(pkt) == 120
        (
            ts2,
            fb_type,
            chs,
            freq,
            amp,
            pulses,
            unp,
            name,
        ) = unpack_feedback_command(pkt)
        assert fb_type == "event"
        assert chs == [8, 9, 10]
        assert freq == 20
        assert abs(amp - 2.5) < 0.01
        assert name == "enemy_kill"


# ---------------------------------------------------------------------------
# VirtualCL1Server unit tests (no actual sockets)
# ---------------------------------------------------------------------------


class TestVirtualCL1:
    def test_init(self):
        server = VirtualCL1Server(
            training_host="localhost",
            n_neurons=500,
            seed=0,
        )
        assert server.neurons.n_neurons == 500

    def test_channel_lookup(self):
        server = VirtualCL1Server(
            training_host="localhost", n_neurons=500, seed=0
        )
        # Encoding channel 8 should map to group 0
        assert server.channel_lookup[8] == 0
        # Attack channel 32 should map to group 7
        assert server.channel_lookup[32] == 7

    def test_apply_stimulation(self):
        server = VirtualCL1Server(
            training_host="localhost", n_neurons=500, seed=0
        )
        freqs = np.ones(8, dtype=np.float32) * 20.0
        amps = np.ones(8, dtype=np.float32) * 2.0
        server.apply_stimulation(freqs, amps)
        # Should not crash; stim state should be populated
        assert len(server.neurons._active_stim) > 0

    def test_collect_spikes_shape(self):
        server = VirtualCL1Server(
            training_host="localhost", n_neurons=500, seed=0
        )
        # Run one tick
        tick = next(server.neurons.loop(ticks_per_second=10))
        counts = server.collect_spikes(tick)
        assert counts.shape == (NUM_CHANNEL_SETS,)
        assert counts.dtype == np.float32
