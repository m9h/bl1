"""UDP bridge — run BL-1 as a virtual CL1 device for doom-neuron.

Implements the binary UDP protocol from doom-neuron so that BL-1's
simulated culture can be used as a drop-in replacement for the real
CL1 neural hardware.

Protocol (matching doom-neuron/udp_protocol.py):
    Stimulation (Training -> CL1): 72 bytes
        [8B timestamp][32B frequencies][32B amplitudes]
    Spikes (CL1 -> Training): 40 bytes
        [8B timestamp][32B spike_counts]
    Feedback (Training -> CL1): 120 bytes
        [8B ts][1B type][1B n_ch][64B channels][4B freq][4B amp][4B pulses][1B unpredict][32B name][1B pad]

Usage:
    python -m bl1.compat.udp_bridge --training-host localhost --tick-frequency 10
"""

from __future__ import annotations

import argparse
import struct
import time
import numpy as np

from bl1.compat.cl_sdk import Neurons, ChannelSet, StimDesign, BurstDesign


# Protocol constants (must match doom-neuron/udp_protocol.py)
NUM_CHANNEL_SETS = 8
STIM_PACKET_SIZE = 8 + (NUM_CHANNEL_SETS * 2 * 4)   # 72
SPIKE_PACKET_SIZE = 8 + (NUM_CHANNEL_SETS * 4)        # 40
FEEDBACK_PACKET_SIZE = 120

STIM_FORMAT = "<Q" + ("f" * NUM_CHANNEL_SETS * 2)
SPIKE_FORMAT = "<Q" + ("f" * NUM_CHANNEL_SETS)

# Channel groups matching doom-neuron's CL1Config
CHANNEL_GROUPS = [
    ("encoding", [8, 9, 10, 17, 18, 25, 27, 28]),
    ("move_forward", [41, 42, 49]),
    ("move_backward", [50, 51, 58]),
    ("move_left", [13, 14, 21]),
    ("move_right", [45, 46, 53]),
    ("turn_left", [29, 30, 31, 37]),
    ("turn_right", [59, 60, 61, 62]),
    ("attack", [32, 33, 34]),
]


# ---------------------------------------------------------------------------
# Packet helpers
# ---------------------------------------------------------------------------

def pack_spike_data(spike_counts: np.ndarray) -> bytes:
    """Pack spike counts into a 40-byte UDP packet."""
    timestamp = int(time.time() * 1_000_000)
    return struct.pack(SPIKE_FORMAT, timestamp, *spike_counts.astype(np.float32).tolist())


def unpack_stimulation_command(packet: bytes):
    """Unpack a 72-byte stimulation command packet."""
    values = struct.unpack(STIM_FORMAT, packet)
    timestamp = values[0]
    frequencies = np.array(values[1 : NUM_CHANNEL_SETS + 1], dtype=np.float32)
    amplitudes = np.array(values[NUM_CHANNEL_SETS + 1 :], dtype=np.float32)
    return timestamp, frequencies, amplitudes


def unpack_feedback_command(packet: bytes):
    """Unpack a 120-byte feedback command packet."""
    unpacked = struct.unpack("<QBB64BIfIB32sx", packet)
    timestamp = unpacked[0]
    type_byte = unpacked[1]
    num_channels = unpacked[2]
    channels_array = unpacked[3:67]
    frequency = unpacked[67]
    amplitude = unpacked[68]
    pulses = unpacked[69]
    unpredictable_byte = unpacked[70]
    event_name_bytes = unpacked[71]

    type_map = {0: "interrupt", 1: "event", 2: "reward"}
    feedback_type = type_map.get(type_byte, "unknown")
    channels = [ch for ch in channels_array[:num_channels] if ch != 0xFF]
    event_name = event_name_bytes.rstrip(b"\x00").decode("utf-8")
    unpredictable = unpredictable_byte != 0

    return timestamp, feedback_type, channels, frequency, amplitude, pulses, unpredictable, event_name


# ---------------------------------------------------------------------------
# Virtual CL1 Server
# ---------------------------------------------------------------------------

class VirtualCL1Server:
    """A virtual CL1 server backed by BL-1's simulated culture.

    Drop-in replacement for cl1_neural_interface.py — speaks the same
    binary UDP protocol so doom-neuron's training_server.py can connect
    without modification.
    """

    def __init__(
        self,
        training_host: str,
        stim_port: int = 12345,
        spike_port: int = 12346,
        event_port: int = 12347,
        feedback_port: int = 12348,
        tick_frequency_hz: int = 10,
        n_neurons: int = 100_000,
        seed: int = 42,
    ):
        self.training_host = training_host
        self.stim_port = stim_port
        self.spike_port = spike_port
        self.event_port = event_port
        self.feedback_port = feedback_port
        self.tick_frequency_hz = tick_frequency_hz

        self.neurons = Neurons(n_neurons=n_neurons, seed=seed)

        self.channel_lookup: dict[int, int] = {}
        for idx, (_, channel_list) in enumerate(CHANNEL_GROUPS):
            for ch in channel_list:
                self.channel_lookup[ch] = idx

        self.packets_received = 0
        self.packets_sent = 0
        self.total_spikes = 0
        self.feedback_received = 0

    def apply_stimulation(self, frequencies: np.ndarray, amplitudes: np.ndarray):
        """Apply stimulation from training server to simulated culture."""
        encoding_channels = CHANNEL_GROUPS[0][1]
        all_channels: list[int] = []
        for _, chs in CHANNEL_GROUPS:
            all_channels.extend(chs)
        self.neurons.interrupt(ChannelSet(*all_channels))

        for i, channel_num in enumerate(encoding_channels):
            if i >= len(frequencies):
                break
            amp = float(amplitudes[i])
            freq = int(frequencies[i])
            if freq > 0 and amp > 0:
                self.neurons.stim(
                    ChannelSet(channel_num),
                    StimDesign(120, -amp, 120, amp),
                    BurstDesign(1, freq),
                )

    def apply_feedback(self, feedback_type, channels, frequency, amplitude, pulses, unpredictable):
        """Apply feedback stimulation to simulated culture."""
        if feedback_type == "interrupt":
            if channels:
                self.neurons.interrupt(ChannelSet(*channels))
            return
        if not channels or frequency <= 0 or amplitude <= 0:
            return
        self.neurons.stim(
            ChannelSet(*channels),
            StimDesign(120, -amplitude, 120, amplitude),
            BurstDesign(pulses, frequency),
        )

    def collect_spikes(self, tick) -> np.ndarray:
        """Count spikes per channel group from a tick."""
        spike_counts = np.zeros(NUM_CHANNEL_SETS, dtype=np.float32)
        for spike in tick.analysis.spikes:
            idx = self.channel_lookup.get(spike.channel)
            if idx is not None:
                spike_counts[idx] += 1
                self.total_spikes += 1
        return spike_counts

    def run(self):
        """Main loop — drop-in replacement for cl1_neural_interface.py."""
        import socket

        print("=" * 70)
        print("BL-1 Virtual CL1 Server")
        print(f"Neurons: {self.neurons.n_neurons} | Tick: {self.tick_frequency_hz} Hz")
        print(f"Training host: {self.training_host}")
        print("=" * 70)

        stim_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stim_sock.bind(("0.0.0.0", self.stim_port))
        stim_sock.setblocking(False)

        spike_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        event_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        event_sock.bind(("0.0.0.0", self.event_port))
        event_sock.setblocking(False)

        fb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fb_sock.bind(("0.0.0.0", self.feedback_port))
        fb_sock.setblocking(False)

        last_stats = time.time()
        tick_count = 0

        try:
            for tick in self.neurons.loop(ticks_per_second=self.tick_frequency_hz):
                tick_count += 1

                try:
                    pkt, _ = stim_sock.recvfrom(STIM_PACKET_SIZE)
                    _, freqs, amps = unpack_stimulation_command(pkt)
                    self.packets_received += 1
                    self.apply_stimulation(freqs, amps)
                except BlockingIOError:
                    pass

                spike_counts = self.collect_spikes(tick)
                try:
                    spike_sock.sendto(
                        pack_spike_data(spike_counts),
                        (self.training_host, self.spike_port),
                    )
                    self.packets_sent += 1
                except Exception:
                    pass

                try:
                    fb_pkt, _ = fb_sock.recvfrom(FEEDBACK_PACKET_SIZE)
                    _, fb_type, chs, freq, amp, pulses, unp, _ = unpack_feedback_command(fb_pkt)
                    self.apply_feedback(fb_type, chs, freq, amp, pulses, unp)
                    self.feedback_received += 1
                except BlockingIOError:
                    pass

                try:
                    evt_pkt, _ = event_sock.recvfrom(4096)
                    import json
                    hdr = struct.unpack("<QI", evt_pkt[:12])
                    data = json.loads(evt_pkt[12 : 12 + hdr[1]])
                    if data.get("event_type") == "training_complete":
                        print("Training complete. Shutting down.")
                        return
                except BlockingIOError:
                    pass
                except Exception:
                    pass

                if time.time() - last_stats >= 10.0:
                    elapsed = time.time() - last_stats
                    print(
                        f"ticks={tick_count} recv={self.packets_received / elapsed:.1f}/s "
                        f"send={self.packets_sent / elapsed:.1f}/s spikes={self.total_spikes}"
                    )
                    last_stats = time.time()
                    self.packets_received = 0
                    self.packets_sent = 0

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            stim_sock.close()
            spike_sock.close()
            event_sock.close()
            fb_sock.close()
            print(f"Total ticks: {tick_count}, Total spikes: {self.total_spikes}")


def main():
    parser = argparse.ArgumentParser(description="BL-1 Virtual CL1 Server")
    parser.add_argument("--training-host", type=str, required=True)
    parser.add_argument("--stim-port", type=int, default=12345)
    parser.add_argument("--spike-port", type=int, default=12346)
    parser.add_argument("--event-port", type=int, default=12347)
    parser.add_argument("--feedback-port", type=int, default=12348)
    parser.add_argument("--tick-frequency", type=int, default=10)
    parser.add_argument("--n-neurons", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    VirtualCL1Server(
        training_host=args.training_host,
        stim_port=args.stim_port,
        spike_port=args.spike_port,
        event_port=args.event_port,
        feedback_port=args.feedback_port,
        tick_frequency_hz=args.tick_frequency,
        n_neurons=args.n_neurons,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
