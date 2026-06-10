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
    Neural Stats (CL1 -> Training): 356 bytes  [port 12351]
        Extended per-electrode spike counts, firing rates, burst detection,
        synchrony metrics, and stimulation state.  Sent alongside the
        standard spike packet for backward compatibility.

Usage:
    python -m bl1.compat.udp_bridge --training-host localhost --tick-frequency 10
"""

from __future__ import annotations

import argparse
import struct
import time

import numpy as np

from bl1.compat.cl_sdk import BurstDesign, ChannelSet, Neurons, StimDesign
from bl1.compat.stats_protocol import (
    NUM_ELECTRODES,
    STATS_PORT,
    pack_neural_stats,
)

# Protocol constants (must match doom-neuron/udp_protocol.py)
NUM_CHANNEL_SETS = 8
STIM_PACKET_SIZE = 8 + (NUM_CHANNEL_SETS * 2 * 4)  # 72
SPIKE_PACKET_SIZE = 8 + (NUM_CHANNEL_SETS * 4)  # 40
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

    return (
        timestamp,
        feedback_type,
        channels,
        frequency,
        amplitude,
        pulses,
        unpredictable,
        event_name,
    )


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
        stats_port: int = STATS_PORT,
        no_stats: bool = False,
        tick_frequency_hz: int = 10,
        n_neurons: int = 100_000,
        seed: int = 42,
        vis_host: str = "0.0.0.0",
        vis_port: int = 12350,
        vis_fps: int = 3,
        no_vis: bool = False,
    ):
        self.training_host = training_host
        self.stim_port = stim_port
        self.spike_port = spike_port
        self.event_port = event_port
        self.feedback_port = feedback_port
        self.stats_port = stats_port
        self.no_stats = no_stats
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

        # -- extended stats tracking ---------------------------------------------
        self._stats_enabled = not no_stats
        self._tick_number: int = 0
        # Per-electrode spike counts (64 channels), recomputed each tick
        self._electrode_spikes = np.zeros(NUM_ELECTRODES, dtype=np.float32)
        # Exponential moving average firing rates per channel group (Hz)
        self._ema_rates = np.zeros(NUM_CHANNEL_SETS, dtype=np.float32)
        # Smoothing: 20-tick window (~2 s at 10 Hz)
        self._ema_smoothing_ticks = 20
        # Running population rate statistics for burst detection
        self._pop_rate_mean: float = 0.0
        self._pop_rate_var: float = 0.0
        self._pop_rate_alpha: float = 2.0 / (self._ema_smoothing_ticks + 1)
        # Current stimulation amplitudes being applied (per group)
        self._current_stim_amplitudes = np.zeros(NUM_CHANNEL_SETS, dtype=np.float32)

        # -- visualization -------------------------------------------------------
        self.no_vis = no_vis
        self.vis_fps = vis_fps
        self._mjpeg = None
        self._monitor = None

        if not no_vis:
            from bl1.monitor.activity import ActivityMonitor
            from bl1.monitor.mjpeg import NeuralMJPEGServer

            self._monitor = ActivityMonitor(n_neurons=n_neurons)
            self._mjpeg = NeuralMJPEGServer(
                host=vis_host,
                port=vis_port,
                path="/neural.mjpeg",
            )
            self._vis_port = vis_port

    def apply_stimulation(self, frequencies: np.ndarray, amplitudes: np.ndarray):
        """Apply stimulation from training server to simulated culture."""
        # Track stimulation amplitudes for the stats packet.  The encoding
        # group (index 0) receives per-electrode stimulation; we record the
        # mean amplitude for the group.  Other groups are not currently
        # stimulated via this path, so they stay at zero.
        self._current_stim_amplitudes[:] = 0.0
        encoding_channels = CHANNEL_GROUPS[0][1]
        all_channels: list[int] = []
        for _, chs in CHANNEL_GROUPS:
            all_channels.extend(chs)
        self.neurons.interrupt(ChannelSet(*all_channels))

        encoding_amp_sum = 0.0
        encoding_amp_count = 0
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
                encoding_amp_sum += amp
                encoding_amp_count += 1
        if encoding_amp_count > 0:
            self._current_stim_amplitudes[0] = encoding_amp_sum / encoding_amp_count

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
        """Count spikes per channel group from a tick.

        As a side-effect, populates ``self._electrode_spikes`` with
        per-electrode counts (64 channels) for the extended stats packet.
        """
        spike_counts = np.zeros(NUM_CHANNEL_SETS, dtype=np.float32)
        self._electrode_spikes[:] = 0.0
        for spike in tick.analysis.spikes:
            # Per-electrode tracking (0..63)
            if 0 <= spike.channel < NUM_ELECTRODES:
                self._electrode_spikes[spike.channel] += 1
            # Grouped counts (original behavior)
            idx = self.channel_lookup.get(spike.channel)
            if idx is not None:
                spike_counts[idx] += 1
                self.total_spikes += 1
        return spike_counts

    # ------------------------------------------------------------------
    # Extended neural stats
    # ------------------------------------------------------------------

    def _compute_and_send_stats(
        self,
        spike_counts: np.ndarray,
        stats_sock,
        timestamp_us: int,
    ) -> None:
        """Compute derived statistics and send the extended stats packet.

        This runs every tick when ``--no-stats`` is *not* set.  Designed to
        be fast (<1 ms) so it does not block the critical stimulation/spike
        path.

        Parameters
        ----------
        spike_counts : np.ndarray
            Grouped spike counts for this tick, shape ``(8,)``.
        stats_sock : socket.socket
            Pre-created UDP socket for stats packets.
        timestamp_us : int
            Timestamp for this packet (microseconds since epoch).
        """
        self._tick_number += 1

        # -- Firing rates (EMA) ------------------------------------------------
        # Convert spike counts to instantaneous Hz (counts / tick_duration_s)
        tick_duration_s = 1.0 / self.tick_frequency_hz
        current_rates = spike_counts / tick_duration_s
        alpha = 2.0 / (self._ema_smoothing_ticks + 1)
        self._ema_rates = (1.0 - alpha) * self._ema_rates + alpha * current_rates

        # -- Population rate ---------------------------------------------------
        total_electrode_spikes = float(self._electrode_spikes.sum())
        population_rate = total_electrode_spikes / tick_duration_s

        # -- Burst detection (population rate > 1.5 std above running mean) ----
        # Update running mean/variance with Welford-style EMA
        pop_diff = population_rate - self._pop_rate_mean
        self._pop_rate_mean += self._pop_rate_alpha * pop_diff
        self._pop_rate_var = (
            1.0 - self._pop_rate_alpha
        ) * self._pop_rate_var + self._pop_rate_alpha * pop_diff * pop_diff
        pop_std = max(self._pop_rate_var**0.5, 1e-6)
        burst_active = 1.0 if population_rate > (self._pop_rate_mean + 1.5 * pop_std) else 0.0

        # -- Synchrony index (Fano factor across electrodes) -------------------
        # Fano factor = var(counts) / mean(counts).  When all electrodes fire
        # equally the Fano factor is 0; Poisson gives ~1; bursting gives >1.
        e_mean = float(self._electrode_spikes.mean())
        if e_mean > 1e-9:
            e_var = float(self._electrode_spikes.var())
            synchrony_index = e_var / e_mean
        else:
            synchrony_index = 0.0

        # -- Pack and send -----------------------------------------------------
        try:
            packet = pack_neural_stats(
                timestamp_us=timestamp_us,
                tick_number=self._tick_number,
                electrode_spikes=self._electrode_spikes,
                firing_rates=self._ema_rates,
                population_rate=population_rate,
                burst_active=burst_active,
                synchrony_index=synchrony_index,
                stim_amplitudes=self._current_stim_amplitudes,
                total_neuron_spikes=total_electrode_spikes,
            )
            stats_sock.sendto(packet, (self.training_host, self.stats_port))
        except Exception:
            # Stats sending must never crash the critical path
            pass

    def run(self):
        """Main loop — drop-in replacement for cl1_neural_interface.py."""
        import socket

        print("=" * 70)
        print("BL-1 Virtual CL1 Server")
        print(f"Neurons: {self.neurons.n_neurons} | Tick: {self.tick_frequency_hz} Hz")
        print(f"Training host: {self.training_host}")
        if self._stats_enabled:
            print(f"Extended stats on port {self.stats_port}")
        else:
            print("Extended stats: disabled")
        if not self.no_vis:
            print(f"Neural visualization at http://localhost:{self._vis_port}/neural.mjpeg")
        print("=" * 70)

        # Start MJPEG streaming server
        if self._mjpeg is not None:
            self._mjpeg.start()

        stim_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stim_sock.bind(("0.0.0.0", self.stim_port))  # nosec B104
        stim_sock.setblocking(False)

        spike_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Stats socket (separate from the spike socket, new port)
        stats_sock: socket.socket | None = None
        if self._stats_enabled:
            stats_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        event_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        event_sock.bind(("0.0.0.0", self.event_port))  # nosec B104
        event_sock.setblocking(False)

        fb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fb_sock.bind(("0.0.0.0", self.feedback_port))  # nosec B104
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

                # -- visualization: record every tick, render at vis_fps --
                if self._monitor is not None:
                    self._monitor.record_tick(
                        spike_counts,
                        electrode_spikes=self._electrode_spikes,
                        timestamp=time.monotonic(),
                    )
                    # Render every N ticks to achieve vis_fps
                    render_interval = max(self.tick_frequency_hz // self.vis_fps, 1)
                    if tick_count % render_interval == 0:
                        frame = self._monitor.render_frame()
                        self._mjpeg.update_frame(frame)

                timestamp_us = int(time.time() * 1_000_000)
                try:
                    spike_sock.sendto(
                        pack_spike_data(spike_counts),
                        (self.training_host, self.spike_port),
                    )
                    self.packets_sent += 1
                except Exception:
                    pass

                # Send extended stats packet (non-critical, never blocks)
                if stats_sock is not None:
                    self._compute_and_send_stats(
                        spike_counts,
                        stats_sock,
                        timestamp_us,
                    )

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
            if self._mjpeg is not None:
                self._mjpeg.stop()
            stim_sock.close()
            spike_sock.close()
            if stats_sock is not None:
                stats_sock.close()
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
    # Extended neural stats
    parser.add_argument(
        "--stats-port",
        type=int,
        default=STATS_PORT,
        help=f"UDP port for extended neural stats (default: {STATS_PORT})",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable extended neural stats packet (port 12351)",
    )
    # Visualization
    parser.add_argument(
        "--vis-port",
        type=int,
        default=12350,
        help="Neural visualization MJPEG port (default: 12350)",
    )
    parser.add_argument(
        "--vis-host",
        type=str,
        default="0.0.0.0",
        help="Neural visualization bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable neural visualization MJPEG stream",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=3,
        help="Visualization render rate in Hz (default: 3)",
    )
    args = parser.parse_args()

    VirtualCL1Server(
        training_host=args.training_host,
        stim_port=args.stim_port,
        spike_port=args.spike_port,
        event_port=args.event_port,
        feedback_port=args.feedback_port,
        stats_port=args.stats_port,
        no_stats=args.no_stats,
        tick_frequency_hz=args.tick_frequency,
        n_neurons=args.n_neurons,
        seed=args.seed,
        vis_host=args.vis_host,
        vis_port=args.vis_port,
        vis_fps=args.vis_fps,
        no_vis=args.no_vis,
    ).run()


if __name__ == "__main__":
    main()
