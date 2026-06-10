"""Extended neural statistics protocol for BL-1 -> doom-neuron.

Sent on port 12351 alongside the standard spike packet (port 12346).
Backward-compatible: doom-neuron works fine without reading this port.

Packet Format (356 bytes):
    [8B  timestamp]             uint64, microseconds since epoch
    [4B  packet_version]        uint32, always 1 for now
    [4B  n_electrodes]          uint32, 64
    [4B  tick_number]           uint32, monotonic counter
    [64 x 4B electrode_spikes]  float32[64], per-electrode spike counts
    [8 x 4B firing_rates]       float32[8], smoothed Hz per channel group
    [4B  population_rate]       float32, overall population firing rate Hz
    [4B  burst_active]          float32, 1.0 if network burst detected, 0.0
    [4B  synchrony_index]       float32, Fano factor in current window
    [8 x 4B stim_amplitudes]    float32[8], current stimulation per group
    [4B  total_neuron_spikes]   float32, total spikes across all neurons

Usage (BL-1 sender side)::

    from bl1.compat.stats_protocol import pack_neural_stats, STATS_PORT

    packet = pack_neural_stats(
        timestamp_us=int(time.time() * 1_000_000),
        tick_number=42,
        electrode_spikes=np.zeros(64, dtype=np.float32),
        firing_rates=np.zeros(8, dtype=np.float32),
        population_rate=0.0,
        burst_active=0.0,
        synchrony_index=1.0,
        stim_amplitudes=np.zeros(8, dtype=np.float32),
        total_neuron_spikes=0.0,
    )
    sock.sendto(packet, (host, STATS_PORT))

Usage (doom-neuron receiver side)::

    from neural_stats_receiver import NeuralStatsReceiver

    receiver = NeuralStatsReceiver(port=12351)
    receiver.start()
    stats = receiver.get_latest()  # non-blocking
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATS_PORT = 12351
STATS_PACKET_VERSION = 1
NUM_ELECTRODES = 64
NUM_CHANNEL_SETS = 8

# Packet size breakdown:
#   8  (timestamp)
# + 4  (version)
# + 4  (n_electrodes)
# + 4  (tick_number)
# + 256 (64 x float32 electrode spikes)
# + 32  (8 x float32 firing rates)
# + 4  (population_rate)
# + 4  (burst_active)
# + 4  (synchrony_index)
# + 32  (8 x float32 stim amplitudes)
# + 4  (total_neuron_spikes)
# = 356
STATS_PACKET_SIZE = 356

# struct format: little-endian
#   Q   = uint64  timestamp
#   I   = uint32  version
#   I   = uint32  n_electrodes
#   I   = uint32  tick_number
#   64f = float32 electrode_spikes[64]
#   8f  = float32 firing_rates[8]
#   f   = float32 population_rate
#   f   = float32 burst_active
#   f   = float32 synchrony_index
#   8f  = float32 stim_amplitudes[8]
#   f   = float32 total_neuron_spikes
_HEADER_FMT = "<QIII"
_BODY_FMT = f"{NUM_ELECTRODES}f{NUM_CHANNEL_SETS}ffff{NUM_CHANNEL_SETS}ff"
STATS_FORMAT = _HEADER_FMT + _BODY_FMT

# Sanity check: struct.calcsize should match STATS_PACKET_SIZE
assert struct.calcsize(STATS_FORMAT) == STATS_PACKET_SIZE, (
    f"Format size {struct.calcsize(STATS_FORMAT)} != expected {STATS_PACKET_SIZE}"
)


# ---------------------------------------------------------------------------
# Pack / Unpack
# ---------------------------------------------------------------------------


def pack_neural_stats(
    timestamp_us: int,
    tick_number: int,
    electrode_spikes: np.ndarray,
    firing_rates: np.ndarray,
    population_rate: float,
    burst_active: float,
    synchrony_index: float,
    stim_amplitudes: np.ndarray,
    total_neuron_spikes: float,
    *,
    version: int = STATS_PACKET_VERSION,
    n_electrodes: int = NUM_ELECTRODES,
) -> bytes:
    """Pack extended neural statistics into a binary UDP packet.

    Parameters
    ----------
    timestamp_us : int
        Timestamp in microseconds since epoch.
    tick_number : int
        Monotonically increasing tick counter.
    electrode_spikes : np.ndarray
        Per-electrode spike counts, shape ``(64,)``.
    firing_rates : np.ndarray
        Smoothed firing rates per channel group (Hz), shape ``(8,)``.
    population_rate : float
        Overall population firing rate (Hz).
    burst_active : float
        1.0 if a network burst is detected, 0.0 otherwise.
    synchrony_index : float
        Fano factor of spike counts across electrodes.
    stim_amplitudes : np.ndarray
        Current stimulation amplitude per channel group, shape ``(8,)``.
    total_neuron_spikes : float
        Total neuron-level spikes across all neurons this tick.
    version : int
        Packet version (default 1).
    n_electrodes : int
        Number of electrodes (default 64).

    Returns
    -------
    bytes
        356-byte binary packet ready for UDP transmission.

    Raises
    ------
    ValueError
        If any array has an unexpected shape.
    """
    electrode_spikes = np.asarray(electrode_spikes, dtype=np.float32)
    firing_rates = np.asarray(firing_rates, dtype=np.float32)
    stim_amplitudes = np.asarray(stim_amplitudes, dtype=np.float32)

    if electrode_spikes.shape != (NUM_ELECTRODES,):
        raise ValueError(
            f"electrode_spikes must have shape ({NUM_ELECTRODES},), got {electrode_spikes.shape}"
        )
    if firing_rates.shape != (NUM_CHANNEL_SETS,):
        raise ValueError(
            f"firing_rates must have shape ({NUM_CHANNEL_SETS},), got {firing_rates.shape}"
        )
    if stim_amplitudes.shape != (NUM_CHANNEL_SETS,):
        raise ValueError(
            f"stim_amplitudes must have shape ({NUM_CHANNEL_SETS},), got {stim_amplitudes.shape}"
        )

    packet = struct.pack(
        STATS_FORMAT,
        int(timestamp_us),
        int(version),
        int(n_electrodes),
        int(tick_number),
        *electrode_spikes.tolist(),
        *firing_rates.tolist(),
        float(population_rate),
        float(burst_active),
        float(synchrony_index),
        *stim_amplitudes.tolist(),
        float(total_neuron_spikes),
    )

    assert len(packet) == STATS_PACKET_SIZE, (
        f"Packet size mismatch: {len(packet)} != {STATS_PACKET_SIZE}"
    )
    return packet


def unpack_neural_stats(packet: bytes) -> dict[str, Any]:
    """Unpack a neural stats packet into a dictionary.

    Parameters
    ----------
    packet : bytes
        Raw 356-byte packet received from UDP.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``timestamp_us`` (int): Microsecond timestamp.
        - ``packet_version`` (int): Protocol version.
        - ``n_electrodes`` (int): Number of electrodes.
        - ``tick_number`` (int): Monotonic tick counter.
        - ``electrode_spikes`` (np.ndarray): shape ``(64,)`` float32.
        - ``firing_rates`` (np.ndarray): shape ``(8,)`` float32.
        - ``population_rate`` (float): Overall population rate Hz.
        - ``burst_active`` (float): 1.0 or 0.0.
        - ``synchrony_index`` (float): Fano factor.
        - ``stim_amplitudes`` (np.ndarray): shape ``(8,)`` float32.
        - ``total_neuron_spikes`` (float): Total neuron spikes this tick.

    Raises
    ------
    ValueError
        If *packet* has an unexpected size.
    """
    if len(packet) != STATS_PACKET_SIZE:
        raise ValueError(f"Expected {STATS_PACKET_SIZE} bytes, got {len(packet)}")

    values = struct.unpack(STATS_FORMAT, packet)

    # Walk through the flat tuple using offsets
    idx = 0

    timestamp_us = values[idx]
    idx += 1
    version = values[idx]
    idx += 1
    n_electrodes = values[idx]
    idx += 1
    tick_number = values[idx]
    idx += 1

    electrode_spikes = np.array(values[idx : idx + NUM_ELECTRODES], dtype=np.float32)
    idx += NUM_ELECTRODES

    firing_rates = np.array(values[idx : idx + NUM_CHANNEL_SETS], dtype=np.float32)
    idx += NUM_CHANNEL_SETS

    population_rate = values[idx]
    idx += 1
    burst_active = values[idx]
    idx += 1
    synchrony_index = values[idx]
    idx += 1

    stim_amplitudes = np.array(values[idx : idx + NUM_CHANNEL_SETS], dtype=np.float32)
    idx += NUM_CHANNEL_SETS

    total_neuron_spikes = values[idx]
    idx += 1

    return {
        "timestamp_us": timestamp_us,
        "packet_version": version,
        "n_electrodes": n_electrodes,
        "tick_number": tick_number,
        "electrode_spikes": electrode_spikes,
        "firing_rates": firing_rates,
        "population_rate": population_rate,
        "burst_active": burst_active,
        "synchrony_index": synchrony_index,
        "stim_amplitudes": stim_amplitudes,
        "total_neuron_spikes": total_neuron_spikes,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    print("Testing stats protocol...")

    ts = int(_time.time() * 1_000_000)
    e_spikes = np.random.poisson(2, size=NUM_ELECTRODES).astype(np.float32)
    f_rates = np.random.uniform(0.5, 10.0, size=NUM_CHANNEL_SETS).astype(np.float32)
    s_amps = np.random.uniform(0.0, 3.0, size=NUM_CHANNEL_SETS).astype(np.float32)

    pkt = pack_neural_stats(
        timestamp_us=ts,
        tick_number=99,
        electrode_spikes=e_spikes,
        firing_rates=f_rates,
        population_rate=5.2,
        burst_active=1.0,
        synchrony_index=1.8,
        stim_amplitudes=s_amps,
        total_neuron_spikes=314.0,
    )
    print(f"  Packet size: {len(pkt)} bytes (expected {STATS_PACKET_SIZE})")

    result = unpack_neural_stats(pkt)
    assert result["timestamp_us"] == ts
    assert result["packet_version"] == STATS_PACKET_VERSION
    assert result["n_electrodes"] == NUM_ELECTRODES
    assert result["tick_number"] == 99
    assert np.allclose(result["electrode_spikes"], e_spikes)
    assert np.allclose(result["firing_rates"], f_rates)
    assert abs(result["population_rate"] - 5.2) < 1e-5
    assert abs(result["burst_active"] - 1.0) < 1e-5
    assert abs(result["synchrony_index"] - 1.8) < 1e-5
    assert np.allclose(result["stim_amplitudes"], s_amps)
    assert abs(result["total_neuron_spikes"] - 314.0) < 1e-5

    print("  Round-trip pack/unpack: OK")
    print("All stats protocol tests passed!")
