"""Criticality analysis: branching ratio and neuronal avalanches.

Tools for assessing whether a cortical culture operates near the critical
point -- a hallmark of healthy in-vitro neural networks (Beggs & Plenz
2003, 2004).  A branching ratio sigma ~ 1.0 indicates criticality;
avalanche size and duration distributions should follow power laws with
exponents -3/2 and -2 respectively.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Spike binning
# ---------------------------------------------------------------------------

def _bin_spikes(
    spike_raster: NDArray,
    dt_ms: float,
    bin_ms: float,
) -> NDArray:
    """Bin a spike raster into coarser time bins.

    Args:
        spike_raster: Boolean or 0/1 array of shape ``(T, N)``.
        dt_ms: Simulation timestep in ms.
        bin_ms: Desired bin width in ms.

    Returns:
        ``(n_bins,)`` array of total spike counts per bin.
    """
    raster = np.asarray(spike_raster, dtype=np.float32)
    if raster.size == 0:
        return np.array([], dtype=np.float64)

    steps_per_bin = max(int(round(bin_ms / dt_ms)), 1)

    T = raster.shape[0]
    n_bins = T // steps_per_bin

    if n_bins == 0:
        return np.array([], dtype=np.float64)

    # Trim any leftover timesteps that don't fill a complete bin
    trimmed = raster[: n_bins * steps_per_bin]

    # Reshape to (n_bins, steps_per_bin, N) and sum over time and neurons
    binned = trimmed.reshape(n_bins, steps_per_bin, -1).sum(axis=(1, 2))
    return binned.astype(np.float64)


# ---------------------------------------------------------------------------
# Branching ratio
# ---------------------------------------------------------------------------

def branching_ratio(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 4.0,
) -> float:
    """Compute branching ratio sigma from a spike raster.

    The branching ratio is defined as the average ratio of activity in
    consecutive time bins:

        sigma = <n_spikes[t+1] / n_spikes[t]>

    computed only for bins where ``n_spikes[t] > 0``.  A value of
    sigma ~ 1.0 indicates the network is operating near criticality.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms (default 0.5).
        bin_ms: Bin width in ms (default 4.0).  The standard choice is
            the average inter-spike interval at the network level, often
            approximated as 2-10 ms for cortical cultures.

    Returns:
        Branching ratio sigma (scalar float).  Returns ``nan`` if there
        are no non-zero ancestor bins.
    """
    binned = _bin_spikes(spike_raster, dt_ms, bin_ms)

    if len(binned) < 2:
        return float("nan")

    ancestors = binned[:-1]
    descendants = binned[1:]

    # Only consider bins where the ancestor has activity
    active_mask = ancestors > 0
    if not np.any(active_mask):
        return float("nan")

    ratios = descendants[active_mask] / ancestors[active_mask]
    sigma = float(np.mean(ratios))
    return sigma


# ---------------------------------------------------------------------------
# Avalanche detection
# ---------------------------------------------------------------------------

def avalanche_size_distribution(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 4.0,
) -> tuple[NDArray, NDArray]:
    """Detect neuronal avalanches and compute size/duration distributions.

    An avalanche is a contiguous sequence of time bins with at least one
    spike, bounded by silent bins (bins with zero spikes).

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms (default 0.5).
        bin_ms: Bin width in ms (default 4.0).

    Returns:
        sizes: 1-D array of avalanche sizes (total spike count per
            avalanche).
        durations: 1-D array of avalanche durations (number of
            consecutive active bins).
    """
    binned = _bin_spikes(spike_raster, dt_ms, bin_ms)

    if len(binned) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    # Identify active bins
    active = binned > 0  # (n_bins,)

    sizes: list[float] = []
    durations: list[int] = []

    in_avalanche = False
    current_size = 0.0
    current_duration = 0

    for i in range(len(binned)):
        if active[i]:
            if not in_avalanche:
                # Start of a new avalanche
                in_avalanche = True
                current_size = 0.0
                current_duration = 0
            current_size += binned[i]
            current_duration += 1
        else:
            if in_avalanche:
                # End of avalanche
                sizes.append(current_size)
                durations.append(current_duration)
                in_avalanche = False

    # Close any avalanche that extends to the end of the recording
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)

    return np.array(sizes, dtype=np.float64), np.array(durations, dtype=np.int64)
