"""Network burst detection and statistics.

Detects synchronous population bursts by threshold-crossing on the
smoothed population firing rate, then computes summary statistics
including inter-burst intervals, burst durations, and recruitment
fractions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


def detect_bursts(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    threshold_std: float = 2.0,
    min_duration_ms: float = 50.0,
) -> list[tuple[float, float, int, float]]:
    """Detect network bursts using threshold crossing on population rate.

    A burst begins when the instantaneous population spike count exceeds
    ``mean + threshold_std * std`` and ends when it drops back below the
    mean.  Only bursts lasting at least ``min_duration_ms`` are retained.

    Args:
        spike_raster: ``(T, N)`` boolean spike raster.
        dt_ms: Simulation timestep in ms (default 0.5).
        threshold_std: Number of standard deviations above the mean for
            burst onset detection (default 2.0).
        min_duration_ms: Minimum burst duration in ms to accept
            (default 50.0).

    Returns:
        List of ``(start_ms, end_ms, n_spikes, fraction_recruited)``
        tuples.  ``fraction_recruited`` is the fraction of neurons that
        fired at least once during the burst.
    """
    raster = np.asarray(spike_raster, dtype=np.float32)
    T, N = raster.shape

    if T == 0 or N == 0:
        return []

    # Population spike count per timestep
    pop_count = raster.sum(axis=1)  # (T,)

    mean_count = np.mean(pop_count)
    std_count = np.std(pop_count)

    # Avoid degenerate cases where std is zero
    if std_count < 1e-12:
        return []

    onset_threshold = mean_count + threshold_std * std_count
    offset_threshold = mean_count

    min_duration_steps = max(int(round(min_duration_ms / dt_ms)), 1)

    bursts: list[tuple[float, float, int, float]] = []
    in_burst = False
    burst_start: int = 0

    for t in range(T):
        if not in_burst:
            if pop_count[t] > onset_threshold:
                in_burst = True
                burst_start = t
        else:
            if pop_count[t] <= offset_threshold:
                # End of burst
                burst_end = t
                duration_steps = burst_end - burst_start

                if duration_steps >= min_duration_steps:
                    burst_raster = raster[burst_start:burst_end]
                    n_spikes = int(burst_raster.sum())
                    # Fraction of neurons that fired at least once
                    neurons_active = np.any(burst_raster > 0, axis=0).sum()
                    fraction_recruited = float(neurons_active) / N

                    start_ms = burst_start * dt_ms
                    end_ms = burst_end * dt_ms
                    bursts.append((start_ms, end_ms, n_spikes, fraction_recruited))

                in_burst = False

    # Handle burst that extends to the end of recording
    if in_burst:
        burst_end = T
        duration_steps = burst_end - burst_start
        if duration_steps >= min_duration_steps:
            burst_raster = raster[burst_start:burst_end]
            n_spikes = int(burst_raster.sum())
            neurons_active = np.any(burst_raster > 0, axis=0).sum()
            fraction_recruited = float(neurons_active) / N
            start_ms = burst_start * dt_ms
            end_ms = burst_end * dt_ms
            bursts.append((start_ms, end_ms, n_spikes, fraction_recruited))

    return bursts


# ---------------------------------------------------------------------------
# Burst statistics
# ---------------------------------------------------------------------------


def burst_statistics(
    bursts: list[tuple[float, float, int, float]],
) -> dict[str, float]:
    """Compute summary statistics from detected bursts.

    Args:
        bursts: Output of :func:`detect_bursts` -- list of
            ``(start_ms, end_ms, n_spikes, fraction_recruited)`` tuples.

    Returns:
        Dict with the following keys:

        - ``ibi_mean``: Mean inter-burst interval in ms.
        - ``ibi_cv``: Coefficient of variation of inter-burst intervals.
        - ``duration_mean``: Mean burst duration in ms.
        - ``recruitment_mean``: Mean fraction of network recruited per burst.
        - ``burst_rate``: Bursts per second (estimated from the time span
          between the first and last burst).
    """
    result: dict[str, float] = {
        "ibi_mean": float("nan"),
        "ibi_cv": float("nan"),
        "duration_mean": float("nan"),
        "recruitment_mean": float("nan"),
        "burst_rate": 0.0,
    }

    if not bursts:
        return result

    starts = np.array([b[0] for b in bursts])
    ends = np.array([b[1] for b in bursts])
    durations = ends - starts
    recruitments = np.array([b[3] for b in bursts])

    result["duration_mean"] = float(np.mean(durations))
    result["recruitment_mean"] = float(np.mean(recruitments))

    # Inter-burst intervals (onset-to-onset)
    if len(starts) >= 2:
        ibis = np.diff(starts)
        result["ibi_mean"] = float(np.mean(ibis))
        ibi_std = float(np.std(ibis))
        result["ibi_cv"] = ibi_std / result["ibi_mean"] if result["ibi_mean"] > 0 else float("nan")

        # Burst rate: number of bursts / total time span (in seconds)
        total_span_s = (starts[-1] - starts[0]) / 1000.0
        if total_span_s > 0:
            result["burst_rate"] = (len(bursts) - 1) / total_span_s
    elif len(bursts) == 1:
        # Single burst -- can't compute IBI
        result["ibi_mean"] = float("nan")
        result["ibi_cv"] = float("nan")
        result["burst_rate"] = 0.0

    return result
