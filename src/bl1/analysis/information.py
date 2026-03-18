"""Information-theoretic metrics for neural activity analysis.

Implements metrics from Varley, Havert, Fosque et al. (2024)
"Information-processing dynamics in in vitro cortical neural circuits"
Network Neuroscience.

These metrics quantify how much information the culture processes
and how it's distributed across neurons.

Methods:
- Active information storage: how much a neuron's future depends on its past
- Mutual information: pairwise shared information between neurons
- Integration: whole-network information beyond individual neurons
- Complexity: balance of integration and segregation (Tononi et al. 1994)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bin_spikes_per_neuron(
    spike_raster: NDArray,
    dt_ms: float,
    bin_ms: float,
) -> NDArray:
    """Bin a spike raster into coarser time bins, preserving per-neuron counts.

    Args:
        spike_raster: Boolean or 0/1 array of shape ``(T, N)``.
        dt_ms: Simulation timestep in ms.
        bin_ms: Desired bin width in ms.

    Returns:
        ``(n_bins, N)`` array of spike counts per bin per neuron.
    """
    raster = np.asarray(spike_raster, dtype=np.float32)
    if raster.size == 0:
        return np.empty((0, raster.shape[1] if raster.ndim == 2 else 0), dtype=np.float64)

    steps_per_bin = max(int(round(bin_ms / dt_ms)), 1)
    T, N = raster.shape
    n_bins = T // steps_per_bin

    if n_bins == 0:
        return np.empty((0, N), dtype=np.float64)

    trimmed = raster[: n_bins * steps_per_bin]
    binned = trimmed.reshape(n_bins, steps_per_bin, N).sum(axis=1)
    return binned.astype(np.float64)


def _discretize(counts: NDArray) -> NDArray:
    """Discretize spike counts into states: 0 (no spikes), 1 (one spike),
    2 (two or more spikes).

    Args:
        counts: Array of spike counts (any shape).

    Returns:
        Integer array with values in {0, 1, 2}.
    """
    return np.minimum(counts, 2).astype(np.int64)


def _entropy(probs: NDArray) -> float:
    """Shannon entropy in bits from a probability vector."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def _entropy_from_samples(samples: NDArray) -> float:
    """Estimate Shannon entropy from discrete integer samples.

    Args:
        samples: 1-D array of discrete integer states.

    Returns:
        Entropy in bits.
    """
    if len(samples) == 0:
        return 0.0
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return _entropy(probs)


def _joint_entropy_from_samples(samples_a: NDArray, samples_b: NDArray) -> float:
    """Estimate joint Shannon entropy from two discrete sample vectors.

    Combines the two variables into a single joint state and computes
    entropy of the joint distribution.

    Args:
        samples_a: 1-D integer array.
        samples_b: 1-D integer array of the same length.

    Returns:
        Joint entropy in bits.
    """
    if len(samples_a) == 0:
        return 0.0
    # Combine into a single state using Cantor-like pairing
    max_b = int(samples_b.max()) + 1 if len(samples_b) > 0 else 1
    joint = samples_a * max_b + samples_b
    return _entropy_from_samples(joint)


# ---------------------------------------------------------------------------
# Active information storage
# ---------------------------------------------------------------------------


def active_information_storage(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    history_length: int = 5,
) -> NDArray:
    """Active information storage per neuron.

    AIS(X) = I(X_past; X_future) -- how much of a neuron's future
    activity is predicted by its own past.

    High AIS = neuron has strong intrinsic dynamics (e.g., bursting).
    Low AIS = neuron is driven by external inputs.

    Uses discrete binning: spike counts are mapped to states
    {0, 1, 2+} and past states are encoded as integer tuples.

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        bin_ms: Time bin width in ms.
        history_length: Number of past bins to use for the "past" variable.

    Returns:
        ``(N,)`` array of AIS values per neuron (in bits).
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N = binned.shape

    discrete = _discretize(binned)  # (n_bins, N), values in {0, 1, 2}

    ais = np.zeros(N, dtype=np.float64)

    if n_bins <= history_length:
        return ais

    n_samples = n_bins - history_length

    for n in range(N):
        signal = discrete[:, n]
        future = signal[history_length:]  # (n_samples,)

        # Encode past as integer: (s[t-k], ..., s[t-1]) -> base-3 integer
        past_windows = np.lib.stride_tricks.sliding_window_view(signal[:n_bins], history_length)[
            :n_samples
        ]  # (n_samples, history_length)

        powers = (3 ** np.arange(history_length, dtype=np.int64))[::-1]
        past_states = (past_windows * powers).sum(axis=1)

        # AIS = I(past; future) = H(future) + H(past) - H(future, past)
        h_future = _entropy_from_samples(future)
        h_past = _entropy_from_samples(past_states)
        h_joint = _joint_entropy_from_samples(past_states, future)

        ais[n] = max(h_future + h_past - h_joint, 0.0)

    return ais


# ---------------------------------------------------------------------------
# Mutual information matrix
# ---------------------------------------------------------------------------


def mutual_information_matrix(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    subset: int | None = None,
) -> NDArray:
    """Pairwise mutual information between neurons.

    MI(X; Y) = H(X) + H(Y) - H(X, Y)

    Symmetric measure of shared information (unlike TE which is directed).

    Uses discrete binning: spike counts are mapped to states {0, 1, 2+}.

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        bin_ms: Time bin width in ms.
        subset: If not ``None``, only compute for first ``subset`` neurons.

    Returns:
        ``(N, N)`` or ``(subset, subset)`` MI matrix (in bits).
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N_total = binned.shape

    N = min(subset, N_total) if subset is not None else N_total
    discrete = _discretize(binned[:, :N])  # (n_bins, N)

    mi = np.zeros((N, N), dtype=np.float64)

    if n_bins < 2 or N == 0:
        return mi

    # Precompute marginal entropies
    h_marginal = np.zeros(N, dtype=np.float64)
    for n in range(N):
        h_marginal[n] = _entropy_from_samples(discrete[:, n])

    for i in range(N):
        mi[i, i] = h_marginal[i]  # MI(X;X) = H(X)
        for j in range(i + 1, N):
            h_joint = _joint_entropy_from_samples(discrete[:, i], discrete[:, j])
            mi_val = max(h_marginal[i] + h_marginal[j] - h_joint, 0.0)
            mi[i, j] = mi_val
            mi[j, i] = mi_val

    return mi


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def integration(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    n_samples: int = 100,
) -> float:
    """Network integration (Tononi et al. 1994).

    I(X) = sum(H(Xi)) - H(X) -- how much the whole network knows
    that individual neurons don't.

    For large N, the joint entropy H(X) is intractable to compute
    directly.  We approximate by computing integration on random
    subsets of neurons (up to 10 at a time) and averaging.

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        bin_ms: Time bin width in ms.
        n_samples: Number of random subsets to sample.

    Returns:
        Estimated integration value in bits.  Non-negative.
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N = binned.shape
    discrete = _discretize(binned)

    if n_bins < 2 or N < 2:
        return 0.0

    rng = np.random.default_rng(42)

    # Subset size for tractable joint entropy
    max_subset_size = min(N, 10)

    integration_values: list[float] = []

    for _ in range(n_samples):
        # Random subset of neurons
        k = rng.integers(2, max_subset_size + 1)
        indices = rng.choice(N, size=k, replace=False)

        # Sum of marginal entropies
        h_sum = 0.0
        for idx in indices:
            h_sum += _entropy_from_samples(discrete[:, idx])

        # Joint entropy via combined state
        # Encode joint state as a single integer using mixed-radix (base 3)
        joint_state = np.zeros(n_bins, dtype=np.int64)
        multiplier = 1
        for idx in indices:
            joint_state = joint_state + discrete[:, idx] * multiplier
            multiplier *= 3

        h_joint = _entropy_from_samples(joint_state)

        # Integration for this subset
        integ = max(h_sum - h_joint, 0.0)
        integration_values.append(integ)

    return float(np.mean(integration_values))


# ---------------------------------------------------------------------------
# Complexity (Tononi, Sporns & Edelman 1994)
# ---------------------------------------------------------------------------


def complexity(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    bin_ms: float = 10.0,
    n_samples: int = 100,
) -> float:
    """Neural complexity (Tononi, Sporns & Edelman 1994).

    C(X) = sum_{k=1}^{N} [k/N * H(X) - <H(X_k)>]

    where <H(X_k)> is the average entropy of subsets of size *k*.
    High complexity = both integration and segregation are high.

    Estimated via random subset sampling since exhaustive enumeration
    is exponential in N.

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        bin_ms: Time bin width in ms.
        n_samples: Number of random subsets to sample per subset size.

    Returns:
        Estimated neural complexity in bits.
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N = binned.shape
    discrete = _discretize(binned)

    if n_bins < 2 or N < 2:
        return 0.0

    rng = np.random.default_rng(42)

    # Estimate H(X) -- joint entropy of all neurons (use subset if N is large)
    max_joint = min(N, 10)
    joint_state = np.zeros(n_bins, dtype=np.int64)
    multiplier = 1
    for n in range(max_joint):
        joint_state = joint_state + discrete[:, n] * multiplier
        multiplier *= 3
    h_total = _entropy_from_samples(joint_state)

    # For each subset size k from 1 to min(N, max_joint), estimate <H(X_k)>
    # by sampling random subsets and computing their joint entropy.
    complexity_sum = 0.0
    max_k = min(N, max_joint)

    for k in range(1, max_k + 1):
        h_k_samples: list[float] = []
        n_iter = min(n_samples, max(1, math.comb(N, k)))

        for _ in range(n_iter):
            if k < N:
                indices = rng.choice(N, size=k, replace=False)
            else:
                indices = np.arange(N)

            # Joint entropy of the subset
            sub_state = np.zeros(n_bins, dtype=np.int64)
            mult = 1
            for idx in indices:
                sub_state = sub_state + discrete[:, idx] * mult
                mult *= 3
            h_k_samples.append(_entropy_from_samples(sub_state))

        avg_h_k = float(np.mean(h_k_samples))
        # Contribution: k/N * H(X) - <H(X_k)>
        contribution = (k / N) * h_total - avg_h_k
        complexity_sum += contribution

    return max(float(complexity_sum), 0.0)
