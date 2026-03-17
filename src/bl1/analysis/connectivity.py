"""Functional connectivity inference from spike trains.

Since cortical culture datasets only contain extracellular recordings
(spike times, not structural connectivity), validation against real data
requires inferring functional connectivity from spike correlations.

Methods:
- Cross-correlation: pairwise spike train correlation at various lags
- Transfer entropy: directed information flow between neuron pairs
- Effective connectivity: thresholded TE graph
- Small-world coefficient: clustering + path length analysis
- Rich-club coefficient: hub connectivity analysis

Reference: Garofalo et al. (2009) "Evaluation of the Performance of
Information Theory-Based Methods and Cross-Correlation to Estimate
the Functional Connectivity in Cortical Networks" PLoS ONE
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Spike binning (neuron-resolved)
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
        return np.empty((0, raster.shape[1] if raster.ndim == 2 else 0),
                        dtype=np.float64)

    steps_per_bin = max(int(round(bin_ms / dt_ms)), 1)
    T, N = raster.shape
    n_bins = T // steps_per_bin

    if n_bins == 0:
        return np.empty((0, N), dtype=np.float64)

    trimmed = raster[: n_bins * steps_per_bin]
    # (n_bins, steps_per_bin, N) -> sum over time axis
    binned = trimmed.reshape(n_bins, steps_per_bin, N).sum(axis=1)
    return binned.astype(np.float64)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def cross_correlation_matrix(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    max_lag_ms: float = 50.0,
    bin_ms: float = 5.0,
) -> NDArray:
    """Compute pairwise cross-correlation matrix between neurons.

    Bins spikes into time bins, computes normalized cross-correlation
    at various lags using FFT, and returns the peak correlation for each
    pair.

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        max_lag_ms: Maximum lag to consider in ms.
        bin_ms: Time bin width for correlation in ms.

    Returns:
        ``(N, N)`` correlation matrix where entry ``[i, j]`` is the peak
        cross-correlation between neurons *i* and *j*.
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N = binned.shape

    if n_bins < 2 or N == 0:
        return np.zeros((N, N), dtype=np.float64)

    max_lag_bins = max(int(round(max_lag_ms / bin_ms)), 1)

    # Zero-mean the signals
    means = binned.mean(axis=0, keepdims=True)
    stds = binned.std(axis=0, keepdims=True)
    stds[stds < 1e-12] = 1.0  # avoid division by zero for silent neurons
    normed = (binned - means) / stds

    # FFT-based cross-correlation: pad to avoid circular effects
    fft_len = 1
    while fft_len < 2 * n_bins:
        fft_len *= 2

    spectra = np.fft.rfft(normed, n=fft_len, axis=0)  # (fft_len//2+1, N)

    cc_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        # Cross-spectra for neuron i against all others
        cross_spec = spectra[:, i : i + 1].conj() * spectra  # (freq, N)
        cc_full = np.fft.irfft(cross_spec, n=fft_len, axis=0) / n_bins

        # Extract the relevant lags: positive lags [0..max_lag_bins]
        # and negative lags (wrapped to end of array) [-max_lag_bins..-1]
        pos_lags = cc_full[:max_lag_bins + 1, :]  # lags 0..max_lag
        neg_lags = cc_full[-max_lag_bins:, :]       # lags -max_lag..-1

        # Combine and take peak absolute correlation
        all_lags = np.concatenate([neg_lags, pos_lags], axis=0)  # (2*max_lag+1, N)
        cc_matrix[i, :] = np.max(np.abs(all_lags), axis=0)

    return cc_matrix


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------

def _binary_history_to_int(history: NDArray) -> NDArray:
    """Convert binary history vectors to integer state indices.

    Args:
        history: ``(n_samples, history_length)`` binary array.

    Returns:
        ``(n_samples,)`` integer array with values in ``[0, 2**history_length)``.
    """
    h = history.shape[1]
    powers = (2 ** np.arange(h, dtype=np.int64))[::-1]
    return (history * powers).sum(axis=1).astype(np.int64)


def _entropy(probs: NDArray) -> float:
    """Shannon entropy in bits from a probability vector."""
    p = probs[probs > 0]
    return -float(np.sum(p * np.log2(p)))


def transfer_entropy(
    spike_raster: NDArray,
    dt_ms: float = 0.5,
    history_bins: int = 5,
    bin_ms: float = 10.0,
    subset: int | None = None,
) -> NDArray:
    """Compute pairwise transfer entropy between neurons.

    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Measures directed information flow: TE(X->Y) > 0 means X's past
    helps predict Y's future beyond Y's own history.

    Uses histogram-based probability estimation with binary binning
    (spike / no-spike per bin).

    Args:
        spike_raster: ``(T, N)`` boolean spike array.
        dt_ms: Simulation timestep in ms.
        history_bins: Number of past bins for conditioning.
        bin_ms: Time bin width in ms.
        subset: If not ``None``, only compute for first ``subset`` neurons
            (for speed on large networks).

    Returns:
        ``(N, N)`` or ``(subset, subset)`` transfer entropy matrix.
        ``TE[i, j]`` = transfer entropy from neuron *i* to neuron *j*.
    """
    binned = _bin_spikes_per_neuron(spike_raster, dt_ms, bin_ms)
    n_bins, N_total = binned.shape

    # Binarize: 1 if any spike in bin, 0 otherwise
    binary = (binned > 0).astype(np.int64)

    N = min(subset, N_total) if subset is not None else N_total
    binary = binary[:, :N]

    if n_bins <= history_bins + 1 or N == 0:
        return np.zeros((N, N), dtype=np.float64)

    n_samples = n_bins - history_bins

    # Pre-compute history and future arrays for each neuron
    # future[n] = binary[history_bins:, n]  (the "next" value)
    # past[n] = sliding window of length history_bins ending just before future
    futures = binary[history_bins:, :]  # (n_samples, N)

    # Build history state indices for each neuron
    history_states = np.zeros((n_samples, N), dtype=np.int64)
    for n in range(N):
        hist_windows = np.lib.stride_tricks.sliding_window_view(
            binary[:n_bins, n], history_bins
        )  # (n_bins - history_bins + 1, history_bins)
        # We want history ending at t-1 for future at t
        # sliding_window_view gives windows starting at each index
        # window[t] = binary[t:t+history_bins], we want t=0..n_samples-1
        history_states[:, n] = _binary_history_to_int(hist_windows[:n_samples])

    n_history_states = 2 ** history_bins

    te_matrix = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        y_future = futures[:, j]
        y_past = history_states[:, j]

        # Precompute H(Y_future | Y_past) -- same for all source neurons i
        # H(Y_future | Y_past) = H(Y_future, Y_past) - H(Y_past)

        # Joint (y_past, y_future) distribution
        joint_yp_yf = np.zeros((n_history_states, 2), dtype=np.float64)
        for s in range(n_history_states):
            mask = y_past == s
            count = mask.sum()
            if count == 0:
                continue
            yf_given_s = y_future[mask]
            joint_yp_yf[s, 0] = np.sum(yf_given_s == 0)
            joint_yp_yf[s, 1] = np.sum(yf_given_s == 1)

        # Marginal of y_past
        p_yp = joint_yp_yf.sum(axis=1) / n_samples
        # Joint probability
        p_yp_yf = joint_yp_yf / n_samples

        h_yp = _entropy(p_yp)
        h_yp_yf = _entropy(p_yp_yf.ravel())
        h_yfuture_given_ypast = h_yp_yf - h_yp

        for i in range(N):
            if i == j:
                te_matrix[i, j] = 0.0
                continue

            x_past = history_states[:, i]

            # Joint (x_past, y_past, y_future) distribution
            # State = x_past * n_history_states + y_past (combined source+target history)
            combined_past = x_past * n_history_states + y_past
            n_combined = n_history_states * n_history_states

            joint_xp_yp_yf = np.zeros((n_combined, 2), dtype=np.float64)
            for s in range(n_combined):
                mask = combined_past == s
                count = mask.sum()
                if count == 0:
                    continue
                yf_given_s = y_future[mask]
                joint_xp_yp_yf[s, 0] = np.sum(yf_given_s == 0)
                joint_xp_yp_yf[s, 1] = np.sum(yf_given_s == 1)

            # Marginal of (x_past, y_past)
            p_xp_yp = joint_xp_yp_yf.sum(axis=1) / n_samples
            # Joint probability
            p_xp_yp_yf = joint_xp_yp_yf / n_samples

            h_xp_yp = _entropy(p_xp_yp)
            h_xp_yp_yf = _entropy(p_xp_yp_yf.ravel())
            h_yfuture_given_xp_yp = h_xp_yp_yf - h_xp_yp

            # TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
            te = h_yfuture_given_ypast - h_yfuture_given_xp_yp
            te_matrix[i, j] = max(te, 0.0)  # clamp to non-negative

    return te_matrix


# ---------------------------------------------------------------------------
# Effective connectivity graph
# ---------------------------------------------------------------------------

def effective_connectivity_graph(
    te_matrix: NDArray,
    threshold_percentile: float = 95.0,
) -> NDArray:
    """Threshold transfer entropy matrix into a binary directed graph.

    Args:
        te_matrix: ``(N, N)`` transfer entropy matrix.
        threshold_percentile: Percentile threshold for significant
            connections (applied to non-zero entries).

    Returns:
        ``(N, N)`` binary adjacency matrix of significant directed
        connections.
    """
    positive = te_matrix[te_matrix > 0]
    if len(positive) == 0:
        return np.zeros_like(te_matrix, dtype=np.float64)
    threshold = np.percentile(positive, threshold_percentile)
    return (te_matrix > threshold).astype(np.float64)


# ---------------------------------------------------------------------------
# Small-world coefficient
# ---------------------------------------------------------------------------

def _clustering_coefficient(adjacency: NDArray) -> float:
    """Compute mean clustering coefficient for a binary graph.

    For each node, the clustering coefficient is the fraction of
    possible triangles through that node that actually exist.
    Treats the graph as undirected by symmetrising the adjacency matrix.
    """
    # Symmetrise for undirected analysis
    A = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    if N < 3:
        return 0.0

    coeffs = []
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        # Count edges among neighbors
        subgraph = A[np.ix_(neighbors, neighbors)]
        actual_edges = subgraph.sum() / 2.0  # undirected
        possible_edges = k * (k - 1) / 2.0
        coeffs.append(actual_edges / possible_edges)

    return float(np.mean(coeffs))


def _mean_shortest_path(adjacency: NDArray) -> float:
    """Compute mean shortest path length via BFS on the largest component.

    Treats the graph as undirected.  Returns ``inf`` if the graph is
    disconnected (no connected component with > 1 node).
    """
    A = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    if N < 2:
        return float("inf")

    # Build adjacency list
    adj_list: list[list[int]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                adj_list[i].append(j)
                adj_list[j].append(i)

    # Find connected components via BFS
    visited = np.zeros(N, dtype=bool)
    components: list[list[int]] = []

    for start in range(N):
        if visited[start]:
            continue
        component: list[int] = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in adj_list[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        components.append(component)

    # Use the largest component
    largest = max(components, key=len)
    if len(largest) < 2:
        return float("inf")

    # BFS from each node in the largest component
    node_set = set(largest)
    total_dist = 0.0
    n_pairs = 0

    for source in largest:
        dist = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for nbr in adj_list[node]:
                if nbr in node_set and nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        for target in largest:
            if target != source and target in dist:
                total_dist += dist[target]
                n_pairs += 1

    if n_pairs == 0:
        return float("inf")

    return total_dist / n_pairs


def small_world_coefficient(
    adjacency: NDArray,
) -> dict:
    """Compute small-world metrics from a binary adjacency matrix.

    Computes clustering coefficient and mean path length for the input
    graph and for an Erdos-Renyi random graph with the same density,
    then derives the small-world sigma.

    Returns:
        Dict with ``'clustering_coefficient'``, ``'mean_path_length'``,
        ``'small_world_sigma'`` (ratio to random graph equivalents).
    """
    A = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    cc = _clustering_coefficient(adjacency)
    mpl = _mean_shortest_path(adjacency)

    # Erdos-Renyi reference with same edge density
    n_edges = A.sum() / 2.0
    max_edges = N * (N - 1) / 2.0
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Expected clustering for ER: p
    cc_random = density

    # Expected mean path length for ER: ln(N) / ln(N*p) for large connected ER
    if density > 0 and N > 1:
        avg_degree = density * (N - 1)
        if avg_degree > 1:
            mpl_random = np.log(N) / np.log(avg_degree) if avg_degree > 1 else float("inf")
        else:
            mpl_random = float("inf")
    else:
        mpl_random = float("inf")

    # Small-world sigma = (C/C_rand) / (L/L_rand)
    if cc_random > 0 and mpl_random > 0 and np.isfinite(mpl_random) and mpl > 0:
        gamma = cc / cc_random
        lam = mpl / mpl_random
        sigma = gamma / lam if lam > 0 else 0.0
    else:
        sigma = 0.0

    return {
        "clustering_coefficient": cc,
        "mean_path_length": mpl,
        "small_world_sigma": sigma,
    }


# ---------------------------------------------------------------------------
# Rich-club coefficient
# ---------------------------------------------------------------------------

def rich_club_coefficient(
    adjacency: NDArray,
    degree_threshold: int | None = None,
) -> dict:
    """Compute rich-club coefficient.

    Rich-club organization means highly connected neurons preferentially
    connect to each other.  This is a key property of mature cortical
    cultures (Nigam et al. 2016).

    For each degree threshold *k*, the rich-club coefficient is::

        phi(k) = 2 * E_k / (N_k * (N_k - 1))

    where *E_k* is the number of edges among nodes with degree >= *k*
    and *N_k* is the count of such nodes.  The normalized coefficient
    divides by the same metric for a random graph with the same degree
    sequence (approximated here by density).

    Args:
        adjacency: ``(N, N)`` binary adjacency matrix.
        degree_threshold: If given, only compute for this specific
            threshold.  Otherwise, compute for all meaningful thresholds.

    Returns:
        Dict with ``'rich_club_coeff'`` (dict mapping threshold *k* to
        phi(k)), ``'degree_distribution'`` (array of node degrees).
    """
    A = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(A, 0)
    N = A.shape[0]

    degrees = A.sum(axis=1).astype(int)

    if degree_threshold is not None:
        thresholds = [degree_threshold]
    else:
        # Use unique non-zero degree values
        unique_deg = np.unique(degrees[degrees > 0])
        thresholds = sorted(unique_deg.tolist())

    rich_club: dict[int, float] = {}

    # Global density for normalization
    total_edges = A.sum() / 2.0
    max_edges = N * (N - 1) / 2.0
    global_density = total_edges / max_edges if max_edges > 0 else 0.0

    for k in thresholds:
        rich_nodes = np.where(degrees >= k)[0]
        n_k = len(rich_nodes)
        if n_k < 2:
            rich_club[k] = 0.0
            continue

        subgraph = A[np.ix_(rich_nodes, rich_nodes)]
        e_k = subgraph.sum() / 2.0
        max_rich_edges = n_k * (n_k - 1) / 2.0
        phi_k = e_k / max_rich_edges if max_rich_edges > 0 else 0.0

        # Normalize by random expectation (density)
        if global_density > 0:
            phi_k_norm = phi_k / global_density
        else:
            phi_k_norm = 0.0

        rich_club[k] = phi_k_norm

    return {
        "rich_club_coeff": rich_club,
        "degree_distribution": degrees,
    }
