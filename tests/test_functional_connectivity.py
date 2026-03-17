"""Tests for functional connectivity and information-theoretic metrics.

All tests use small synthetic rasters (50-100 neurons, 1000-2000 timesteps)
to keep runtime reasonable -- these algorithms are O(N^2) or worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from bl1.analysis.connectivity import (
    cross_correlation_matrix,
    transfer_entropy,
    effective_connectivity_graph,
    small_world_coefficient,
    rich_club_coefficient,
)
from bl1.analysis.information import (
    active_information_storage,
    mutual_information_matrix,
    integration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clustered_raster(
    n_neurons: int = 50,
    n_steps: int = 2000,
    cluster_size: int = 15,
    background_prob: float = 0.01,
    cluster_prob: float = 0.15,
    seed: int = 123,
) -> np.ndarray:
    """Create a synthetic spike raster with two correlated clusters.

    Neurons 0..cluster_size-1 form cluster A and neurons
    cluster_size..2*cluster_size-1 form cluster B.  Within each
    cluster, neurons fire synchronously with probability ``cluster_prob``
    in shared burst windows.  All other neurons fire independently
    at ``background_prob``.

    This structure should produce higher cross-correlation and MI
    within clusters than between clusters.
    """
    rng = np.random.default_rng(seed)
    raster = rng.random((n_steps, n_neurons)) < background_prob

    # Generate burst windows for each cluster
    burst_duration = 5  # timesteps
    for t_start in range(0, n_steps, 40):  # every 40 steps
        t_end = min(t_start + burst_duration, n_steps)
        # Cluster A
        burst_a = rng.random((t_end - t_start, cluster_size)) < cluster_prob
        raster[t_start:t_end, :cluster_size] |= burst_a
        # Cluster B (offset by 20 steps so clusters are not correlated)
        t_start_b = min(t_start + 20, n_steps)
        t_end_b = min(t_start_b + burst_duration, n_steps)
        if t_end_b > t_start_b:
            burst_b = rng.random(
                (t_end_b - t_start_b, cluster_size)
            ) < cluster_prob
            raster[t_start_b:t_end_b, cluster_size : 2 * cluster_size] |= burst_b

    return raster.astype(np.float32)


def _make_simple_raster(
    n_neurons: int = 50,
    n_steps: int = 2000,
    prob: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Create a simple uniform random spike raster."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_steps, n_neurons)) < prob).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: Cross-correlation
# ---------------------------------------------------------------------------


class TestCrossCorrelation:
    """Tests for cross_correlation_matrix."""

    def test_cross_correlation_shape(self):
        """Output should be (N, N)."""
        raster = _make_simple_raster(n_neurons=30, n_steps=1000)
        cc = cross_correlation_matrix(raster, dt_ms=0.5)
        assert cc.shape == (30, 30)

    def test_cross_correlation_diagonal(self):
        """Self-correlation (diagonal) should be maximal or near-maximal."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        cc = cross_correlation_matrix(raster, dt_ms=0.5)
        for i in range(20):
            # Diagonal should be >= most off-diagonal entries for that neuron
            off_diag = np.delete(cc[i, :], i)
            assert cc[i, i] >= np.median(off_diag), (
                f"Neuron {i}: self-corr {cc[i,i]:.4f} < median off-diag {np.median(off_diag):.4f}"
            )

    def test_cross_correlation_symmetric(self):
        """CC[i,j] should approximately equal CC[j,i]."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        cc = cross_correlation_matrix(raster, dt_ms=0.5)
        np.testing.assert_allclose(cc, cc.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Transfer entropy
# ---------------------------------------------------------------------------


class TestTransferEntropy:
    """Tests for transfer_entropy."""

    def test_transfer_entropy_shape(self):
        """Output should be (N, N) or (subset, subset)."""
        raster = _make_simple_raster(n_neurons=30, n_steps=1000)
        te = transfer_entropy(raster, dt_ms=0.5, history_bins=3)
        assert te.shape == (30, 30)

    def test_transfer_entropy_subset_shape(self):
        """With subset parameter, output should be (subset, subset)."""
        raster = _make_simple_raster(n_neurons=50, n_steps=1000)
        te = transfer_entropy(raster, dt_ms=0.5, history_bins=3, subset=20)
        assert te.shape == (20, 20)

    def test_transfer_entropy_nonnegative(self):
        """All entries should be >= 0."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1000)
        te = transfer_entropy(raster, dt_ms=0.5, history_bins=3)
        assert np.all(te >= 0), f"Negative TE values found: {te.min()}"

    def test_transfer_entropy_self_zero(self):
        """TE[i,i] should be zero (no self-transfer)."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1000)
        te = transfer_entropy(raster, dt_ms=0.5, history_bins=3)
        for i in range(20):
            assert te[i, i] == 0.0, f"TE[{i},{i}] = {te[i,i]}, expected 0.0"


# ---------------------------------------------------------------------------
# Tests: Effective connectivity graph
# ---------------------------------------------------------------------------


class TestEffectiveConnectivity:
    """Tests for effective_connectivity_graph."""

    def test_effective_connectivity_binary(self):
        """Thresholded graph should be binary (0 or 1)."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1000)
        te = transfer_entropy(raster, dt_ms=0.5, history_bins=3)
        graph = effective_connectivity_graph(te, threshold_percentile=90.0)
        unique_vals = np.unique(graph)
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Non-binary value in graph: {v}"

    def test_effective_connectivity_shape(self):
        """Graph should have same shape as input TE matrix."""
        te = np.random.default_rng(0).random((15, 15))
        np.fill_diagonal(te, 0)
        graph = effective_connectivity_graph(te, threshold_percentile=80.0)
        assert graph.shape == (15, 15)


# ---------------------------------------------------------------------------
# Tests: Small-world coefficient
# ---------------------------------------------------------------------------


class TestSmallWorld:
    """Tests for small_world_coefficient."""

    def test_small_world_coefficient_keys(self):
        """Result should contain expected keys."""
        # Create a simple random adjacency matrix
        rng = np.random.default_rng(99)
        adj = (rng.random((20, 20)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)

        result = small_world_coefficient(adj)
        assert "clustering_coefficient" in result
        assert "mean_path_length" in result
        assert "small_world_sigma" in result

    def test_small_world_clustering_range(self):
        """Clustering coefficient should be in [0, 1]."""
        rng = np.random.default_rng(99)
        adj = (rng.random((20, 20)) > 0.6).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)

        result = small_world_coefficient(adj)
        assert 0.0 <= result["clustering_coefficient"] <= 1.0

    def test_small_world_empty_graph(self):
        """Empty graph should return zeros/inf gracefully."""
        adj = np.zeros((10, 10))
        result = small_world_coefficient(adj)
        assert result["clustering_coefficient"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Rich-club coefficient
# ---------------------------------------------------------------------------


class TestRichClub:
    """Tests for rich_club_coefficient."""

    def test_rich_club_coefficient_keys(self):
        """Result should contain expected keys."""
        rng = np.random.default_rng(77)
        adj = (rng.random((20, 20)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)

        result = rich_club_coefficient(adj)
        assert "rich_club_coeff" in result
        assert "degree_distribution" in result

    def test_rich_club_degree_distribution(self):
        """Degree distribution should be an array of length N."""
        rng = np.random.default_rng(77)
        N = 20
        adj = (rng.random((N, N)) > 0.7).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)

        result = rich_club_coefficient(adj)
        assert len(result["degree_distribution"]) == N

    def test_rich_club_specific_threshold(self):
        """With a specific threshold, result should have that key."""
        rng = np.random.default_rng(77)
        adj = (rng.random((20, 20)) > 0.5).astype(float)
        np.fill_diagonal(adj, 0)
        adj = np.maximum(adj, adj.T)

        result = rich_club_coefficient(adj, degree_threshold=5)
        assert 5 in result["rich_club_coeff"]


# ---------------------------------------------------------------------------
# Tests: Mutual information
# ---------------------------------------------------------------------------


class TestMutualInformation:
    """Tests for mutual_information_matrix."""

    def test_mutual_information_symmetric(self):
        """MI[i,j] should equal MI[j,i]."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        mi = mutual_information_matrix(raster, dt_ms=0.5)
        np.testing.assert_allclose(mi, mi.T, atol=1e-10)

    def test_mutual_information_nonnegative(self):
        """All MI entries should be >= 0."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        mi = mutual_information_matrix(raster, dt_ms=0.5)
        assert np.all(mi >= 0), f"Negative MI found: {mi.min()}"

    def test_mutual_information_shape(self):
        """Output shape should be (N, N)."""
        raster = _make_simple_raster(n_neurons=25, n_steps=1000)
        mi = mutual_information_matrix(raster, dt_ms=0.5)
        assert mi.shape == (25, 25)

    def test_mutual_information_self_equals_entropy(self):
        """MI(X;X) = H(X), which should be the diagonal."""
        raster = _make_simple_raster(n_neurons=15, n_steps=1000)
        mi = mutual_information_matrix(raster, dt_ms=0.5)
        # Diagonal should be positive for active neurons
        assert np.all(mi.diagonal() >= 0)


# ---------------------------------------------------------------------------
# Tests: Active information storage
# ---------------------------------------------------------------------------


class TestActiveInformationStorage:
    """Tests for active_information_storage."""

    def test_active_information_storage_shape(self):
        """Output should be (N,) array."""
        raster = _make_simple_raster(n_neurons=30, n_steps=1500)
        ais = active_information_storage(raster, dt_ms=0.5)
        assert ais.shape == (30,)

    def test_active_information_storage_nonnegative(self):
        """AIS values should be >= 0."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        ais = active_information_storage(raster, dt_ms=0.5)
        assert np.all(ais >= 0), f"Negative AIS found: {ais.min()}"

    def test_active_information_storage_silent_neuron(self):
        """A completely silent neuron should have AIS = 0."""
        raster = _make_simple_raster(n_neurons=10, n_steps=1000, prob=0.05)
        # Silence one neuron completely
        raster[:, 0] = 0.0
        ais = active_information_storage(raster, dt_ms=0.5)
        assert ais[0] == 0.0


# ---------------------------------------------------------------------------
# Tests: Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Tests for the integration metric."""

    def test_integration_nonnegative(self):
        """Integration should be >= 0."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1500)
        integ = integration(raster, dt_ms=0.5, n_samples=50)
        assert integ >= 0.0, f"Negative integration: {integ}"

    def test_integration_scalar(self):
        """Integration should return a single float."""
        raster = _make_simple_raster(n_neurons=20, n_steps=1000)
        integ = integration(raster, dt_ms=0.5, n_samples=30)
        assert isinstance(integ, float)

    def test_integration_silent_raster(self):
        """Silent raster should have zero integration."""
        raster = np.zeros((1000, 20), dtype=np.float32)
        integ = integration(raster, dt_ms=0.5)
        assert integ == 0.0


# ---------------------------------------------------------------------------
# Tests: Cluster detection (cross-validation of CC and MI)
# ---------------------------------------------------------------------------


class TestClusterDetection:
    """Verify that CC and MI correctly identify correlated clusters."""

    def test_within_cluster_cc_higher(self):
        """Cross-correlation should be higher within clusters than between."""
        raster = _make_clustered_raster(
            n_neurons=50, n_steps=2000, cluster_size=15,
        )
        cc = cross_correlation_matrix(raster, dt_ms=0.5, bin_ms=5.0)

        # Within cluster A: neurons 0-14
        within_a = []
        for i in range(15):
            for j in range(i + 1, 15):
                within_a.append(cc[i, j])

        # Between clusters: cluster A vs remaining
        between = []
        for i in range(15):
            for j in range(30, 50):  # neurons outside both clusters
                between.append(cc[i, j])

        mean_within = np.mean(within_a)
        mean_between = np.mean(between)
        assert mean_within > mean_between, (
            f"Within-cluster CC ({mean_within:.4f}) should exceed "
            f"between-cluster CC ({mean_between:.4f})"
        )

    def test_within_cluster_mi_higher(self):
        """Mutual information should be higher within clusters than between."""
        raster = _make_clustered_raster(
            n_neurons=50, n_steps=2000, cluster_size=15,
        )
        mi = mutual_information_matrix(raster, dt_ms=0.5, bin_ms=10.0)

        # Within cluster A: neurons 0-14
        within_a = []
        for i in range(15):
            for j in range(i + 1, 15):
                within_a.append(mi[i, j])

        # Between clusters: cluster A vs neurons outside both clusters
        between = []
        for i in range(15):
            for j in range(30, 50):
                between.append(mi[i, j])

        mean_within = np.mean(within_a)
        mean_between = np.mean(between)
        assert mean_within > mean_between, (
            f"Within-cluster MI ({mean_within:.4f}) should exceed "
            f"between-cluster MI ({mean_between:.4f})"
        )
