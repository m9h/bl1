"""Tests for bl1.validation.loaders -- external data loading utilities."""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
import pytest

from bl1.validation.loaders import (
    compute_recording_statistics,
    load_maxwell_h5,
    load_nwb_spike_trains,
    spike_trains_to_raster,
)

# ============================================================================
# spike_trains_to_raster
# ============================================================================


class TestSpikeTrainsToRaster:
    """Tests for the spike-train to raster conversion function."""

    def test_basic_conversion(self):
        """Single spike per unit placed in the correct bin."""
        spike_times = [
            np.array([0.001]),  # 1 ms -> bin 2 (0.5 ms bins)
            np.array([0.005]),  # 5 ms -> bin 10
        ]
        raster = spike_trains_to_raster(spike_times, duration_s=0.01, dt=0.5e-3)

        assert raster.shape == (20, 2)
        assert raster.dtype == np.float32

        # Unit 0 spike at 1 ms -> bin index floor(0.001 / 0.0005) = 2
        assert raster[2, 0] == 1.0
        # Unit 1 spike at 5 ms -> bin index floor(0.005 / 0.0005) = 10
        assert raster[10, 1] == 1.0

        # Total spike count should be 2
        assert raster.sum() == 2.0

    def test_empty_spike_trains(self):
        """Units with no spikes produce an all-zero raster."""
        spike_times = [np.array([]), np.array([])]
        raster = spike_trains_to_raster(spike_times, duration_s=1.0, dt=1e-3)

        assert raster.shape == (1000, 2)
        assert raster.sum() == 0.0

    def test_multiple_spikes_same_bin(self):
        """Multiple spikes in the same bin produce a value of 1 (binary)."""
        spike_times = [np.array([0.0001, 0.0002, 0.0003])]  # all in bin 0
        raster = spike_trains_to_raster(spike_times, duration_s=0.001, dt=0.5e-3)

        assert raster[0, 0] == 1.0
        assert raster.sum() == 1.0  # binary raster

    def test_spike_at_boundary(self):
        """Spikes exactly at the duration boundary are excluded."""
        spike_times = [np.array([1.0])]  # exactly at the end
        raster = spike_trains_to_raster(spike_times, duration_s=1.0, dt=0.5e-3)

        # Bin index = floor(1.0 / 0.0005) = 2000, but n_steps = 2000
        # so bin 2000 is out of range -> spike clipped
        assert raster.sum() == 0.0

    def test_many_units(self):
        """Handles a large number of units efficiently."""
        rng = np.random.default_rng(42)
        n_units = 100
        spike_times = [rng.uniform(0, 10, size=50) for _ in range(n_units)]
        raster = spike_trains_to_raster(spike_times, duration_s=10.0, dt=1e-3)

        assert raster.shape == (10000, 100)
        assert raster.sum() > 0

    def test_custom_dt(self):
        """Non-default time bin width produces correctly sized raster."""
        spike_times = [np.array([0.5])]
        raster = spike_trains_to_raster(spike_times, duration_s=1.0, dt=0.01)

        assert raster.shape == (100, 1)
        # Bin index = floor(0.5 / 0.01) = 50
        assert raster[50, 0] == 1.0

    def test_invalid_duration_raises(self):
        """Negative or zero duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_s must be positive"):
            spike_trains_to_raster([np.array([0.1])], duration_s=0.0)
        with pytest.raises(ValueError, match="duration_s must be positive"):
            spike_trains_to_raster([np.array([0.1])], duration_s=-1.0)

    def test_invalid_dt_raises(self):
        """Negative or zero dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            spike_trains_to_raster([np.array([0.1])], duration_s=1.0, dt=0.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            spike_trains_to_raster([np.array([0.1])], duration_s=1.0, dt=-0.001)

    def test_no_units(self):
        """Empty list of spike trains produces zero-column raster."""
        raster = spike_trains_to_raster([], duration_s=1.0, dt=1e-3)
        assert raster.shape == (1000, 0)


# ============================================================================
# compute_recording_statistics
# ============================================================================


class TestComputeRecordingStatistics:
    """Tests for computing statistics from loaded spike data."""

    def _make_bursty_spike_data(self, rng: np.random.Generator) -> dict:
        """Create synthetic spike data with periodic bursts.

        Generates 20 units with Poisson background firing plus periodic
        synchronous bursts every ~5 seconds.  Burst events are intense
        (200 Hz per unit, 100 ms duration) to ensure they are clearly
        detectable at typical bin widths.
        """
        n_units = 20
        duration_s = 60.0  # 1 minute
        burst_interval_s = 5.0
        burst_duration_s = 0.1
        burst_rate_hz = 200.0  # very high rate during bursts
        background_rate_hz = 0.3  # low background

        spike_times_list: list[np.ndarray] = []

        for _ in range(n_units):
            spikes: list[np.ndarray] = []

            # Background spikes
            n_background = rng.poisson(background_rate_hz * duration_s)
            if n_background > 0:
                bg_spikes = rng.uniform(0, duration_s, size=n_background)
                spikes.append(bg_spikes)

            # Burst spikes -- all units participate in every burst
            burst_starts = np.arange(burst_interval_s, duration_s, burst_interval_s)
            for bs in burst_starts:
                n_burst = rng.poisson(burst_rate_hz * burst_duration_s)
                if n_burst > 0:
                    burst_spikes = rng.uniform(bs, bs + burst_duration_s, size=n_burst)
                    spikes.append(burst_spikes)

            all_spikes = np.concatenate(spikes) if spikes else np.array([])
            all_spikes = np.sort(all_spikes[all_spikes < duration_s])
            spike_times_list.append(all_spikes)

        return {
            "spike_times": spike_times_list,
            "duration_s": duration_s,
            "n_units": n_units,
        }

    def test_basic_statistics_keys(self):
        """Verify all expected keys are present in the output."""
        rng = np.random.default_rng(42)
        spike_data = self._make_bursty_spike_data(rng)
        stats = compute_recording_statistics(spike_data)

        expected_keys = {
            "mean_firing_rate_hz",
            "burst_rate_per_min",
            "ibi_mean_ms",
            "burst_duration_mean_ms",
            "recruitment_mean",
        }
        assert set(stats.keys()) == expected_keys

    def test_firing_rate_positive(self):
        """Mean firing rate should be positive for data with spikes."""
        rng = np.random.default_rng(42)
        spike_data = self._make_bursty_spike_data(rng)
        stats = compute_recording_statistics(spike_data)

        assert stats["mean_firing_rate_hz"] > 0

    def test_burst_detection(self):
        """Bursty data should yield a non-zero burst rate."""
        rng = np.random.default_rng(42)
        spike_data = self._make_bursty_spike_data(rng)
        # Use a coarser bin (5 ms) so the burst detector can aggregate
        # enough spikes per bin to cross the threshold.
        stats = compute_recording_statistics(spike_data, dt_ms=5.0)

        assert stats["burst_rate_per_min"] > 0

    def test_firing_rate_accuracy(self):
        """Verify firing rate is approximately correct for known input."""
        # Each unit fires exactly once per second for 10 seconds
        n_units = 10
        duration_s = 10.0
        spike_times = [np.arange(0.5, duration_s, 1.0) for _ in range(n_units)]

        spike_data = {
            "spike_times": spike_times,
            "duration_s": duration_s,
            "n_units": n_units,
        }
        stats = compute_recording_statistics(spike_data, dt_ms=1.0)

        # Each unit fires ~10 spikes in 10 seconds -> ~1 Hz
        assert 0.5 < stats["mean_firing_rate_hz"] < 2.0

    def test_empty_data(self):
        """Empty spike data returns zero/NaN statistics without crashing."""
        spike_data = {
            "spike_times": [np.array([])],
            "duration_s": 10.0,
            "n_units": 1,
        }
        stats = compute_recording_statistics(spike_data)

        assert stats["mean_firing_rate_hz"] == 0.0
        assert stats["burst_rate_per_min"] == 0.0

    def test_zero_duration(self):
        """Zero duration returns safe defaults."""
        spike_data = {
            "spike_times": [],
            "duration_s": 0.0,
            "n_units": 0,
        }
        stats = compute_recording_statistics(spike_data)

        assert stats["mean_firing_rate_hz"] == 0.0
        assert stats["burst_rate_per_min"] == 0.0

    def test_compatible_with_compare_statistics(self):
        """Output dict can be passed to compare_statistics without error."""
        from bl1.validation.datasets import compare_statistics

        rng = np.random.default_rng(42)
        spike_data = self._make_bursty_spike_data(rng)
        stats = compute_recording_statistics(spike_data)

        # Should not raise
        result = compare_statistics(stats, "wagenaar_2006")
        assert isinstance(result, dict)
        # At least mean_firing_rate_hz should be comparable
        assert "mean_firing_rate_hz" in result


# ============================================================================
# Maxwell HDF5 loader (with synthetic test file)
# ============================================================================


class TestLoadMaxwellH5:
    """Tests for loading Maxwell Biosystems HDF5 files."""

    def _create_test_maxwell_file(self, filepath: str) -> None:
        """Create a minimal Maxwell-format HDF5 file for testing."""
        rng = np.random.default_rng(123)

        n_channels = 5
        n_spikes_per_channel = 50
        sampling_rate = 20000.0

        # Generate spike data
        all_times: list[np.ndarray] = []
        all_channels: list[np.ndarray] = []

        for ch in range(n_channels):
            spike_samples = np.sort(
                rng.integers(0, int(10 * sampling_rate), size=n_spikes_per_channel)
            )
            all_times.append(spike_samples)
            all_channels.append(np.full(n_spikes_per_channel, ch, dtype=np.int64))

        spike_times = np.concatenate(all_times)
        spike_channels = np.concatenate(all_channels)

        # Sort by time
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spike_channels = spike_channels[sort_idx]

        # Electrode positions
        mapping_dtype = np.dtype([
            ("channel", np.int64),
            ("x", np.float64),
            ("y", np.float64),
        ])
        mapping_data = np.zeros(n_channels, dtype=mapping_dtype)
        for ch in range(n_channels):
            mapping_data[ch] = (ch, ch * 17.5, ch * 17.5)

        with h5py.File(filepath, "w") as f:
            # /proc0 group
            proc0 = f.create_group("proc0")
            proc0.create_dataset("spikeTimes", data=spike_times)
            proc0.create_dataset("spikeChannels", data=spike_channels)

            # /mapping
            f.create_dataset("mapping", data=mapping_data)

            # /settings
            settings = f.create_group("settings")
            settings.attrs["sampling"] = sampling_rate

    def test_load_synthetic_maxwell(self):
        """Load a synthetic Maxwell file and verify structure."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmppath = tmp.name

        try:
            self._create_test_maxwell_file(tmppath)
            result = load_maxwell_h5(tmppath)

            assert result["n_units"] == 5
            assert len(result["spike_times"]) == 5
            assert len(result["unit_ids"]) == 5
            assert result["sampling_rate"] == 20000.0
            assert result["duration_s"] > 0
            assert result["electrode_positions"] is not None
            assert result["electrode_positions"].shape == (5, 2)

            # Check spike times are in seconds
            for st in result["spike_times"]:
                assert len(st) > 0
                assert np.all(st >= 0)
                assert np.all(st <= result["duration_s"])
                # Verify sorted
                assert np.all(np.diff(st) >= 0)
        finally:
            os.unlink(tmppath)

    def test_maxwell_statistics(self):
        """Compute statistics from a synthetic Maxwell file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmppath = tmp.name

        try:
            self._create_test_maxwell_file(tmppath)
            result = load_maxwell_h5(tmppath)
            stats = compute_recording_statistics(result)

            assert stats["mean_firing_rate_hz"] > 0
            assert isinstance(stats["burst_rate_per_min"], float)
        finally:
            os.unlink(tmppath)

    def test_missing_file_raises(self):
        """Loading a non-existent file raises an error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_maxwell_h5("/nonexistent/path/to/file.h5")


# ============================================================================
# NWB loader (skip if no pynwb or no test file)
# ============================================================================


_NWB_TEST_FILE = os.environ.get("BL1_NWB_TEST_FILE", "")


@pytest.mark.skipif(
    not _NWB_TEST_FILE or not os.path.isfile(_NWB_TEST_FILE),
    reason="No NWB test file available (set BL1_NWB_TEST_FILE env var)",
)
class TestLoadNWBFile:
    """Tests for NWB file loading (requires real NWB test data)."""

    def test_load_nwb_structure(self):
        """Load a real NWB file and verify the output dict structure."""
        result = load_nwb_spike_trains(_NWB_TEST_FILE)

        assert "spike_times" in result
        assert "unit_ids" in result
        assert "duration_s" in result
        assert "n_units" in result
        assert "metadata" in result

        assert result["n_units"] == len(result["spike_times"])
        assert result["n_units"] == len(result["unit_ids"])
        assert result["duration_s"] > 0

    def test_nwb_statistics(self):
        """Compute statistics from a real NWB file."""
        result = load_nwb_spike_trains(_NWB_TEST_FILE)
        stats = compute_recording_statistics(result)

        assert stats["mean_firing_rate_hz"] >= 0
        assert isinstance(stats["burst_rate_per_min"], float)


class TestNWBImportGuard:
    """Test that missing pynwb raises a helpful error."""

    def test_import_error_message(self):
        """If pynwb is not available, the error message should include install instructions."""
        from bl1.validation import loaders

        original = loaders._PYNWB_AVAILABLE
        try:
            loaders._PYNWB_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install bl1"):
                load_nwb_spike_trains("/fake/file.nwb")
        finally:
            loaders._PYNWB_AVAILABLE = original


_MAXWELL_TEST_FILE = os.environ.get("BL1_MAXWELL_TEST_FILE", "")


@pytest.mark.skipif(
    not _MAXWELL_TEST_FILE or not os.path.isfile(_MAXWELL_TEST_FILE),
    reason="No Maxwell test file available (set BL1_MAXWELL_TEST_FILE env var)",
)
class TestLoadRealMaxwellFile:
    """Tests for loading real Maxwell HDF5 files (requires test data)."""

    def test_load_real_maxwell(self):
        """Load a real Maxwell file and verify basic structure."""
        result = load_maxwell_h5(_MAXWELL_TEST_FILE)

        assert result["n_units"] > 0
        assert result["duration_s"] > 0
        assert result["sampling_rate"] > 0
