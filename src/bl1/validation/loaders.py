"""Load external electrophysiology data for comparison with BL-1 simulations.

Supports two data sources commonly used in cortical culture research:

- **NWB files** (Neurodata Without Borders) from DANDI and other archives.
  Requires the optional ``pynwb`` package (install via ``pip install bl1[nwb]``).
- **Maxwell Biosystems HDF5 files** from Zenodo, Dryad, and similar
  repositories.  Uses ``h5py`` which is a core BL-1 dependency.

Both loaders return a uniform dict structure with spike times, unit IDs,
electrode positions, and metadata.  The helper functions
:func:`spike_trains_to_raster` and :func:`compute_recording_statistics`
convert loaded data into formats compatible with BL-1 analysis routines.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Optional dependency: pynwb
# ---------------------------------------------------------------------------

try:
    import pynwb  # noqa: F401

    _PYNWB_AVAILABLE = True
except ImportError:
    _PYNWB_AVAILABLE = False


# ---------------------------------------------------------------------------
# NWB loader (DANDI datasets)
# ---------------------------------------------------------------------------


def load_nwb_spike_trains(filepath: str) -> dict:
    """Load spike trains from an NWB file.

    Reads spike-sorted unit data from the ``Units`` table in an NWB file.
    If the file contains electrode position metadata, that is extracted
    as well.

    Args:
        filepath: Path to a ``.nwb`` file.

    Returns:
        Dict with the following keys:

        - ``spike_times``: list of 1-D ``np.ndarray``, one per unit
          (times in seconds).
        - ``unit_ids``: list of unit identifiers (int or str).
        - ``duration_s``: total recording duration in seconds.
        - ``n_units``: number of spike-sorted units.
        - ``electrode_positions``: ``(n_units, 2)`` array of (x, y)
          positions in micrometres if available, otherwise ``None``.
        - ``metadata``: dict of session-level metadata (identifier,
          description, session start time, etc.).

    Raises:
        ImportError: If ``pynwb`` is not installed.
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If no spike-sorted units are found in the file.
    """
    if not _PYNWB_AVAILABLE:
        raise ImportError(
            "pynwb is required to load NWB files.  Install it with:\n"
            "    pip install bl1[nwb]\n"
            "or:\n"
            "    pip install pynwb>=2.0"
        )

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(filepath, "r") as io:
        nwbfile = io.read()

        # --- Session metadata ------------------------------------------------
        metadata: dict = {
            "identifier": nwbfile.identifier,
            "session_description": getattr(nwbfile, "session_description", ""),
            "session_start_time": str(getattr(nwbfile, "session_start_time", "")),
        }

        # --- Recording duration ----------------------------------------------
        # Try to get duration from trials, acquisition, or epochs
        duration_s = _nwb_infer_duration(nwbfile)

        # --- Units table (spike-sorted data) ---------------------------------
        if nwbfile.units is None:
            raise ValueError(
                f"No Units table found in {filepath}.  "
                "This file may contain only raw acquisition data."
            )

        units_table = nwbfile.units
        n_units = len(units_table)

        spike_times: list[np.ndarray] = []
        unit_ids: list = []

        for idx in range(n_units):
            st = np.asarray(units_table["spike_times"][idx], dtype=np.float64)
            spike_times.append(st)

            # Try to get a meaningful unit ID
            if "unit_id" in units_table.colnames:
                unit_ids.append(units_table["unit_id"][idx])
            else:
                unit_ids.append(idx)

        # Refine duration from spike times if needed
        if duration_s <= 0.0 and spike_times:
            all_times = np.concatenate([st for st in spike_times if len(st) > 0])
            if len(all_times) > 0:
                duration_s = float(np.max(all_times))

        # --- Electrode positions (optional) ----------------------------------
        electrode_positions = _nwb_extract_electrode_positions(nwbfile, units_table, n_units)

    return {
        "spike_times": spike_times,
        "unit_ids": unit_ids,
        "duration_s": duration_s,
        "n_units": n_units,
        "electrode_positions": electrode_positions,
        "metadata": metadata,
    }


def _nwb_infer_duration(nwbfile) -> float:
    """Try to infer total recording duration from an NWB file.

    Checks (in order): acquisition time-series length, trials table,
    epochs table.  Returns 0.0 if nothing is found.
    """
    # Check acquisition time series
    for name in nwbfile.acquisition:
        ts = nwbfile.acquisition[name]
        if hasattr(ts, "timestamps") and ts.timestamps is not None:
            return float(np.max(ts.timestamps))
        if hasattr(ts, "data") and hasattr(ts, "rate") and ts.rate is not None:
            n_samples = ts.data.shape[0] if hasattr(ts.data, "shape") else len(ts.data)
            return n_samples / ts.rate

    # Check trials
    if (
        nwbfile.trials is not None
        and len(nwbfile.trials) > 0
        and "stop_time" in nwbfile.trials.colnames
    ):
        return float(np.max(nwbfile.trials["stop_time"][:]))

    # Check epochs
    if (
        nwbfile.epochs is not None
        and len(nwbfile.epochs) > 0
        and "stop_time" in nwbfile.epochs.colnames
    ):
        return float(np.max(nwbfile.epochs["stop_time"][:]))

    return 0.0


def _nwb_extract_electrode_positions(nwbfile, units_table, n_units: int):
    """Extract electrode positions for each unit if available.

    Returns an (n_units, 2) ndarray or None.
    """
    positions = []

    # Option 1: units table has 'electrodes' column referencing the electrodes table
    if "electrodes" in units_table.colnames and nwbfile.electrodes is not None:
        elec_table = nwbfile.electrodes
        has_x = "x" in elec_table.colnames
        has_y = "y" in elec_table.colnames

        if has_x and has_y:
            for idx in range(n_units):
                try:
                    elec_idx = units_table["electrodes"][idx]
                    # elec_idx may be a DynamicTableRegion; take the first element
                    if hasattr(elec_idx, "__len__"):
                        elec_idx = elec_idx[0] if len(elec_idx) > 0 else 0
                    x = float(elec_table["x"][int(elec_idx)])
                    y = float(elec_table["y"][int(elec_idx)])
                    positions.append((x, y))
                except (IndexError, TypeError, KeyError):
                    positions.append((np.nan, np.nan))

    # Option 2: electrodes table exists but units don't reference it directly
    if not positions and nwbfile.electrodes is not None:
        elec_table = nwbfile.electrodes
        if "x" in elec_table.colnames and "y" in elec_table.colnames:
            n_elec = len(elec_table)
            if n_elec == n_units:
                for idx in range(n_elec):
                    x = float(elec_table["x"][idx])
                    y = float(elec_table["y"][idx])
                    positions.append((x, y))

    if positions and len(positions) == n_units:
        return np.array(positions, dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Maxwell Biosystems HDF5 loader
# ---------------------------------------------------------------------------


def load_maxwell_h5(filepath: str) -> dict:
    """Load spike data from a Maxwell Biosystems HDF5 file.

    Maxwell Biosystems (MaxOne / MaxTwo) saves processed spike data in
    HDF5 files with a standard internal layout::

        /proc0/spikeTimes      -- spike time samples (int64)
        /proc0/spikeChannels   -- channel index per spike (int64)
        /mapping               -- electrode mapping table with x, y, channel
        /settings              -- recording settings (sampling rate, etc.)

    Args:
        filepath: Path to a Maxwell ``.h5`` file.

    Returns:
        Dict with the following keys:

        - ``spike_times``: list of 1-D ``np.ndarray``, one per unit
          (times in seconds, converted from sample indices).
        - ``unit_ids``: list of channel identifiers (integers).
        - ``duration_s``: recording duration in seconds.
        - ``n_units``: number of channels with spikes.
        - ``electrode_positions``: ``(n_units, 2)`` array of (x, y)
          in micrometres.
        - ``sampling_rate``: sampling rate in Hz.
        - ``metadata``: dict of recording settings.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        KeyError: If the expected HDF5 groups are not present.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        # --- Sampling rate ---------------------------------------------------
        sampling_rate = _maxwell_get_sampling_rate(f)

        # --- Spike data ------------------------------------------------------
        spike_samples, spike_channels = _maxwell_read_spikes(f)

        # --- Electrode mapping -----------------------------------------------
        channel_positions, channel_ids = _maxwell_read_mapping(f)

        # --- Recording metadata ----------------------------------------------
        metadata = _maxwell_read_settings(f)
        metadata["sampling_rate_hz"] = sampling_rate

        # --- Infer duration --------------------------------------------------
        if "duration_s" in metadata:
            duration_s = float(metadata["duration_s"])
        elif len(spike_samples) > 0:
            duration_s = float(np.max(spike_samples)) / sampling_rate
        else:
            duration_s = 0.0

    # --- Group spikes by channel ---------------------------------------------
    unique_channels = np.unique(spike_channels)
    spike_times_list: list[np.ndarray] = []
    unit_ids: list[int] = []
    position_list: list[tuple[float, float]] = []

    for ch in unique_channels:
        mask = spike_channels == ch
        times_s = spike_samples[mask].astype(np.float64) / sampling_rate
        spike_times_list.append(np.sort(times_s))
        unit_ids.append(int(ch))

        # Look up electrode position
        if ch in channel_positions:
            position_list.append(channel_positions[ch])
        else:
            position_list.append((np.nan, np.nan))

    electrode_positions = np.array(position_list, dtype=np.float64) if position_list else None

    return {
        "spike_times": spike_times_list,
        "unit_ids": unit_ids,
        "duration_s": duration_s,
        "n_units": len(unit_ids),
        "electrode_positions": electrode_positions,
        "sampling_rate": sampling_rate,
        "metadata": metadata,
    }


def _maxwell_get_sampling_rate(f) -> float:
    """Extract sampling rate from a Maxwell HDF5 file."""
    # Try /settings group first
    if "settings" in f:
        settings = f["settings"]
        for key in ("lsb", "sampling", "samplingRate", "fs"):
            if key in settings.attrs:
                return float(settings.attrs[key])
        # Try as a dataset
        for key in ("sampling", "samplingRate", "fs"):
            if key in settings:
                return float(np.array(settings[key]).flat[0])

    # Try root attributes
    for key in ("sampling_rate", "samplingRate", "fs"):
        if key in f.attrs:
            return float(f.attrs[key])

    # Default Maxwell sampling rate
    return 20000.0


def _maxwell_read_spikes(f) -> tuple[np.ndarray, np.ndarray]:
    """Read spike times and channel indices from a Maxwell HDF5 file."""
    # Standard Maxwell layout: /proc0/spikeTimes, /proc0/spikeChannels
    if "proc0" in f:
        proc = f["proc0"]
        if "spikeTimes" in proc and "spikeChannels" in proc:
            spike_samples = np.asarray(proc["spikeTimes"], dtype=np.int64)
            spike_channels = np.asarray(proc["spikeChannels"], dtype=np.int64)
            return spike_samples, spike_channels

    # Alternative: /spikeTimes at root level
    if "spikeTimes" in f and "spikeChannels" in f:
        spike_samples = np.asarray(f["spikeTimes"], dtype=np.int64)
        spike_channels = np.asarray(f["spikeChannels"], dtype=np.int64)
        return spike_samples, spike_channels

    # Alternative: compound dataset with fields (frameno, channel, ...)
    # Check both "spikes" and "spikeTimes" keys (Sharf 2022 uses the latter)
    for group_name in ("proc0", ""):
        grp = f[group_name] if group_name else f
        for ds_name in ("spikes", "spikeTimes"):
            if ds_name in grp:
                ds = grp[ds_name]
                if (
                    hasattr(ds.dtype, "names")
                    and ds.dtype.names is not None
                    and "frameno" in ds.dtype.names
                    and "channel" in ds.dtype.names
                ):
                    spike_samples = np.asarray(ds["frameno"], dtype=np.int64)
                    spike_channels = np.asarray(ds["channel"], dtype=np.int64)
                    return spike_samples, spike_channels

    raise KeyError(
        "Could not locate spike data in the HDF5 file.  "
        "Expected /proc0/spikeTimes and /proc0/spikeChannels."
    )


def _maxwell_read_mapping(f) -> tuple[dict[int, tuple[float, float]], list[int]]:
    """Read electrode position mapping from a Maxwell HDF5 file.

    Returns:
        Tuple of (channel_positions, channel_ids) where channel_positions
        maps channel_id -> (x_um, y_um).
    """
    channel_positions: dict[int, tuple[float, float]] = {}
    channel_ids: list[int] = []

    if "mapping" in f:
        mapping = f["mapping"]

        # Compound dataset with named fields
        if hasattr(mapping.dtype, "names") and mapping.dtype.names is not None:
            data = np.asarray(mapping)
            ch_field = None
            x_field = None
            y_field = None

            for name in mapping.dtype.names:
                lower = name.lower()
                if lower in ("channel", "ch", "electrode"):
                    ch_field = name
                elif lower == "x":
                    x_field = name
                elif lower == "y":
                    y_field = name

            if ch_field and x_field and y_field:
                for row in data:
                    ch = int(row[ch_field])
                    x = float(row[x_field])
                    y = float(row[y_field])
                    channel_positions[ch] = (x, y)
                    channel_ids.append(ch)

        # Group with separate datasets
        elif isinstance(mapping, dict) or hasattr(mapping, "keys"):
            if "channel" in mapping and "x" in mapping and "y" in mapping:
                channels = np.asarray(mapping["channel"], dtype=np.int64)
                xs = np.asarray(mapping["x"], dtype=np.float64)
                ys = np.asarray(mapping["y"], dtype=np.float64)
                for ch, x, y in zip(channels, xs, ys):
                    channel_positions[int(ch)] = (float(x), float(y))
                    channel_ids.append(int(ch))

    return channel_positions, channel_ids


def _maxwell_read_settings(f) -> dict:
    """Read recording settings from a Maxwell HDF5 file."""
    metadata: dict = {}

    if "settings" in f:
        settings = f["settings"]
        for key in settings.attrs:
            val = settings.attrs[key]
            # Convert numpy scalars to Python types
            if hasattr(val, "item"):
                val = val.item()
            elif isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            metadata[key] = val

    # Also grab root-level attributes
    for key in f.attrs:
        val = f.attrs[key]
        if hasattr(val, "item"):
            val = val.item()
        elif isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        metadata[f"root_{key}"] = val

    return metadata


# ---------------------------------------------------------------------------
# Spike-train to raster conversion
# ---------------------------------------------------------------------------


def spike_trains_to_raster(
    spike_times: list[np.ndarray],
    duration_s: float,
    dt: float = 0.5e-3,
) -> NDArray:
    """Convert spike time lists to a binary raster matrix.

    Produces a ``(n_steps, n_units)`` boolean raster compatible with
    BL-1 analysis functions (burst detection, criticality, etc.).

    Args:
        spike_times: List of 1-D arrays, each containing spike times
            in seconds for one unit.
        duration_s: Total recording duration in seconds.
        dt: Time bin width in seconds (default 0.5 ms = 0.0005 s).

    Returns:
        Binary raster of shape ``(n_steps, n_units)`` where
        ``n_steps = ceil(duration_s / dt)``.  Entry ``[t, u]`` is 1
        if unit *u* fired at least once during time bin *t*.

    Raises:
        ValueError: If *duration_s* <= 0 or *dt* <= 0.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    n_steps = int(np.ceil(duration_s / dt))
    n_units = len(spike_times)
    raster = np.zeros((n_steps, n_units), dtype=np.float32)

    for u, st in enumerate(spike_times):
        if len(st) == 0:
            continue
        # Convert times to bin indices
        bins = np.floor(st / dt).astype(np.int64)
        # Clip to valid range
        bins = bins[(bins >= 0) & (bins < n_steps)]
        raster[bins, u] = 1.0

    return raster


# ---------------------------------------------------------------------------
# Recording statistics (compatible with bl1.validation.datasets)
# ---------------------------------------------------------------------------


def compute_recording_statistics(
    spike_data: dict,
    dt_ms: float = 0.5,
    burst_threshold_std: float = 2.0,
    burst_min_duration_ms: float = 50.0,
) -> dict[str, float]:
    """Compute standard statistics from loaded spike data.

    Converts the spike data to a raster and then uses BL-1's built-in
    burst detection to compute statistics that are directly comparable
    to published dataset ranges via
    :func:`bl1.validation.datasets.compare_statistics`.

    Args:
        spike_data: Dict as returned by :func:`load_nwb_spike_trains` or
            :func:`load_maxwell_h5`.  Must contain ``spike_times`` and
            ``duration_s``.
        dt_ms: Time bin width in milliseconds for raster construction
            (default 0.5 ms).
        burst_threshold_std: Standard deviations for burst onset
            detection (default 2.0).
        burst_min_duration_ms: Minimum burst duration in ms
            (default 50.0).

    Returns:
        Dict with the following keys (all ``float``):

        - ``mean_firing_rate_hz`` -- Mean firing rate across units.
        - ``burst_rate_per_min`` -- Number of detected bursts per minute.
        - ``ibi_mean_ms`` -- Mean inter-burst interval in ms.
        - ``burst_duration_mean_ms`` -- Mean burst duration in ms.
        - ``recruitment_mean`` -- Mean fraction of units recruited per
          burst.
    """
    from bl1.analysis.bursts import burst_statistics, detect_bursts

    spike_times = spike_data["spike_times"]
    duration_s = spike_data["duration_s"]
    n_units = spike_data.get("n_units", len(spike_times))

    if duration_s <= 0 or n_units == 0:
        return {
            "mean_firing_rate_hz": 0.0,
            "burst_rate_per_min": 0.0,
            "ibi_mean_ms": float("nan"),
            "burst_duration_mean_ms": float("nan"),
            "recruitment_mean": float("nan"),
        }

    # Convert to raster
    dt_s = dt_ms / 1000.0
    raster = spike_trains_to_raster(spike_times, duration_s, dt=dt_s)

    # --- Firing rate ---------------------------------------------------------
    total_spikes = float(raster.sum())
    mean_firing_rate = total_spikes / (n_units * duration_s) if n_units > 0 else 0.0

    # --- Burst detection -----------------------------------------------------
    bursts = detect_bursts(
        raster,
        dt_ms=dt_ms,
        threshold_std=burst_threshold_std,
        min_duration_ms=burst_min_duration_ms,
    )
    bstats = burst_statistics(bursts)

    total_time_min = duration_s / 60.0
    burst_rate = len(bursts) / total_time_min if total_time_min > 0 else 0.0

    return {
        "mean_firing_rate_hz": mean_firing_rate,
        "burst_rate_per_min": burst_rate,
        "ibi_mean_ms": bstats["ibi_mean"],
        "burst_duration_mean_ms": bstats["duration_mean"],
        "recruitment_mean": bstats["recruitment_mean"],
    }
