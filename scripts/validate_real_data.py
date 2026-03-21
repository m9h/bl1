#!/usr/bin/env python3
"""Validate BL-1 against real-world electrophysiology datasets.

Loads downloaded NWB/HDF5 recordings, computes statistics, then runs
matched BL-1 simulations and compares both to published ranges.

Usage:
    python scripts/validate_real_data.py --data-dir /data/datasets/bl1
    python scripts/validate_real_data.py --data-dir /data/datasets/bl1 --max-files 5
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bl1.analysis.bursts import burst_statistics, detect_bursts
from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
    nmda_step,
)
from bl1.network.topology import build_connectivity, place_neurons
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step
from bl1.validation.comparison import compute_culture_statistics
from bl1.validation.datasets import DATASETS, compare_statistics
from bl1.validation.loaders import (
    compute_recording_statistics,
    load_maxwell_h5,
    load_nwb_spike_trains,
    spike_trains_to_raster,
)


# ============================================================================
# Wagenaar-calibrated simulation (from run_validation.sh Step C)
# ============================================================================


def run_matched_simulation(
    n_neurons: int = 5000,
    duration_ms: float = 60_000.0,
    seed: int = 42,
) -> dict[str, float]:
    """Run a Wagenaar-calibrated BL-1 simulation and return statistics."""
    DT = 0.5
    G_EXC, G_INH = 0.12, 0.36
    NMDA_RATIO = 0.37
    U_EXC, TAU_REC = 0.30, 800.0

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=G_EXC, g_inh=G_INH,
    )
    syn = create_synapse_state(n_neurons)
    W_ampa = W_exc * (1.0 - NMDA_RATIO)
    W_nmda = W_exc * NMDA_RATIO

    U = jnp.where(is_exc, U_EXC, 0.04)
    tau_rec = jnp.where(is_exc, TAU_REC, 100.0)
    tau_fac = jnp.where(is_exc, 0.001, 1000.0)
    stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
    stp_state = init_stp_state(n_neurons, stp_params)

    n_steps = int(duration_ms / DT)
    I_noise = 1.0 + 3.0 * jax.random.normal(k4, (n_steps, n_neurons))

    def step_fn(carry, I_t):
        ns, ss, st = carry
        ns = izhikevich_step(ns, params, compute_synaptic_current(ss, ns.v) + I_t, DT)
        st, scale = stp_step(st, stp_params, ns.spikes, DT)
        nr, nd, _ = nmda_step(ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT)
        ss = SynapseState(
            ampa_step(ss.g_ampa, scale, W_ampa, DT),
            gaba_a_step(ss.g_gaba_a, scale, W_inh, DT),
            nr, nd, ss.g_gaba_b_rise, ss.g_gaba_b_decay,
        )
        return (ns, ss, st), ns.spikes

    t0 = time.perf_counter()
    (_, _, _), spikes = jax.lax.scan(step_fn, (state, syn, stp_state), I_noise)
    spikes.block_until_ready()
    wall = time.perf_counter() - t0

    raster = np.asarray(spikes)
    stats = compute_culture_statistics(raster, dt_ms=DT, burst_threshold_std=1.5)
    stats["_wall_time_s"] = wall
    stats["_realtime_ratio"] = (duration_ms / 1000.0) / wall
    return stats


# ============================================================================
# Process a single recording file
# ============================================================================


def process_recording(
    filepath: str,
    dt_ms: float = 0.5,
    max_duration_s: float = 120.0,
) -> dict | None:
    """Load a recording and compute statistics. Returns None on failure.

    For very long recordings, only the first ``max_duration_s`` seconds
    are analyzed to avoid memory issues with raster construction.
    """
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".nwb":
            data = load_nwb_spike_trains(filepath)
        elif ext in (".h5", ".hdf5"):
            data = load_maxwell_h5(filepath)
        else:
            return None

        if data["n_units"] < 5:
            return None

        # Clamp duration and trim spike trains to avoid OOM on long recordings
        actual_dur = data["duration_s"]
        if actual_dur <= 0:
            # Infer from spike times
            all_times = [st for st in data["spike_times"] if len(st) > 0]
            if all_times:
                actual_dur = float(max(st.max() for st in all_times))
            else:
                return None

        # If spike times look like sample indices (unreasonably long),
        # try to detect and convert. HD-MEA standard is 20kHz.
        if actual_dur > 86400:  # > 24 hours — likely sample indices
            sr = data.get("sampling_rate", 20000.0)
            actual_dur = actual_dur / sr
            data["spike_times"] = [st / sr for st in data["spike_times"]]
            data["duration_s"] = actual_dur

        use_dur = min(actual_dur, max_duration_s)
        if use_dur < 10:
            return None

        # Trim spike trains to window
        trimmed_times = []
        for st in data["spike_times"]:
            trimmed_times.append(st[st <= use_dur])

        trimmed_data = {
            "spike_times": trimmed_times,
            "duration_s": use_dur,
            "n_units": data["n_units"],
        }

        stats = compute_recording_statistics(
            trimmed_data, dt_ms=dt_ms, burst_threshold_std=1.5,
        )
        stats["n_units"] = data["n_units"]
        stats["duration_s"] = use_dur
        stats["full_duration_s"] = actual_dur
        stats["filepath"] = filepath
        return stats

    except Exception as e:
        print(f"    WARN: Failed to load {filepath}: {e}")
        return None


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate BL-1 against real data")
    parser.add_argument("--data-dir", default="/data/datasets/bl1",
                        help="Path to downloaded datasets")
    parser.add_argument("--max-files", type=int, default=20,
                        help="Max files to process per dataset")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip the BL-1 simulation comparison")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    print("=" * 78)
    print("  BL-1 Real-World Data Validation")
    print("=" * 78)
    print(f"  Data dir:  {data_dir}")
    print(f"  Max files: {args.max_files}")
    print()

    # --- Discover files -------------------------------------------------------
    dataset_dirs = {
        "DANDI 001611 (rat cortical)": data_dir / "dandi_001611_rat_cortical",
        "DishBrain (OSF)": data_dir / "osf_dishbrain",
        "Sharf 2022 organoid": data_dir / "zenodo_sharf_2022",
    }

    all_recording_stats = []

    for ds_name, ds_path in dataset_dirs.items():
        if not ds_path.exists():
            print(f"  [{ds_name}] Not downloaded yet — skipping")
            continue

        # Find NWB and HDF5 files
        files = sorted(ds_path.rglob("*.nwb")) + sorted(ds_path.rglob("*.h5"))
        if not files:
            print(f"  [{ds_name}] No .nwb or .h5 files found in {ds_path}")
            continue

        print(f"  [{ds_name}] Found {len(files)} files, processing up to {args.max_files}...")

        processed = 0
        for f in files[:args.max_files]:
            stats = process_recording(str(f))
            if stats is not None:
                stats["dataset"] = ds_name
                all_recording_stats.append(stats)
                processed += 1
                print(f"    {f.name}: {stats['n_units']} units, "
                      f"{stats['duration_s']:.0f}s, "
                      f"FR={stats['mean_firing_rate_hz']:.2f} Hz, "
                      f"bursts={stats['burst_rate_per_min']:.1f}/min")

        print(f"    Processed: {processed}/{min(len(files), args.max_files)}")
        print()

    if not all_recording_stats:
        print("  No recordings were successfully loaded. Check that datasets are downloaded.")
        return

    # --- Aggregate real-data statistics ----------------------------------------
    print("=" * 78)
    print("  Real-Data Summary (aggregated across recordings)")
    print("=" * 78)

    fr_vals = [s["mean_firing_rate_hz"] for s in all_recording_stats]
    br_vals = [s["burst_rate_per_min"] for s in all_recording_stats]
    dur_vals = [s["burst_duration_mean_ms"] for s in all_recording_stats
                if not math.isnan(s.get("burst_duration_mean_ms", float("nan")))]

    print(f"  Recordings loaded:    {len(all_recording_stats)}")
    print(f"  Firing rate (Hz):     {np.mean(fr_vals):.2f} +/- {np.std(fr_vals):.2f}  "
          f"[{np.min(fr_vals):.2f} - {np.max(fr_vals):.2f}]")
    print(f"  Burst rate (/min):    {np.mean(br_vals):.1f} +/- {np.std(br_vals):.1f}  "
          f"[{np.min(br_vals):.1f} - {np.max(br_vals):.1f}]")
    if dur_vals:
        print(f"  Burst duration (ms):  {np.mean(dur_vals):.0f} +/- {np.std(dur_vals):.0f}  "
              f"[{np.min(dur_vals):.0f} - {np.max(dur_vals):.0f}]")
    print()

    # --- BL-1 simulation comparison -------------------------------------------
    if not args.skip_sim:
        print("=" * 78)
        print("  BL-1 Simulation (Wagenaar-calibrated, 60s)")
        print("=" * 78)

        sim_stats = run_matched_simulation()
        print(f"  Wall time:        {sim_stats['_wall_time_s']:.1f}s "
              f"({sim_stats['_realtime_ratio']:.1f}x realtime)")
        print(f"  Firing rate:      {sim_stats['mean_firing_rate_hz']:.2f} Hz")
        print(f"  Burst rate:       {sim_stats['burst_rate_per_min']:.1f}/min")
        dur = sim_stats.get("burst_duration_mean_ms", float("nan"))
        print(f"  Burst duration:   {dur:.0f} ms" if not math.isnan(dur) else "  Burst duration:   N/A")
        print()

        # Compare both to Wagenaar ranges
        print("=" * 78)
        print("  Comparison: Real Data vs BL-1 vs Wagenaar 2006 Ranges")
        print("=" * 78)

        real_mean_stats = {
            "mean_firing_rate_hz": np.mean(fr_vals),
            "burst_rate_per_min": np.mean(br_vals),
        }
        if dur_vals:
            real_mean_stats["burst_duration_mean_ms"] = np.mean(dur_vals)

        print(f"\n  {'Metric':<30s} {'Real Data':>12s} {'BL-1 Sim':>12s} {'Wagenaar Range':>20s}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*20}")

        metrics = [
            ("mean_firing_rate_hz", "Firing rate (Hz)", (0.1, 5.0)),
            ("burst_rate_per_min", "Burst rate (/min)", (0.2, 20.0)),
            ("burst_duration_mean_ms", "Burst duration (ms)", (100.0, 2000.0)),
        ]

        for key, label, wag_range in metrics:
            real_val = real_mean_stats.get(key, float("nan"))
            sim_val = sim_stats.get(key, float("nan"))
            real_s = f"{real_val:.2f}" if not math.isnan(real_val) else "N/A"
            sim_s = f"{sim_val:.2f}" if not math.isnan(sim_val) else "N/A"
            range_s = f"[{wag_range[0]:.1f}, {wag_range[1]:.1f}]"
            print(f"  {label:<30s} {real_s:>12s} {sim_s:>12s} {range_s:>20s}")

    print()
    print("=" * 78)
    print("  Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
