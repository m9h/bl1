#!/usr/bin/env python3
"""Analyze all downloaded datasets and produce a comprehensive report.

Processes:
  - DANDI 001611: rat cortical HD-MEA (2700 NWB files, 12 subjects)
  - Sharf 2022: human brain organoid HD-MEA (33 HDF5 files)

Outputs per-recording statistics + aggregate summaries + comparison to BL-1.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from bl1.validation.loaders import (
    compute_recording_statistics,
    load_maxwell_h5,
    load_nwb_spike_trains,
    spike_trains_to_raster,
)
from bl1.validation.datasets import DATASETS

DATA_DIR = Path("/data/datasets/bl1")
RESULTS_DIR = Path("results/dataset_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def process_nwb(filepath, max_duration_s=60.0):
    """Load NWB, handle sample-index conversion, compute stats."""
    data = load_nwb_spike_trains(str(filepath))
    if data["n_units"] < 5:
        return None

    # Auto-detect sample indices
    if data["duration_s"] > 86400:
        sr = 20000.0
        data["spike_times"] = [st / sr for st in data["spike_times"]]
        data["duration_s"] /= sr

    dur = min(data["duration_s"], max_duration_s)
    if dur < 5:
        return None

    trimmed = {
        "spike_times": [st[st <= dur] for st in data["spike_times"]],
        "duration_s": dur,
        "n_units": data["n_units"],
    }
    stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
    stats["n_units"] = data["n_units"]
    stats["duration_s"] = dur
    stats["full_duration_s"] = data["duration_s"]
    return stats


def process_maxwell(filepath, max_duration_s=60.0):
    """Load Maxwell HDF5, compute stats."""
    data = load_maxwell_h5(str(filepath))
    if data["n_units"] < 5:
        return None

    # Find actual start of spiking activity (some files have late-starting data)
    all_times = [st for st in data["spike_times"] if len(st) > 0]
    if not all_times:
        return None
    t_min = float(min(st.min() for st in all_times))
    t_max = float(max(st.max() for st in all_times))
    actual_dur = t_max - t_min
    if actual_dur < 5:
        return None

    # Trim window relative to start of activity
    window_end = t_min + min(actual_dur, max_duration_s)
    trimmed_times = []
    for st in data["spike_times"]:
        mask = (st >= t_min) & (st <= window_end)
        trimmed_times.append(st[mask] - t_min)  # shift to t=0

    use_dur = min(actual_dur, max_duration_s)
    trimmed = {
        "spike_times": trimmed_times,
        "duration_s": use_dur,
        "n_units": data["n_units"],
    }
    stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
    stats["n_units"] = data["n_units"]
    stats["duration_s"] = use_dur
    stats["actual_duration_s"] = actual_dur
    stats["sampling_rate"] = data.get("sampling_rate", 20000.0)
    return stats


def summarize(records, label):
    """Print aggregate stats for a collection of recordings."""
    if not records:
        print(f"  No valid recordings for {label}")
        return {}

    fr = [r["mean_firing_rate_hz"] for r in records]
    br = [r["burst_rate_per_min"] for r in records]
    dur = [r.get("burst_duration_mean_ms", float("nan")) for r in records]
    dur = [d for d in dur if not math.isnan(d)]
    units = [r["n_units"] for r in records]

    summary = {
        "n_recordings": len(records),
        "n_units_range": [int(min(units)), int(max(units))],
        "firing_rate_hz": {"mean": np.mean(fr), "std": np.std(fr),
                           "min": np.min(fr), "max": np.max(fr)},
        "burst_rate_per_min": {"mean": np.mean(br), "std": np.std(br),
                                "min": np.min(br), "max": np.max(br)},
    }
    if dur:
        summary["burst_duration_ms"] = {"mean": np.mean(dur), "std": np.std(dur),
                                         "min": np.min(dur), "max": np.max(dur)}

    print(f"\n  {label}: {len(records)} recordings")
    print(f"    Units:      {min(units)} - {max(units)}")
    print(f"    FR (Hz):    {np.mean(fr):.2f} +/- {np.std(fr):.2f}  [{np.min(fr):.2f} - {np.max(fr):.2f}]")
    print(f"    Burst/min:  {np.mean(br):.1f} +/- {np.std(br):.1f}  [{np.min(br):.1f} - {np.max(br):.1f}]")
    if dur:
        print(f"    Burst dur:  {np.mean(dur):.0f} +/- {np.std(dur):.0f} ms  [{np.min(dur):.0f} - {np.max(dur):.0f}]")
    return summary


def main():
    t0 = time.time()
    print("=" * 78)
    print("  BL-1 Multi-Dataset Analysis")
    print("=" * 78)

    all_summaries = {}

    # -----------------------------------------------------------------------
    # 1. DANDI 001611 — sample across subjects (10 files per subject)
    # -----------------------------------------------------------------------
    print("\n[1] DANDI 001611: Rat cortical HD-MEA")
    dandi_dir = DATA_DIR / "dandi_001611_rat_cortical" / "001611"
    dandi_records = []
    if dandi_dir.exists():
        subjects = sorted([d for d in dandi_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
        print(f"    Subjects: {len(subjects)}")
        for subj in subjects:
            nwb_files = sorted(subj.glob("*.nwb"))
            # Sample up to 5 files per subject for speed
            sample = nwb_files[:5]
            for f in sample:
                try:
                    stats = process_nwb(f)
                    if stats:
                        stats["subject"] = subj.name
                        stats["filename"] = f.name
                        dandi_records.append(stats)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write("x")
                    sys.stdout.flush()
        print()
        all_summaries["dandi_001611"] = summarize(dandi_records, "DANDI 001611 (rat cortical)")
    else:
        print("    Not found")

    # -----------------------------------------------------------------------
    # 2. Sharf 2022 — ALL organoid files (33 total)
    # -----------------------------------------------------------------------
    print("\n[2] Sharf 2022: Human brain organoid HD-MEA")
    sharf_dir = DATA_DIR / "zenodo_sharf_2022"
    sharf_records = []
    sharf_dev = []
    sharf_drug = []
    sharf_baseline = []

    if sharf_dir.exists():
        h5_files = sorted(sharf_dir.glob("*.h5"))
        print(f"    Files: {len(h5_files)}")
        for f in h5_files:
            try:
                stats = process_maxwell(f)
                if stats:
                    stats["filename"] = f.name
                    sharf_records.append(stats)
                    # Categorize
                    if f.name.startswith("Development_"):
                        sharf_dev.append(stats)
                    elif f.name.startswith("Drug_"):
                        sharf_drug.append(stats)
                    else:
                        sharf_baseline.append(stats)
                    sys.stdout.write(".")
                    sys.stdout.flush()
            except Exception as e:
                sys.stdout.write("x")
                sys.stdout.flush()
        print()
        all_summaries["sharf_2022_all"] = summarize(sharf_records, "Sharf 2022 (all)")
        if sharf_baseline:
            all_summaries["sharf_2022_baseline"] = summarize(sharf_baseline, "Sharf 2022 (7-month baseline)")
        if sharf_dev:
            all_summaries["sharf_2022_development"] = summarize(sharf_dev, "Sharf 2022 (development series)")
        if sharf_drug:
            all_summaries["sharf_2022_drug"] = summarize(sharf_drug, "Sharf 2022 (drug dose-response)")

    # -----------------------------------------------------------------------
    # 3. Print comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Cross-Dataset Comparison")
    print("=" * 78)
    print(f"\n  {'Dataset':<35s} {'N':>4s} {'FR (Hz)':>12s} {'Burst/min':>12s} {'Units':>10s}")
    print(f"  {'-'*35} {'-'*4} {'-'*12} {'-'*12} {'-'*10}")

    for name, s in all_summaries.items():
        if not s:
            continue
        fr = s["firing_rate_hz"]
        br = s["burst_rate_per_min"]
        u = s["n_units_range"]
        print(f"  {name:<35s} {s['n_recordings']:4d} "
              f"{fr['mean']:5.1f}+/-{fr['std']:4.1f} "
              f"{br['mean']:5.1f}+/-{br['std']:4.1f} "
              f"{u[0]:4d}-{u[1]:4d}")

    # Wagenaar reference
    w = DATASETS["wagenaar_2006"]
    print(f"  {'Wagenaar 2006 (reference)':<35s} {'59':>4s} "
          f"{'0.1-5.0':>12s} {'0.2-20':>12s} {'60ch':>10s}")

    # -----------------------------------------------------------------------
    # 4. Save detailed results
    # -----------------------------------------------------------------------
    # Convert numpy to native Python for JSON
    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    out = {
        "summaries": to_native(all_summaries),
        "dandi_records": to_native(dandi_records),
        "sharf_records": to_native(sharf_records),
    }
    out_path = RESULTS_DIR / "dataset_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Results saved to: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
