#!/usr/bin/env python3
"""Train BL-1 against all 33 Sharf 2022 organoid recordings.

Results saved to /data/datasets/bl1/results/ with organized naming:
  sharf_2022/{condition}/{recording_id}/
    trained_weights.npz
    loss_history.json

Experiment tracking via trackio (Gradio dashboard viewable from any machine).

Usage:
    python scripts/train_all_sharf.py                              # full run
    python scripts/train_all_sharf.py --n-neurons 1000 --n-epochs 30  # fast test
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import trackio

SHARF_DIR = Path("/data/datasets/bl1/zenodo_sharf_2022")
RESULTS_BASE = Path("/data/datasets/bl1/results/sharf_2022")


def classify_recording(filename: str) -> tuple[str, str]:
    """Classify a Sharf filename into (condition, recording_id).

    Returns:
        (condition, recording_id) where condition is one of:
        'baseline', 'development', 'drug_dose_response'
    """
    name = filename.replace(".raw.h5", "").replace(".h5", "")

    if name.startswith("Development_"):
        # Development_2953_5month -> development/2953_5month
        m = re.match(r"Development_(\d+)_(\w+)", name)
        if m:
            return "development", f"{m.group(1)}_{m.group(2)}"
        return "development", name.replace("Development_", "")

    elif name.startswith("Drug_"):
        # Drug_2950_30uM -> drug_dose_response/2950_30uM
        m = re.match(r"Drug_(\d+)_(.*)", name)
        if m:
            return "drug_dose_response", f"{m.group(1)}_{m.group(2)}"
        return "drug_dose_response", name.replace("Drug_", "")

    else:
        # 7month_2950 -> baseline/7month_2950
        return "baseline", name


def extract_targets(filepath):
    """Load a Maxwell HDF5 and extract targets from the activity window."""
    from bl1.validation.loaders import load_maxwell_h5, compute_recording_statistics

    data = load_maxwell_h5(str(filepath))
    if data["n_units"] < 5:
        return None

    active = [st for st in data["spike_times"] if len(st) > 0]
    if not active:
        return None
    t_min = float(min(st.min() for st in active))
    t_max = float(max(st.max() for st in active))
    dur = t_max - t_min
    if dur < 5:
        return None

    use_dur = min(dur, 120.0)
    trimmed = {
        "spike_times": [st[(st >= t_min) & (st <= t_min + use_dur)] - t_min
                        for st in data["spike_times"]],
        "duration_s": use_dur,
        "n_units": data["n_units"],
    }
    stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
    return {
        "target_fr": stats["mean_firing_rate_hz"],
        "target_burst": stats["burst_rate_per_min"],
        "n_units": data["n_units"],
        "duration_s": use_dur,
        "actual_duration_s": dur,
    }


def train_one(filepath, n_neurons, n_epochs, output_dir, run_tracker):
    """Train BL-1 against one recording."""
    from bl1.training.trainer import TrainingConfig, train_weights

    name = filepath.stem
    condition, rec_id = classify_recording(filepath.name)

    targets = extract_targets(filepath)
    if targets is None:
        return {"name": name, "condition": condition, "rec_id": rec_id,
                "status": "SKIP", "reason": "no spikes"}

    # Organized output path
    rec_dir = output_dir / condition / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        n_neurons=n_neurons,
        n_epochs=n_epochs,
        sim_duration_ms=500.0,
        dt=0.5,
        learning_rate=1e-4,
        target_firing_rate_hz=targets["target_fr"],
        target_burst_rate_per_min=targets["target_burst"],
        w_firing_rate=1.0,
        w_burst_rate=0.05,
        w_synchrony=0.5,
        w_weight_reg=0.01,
        surrogate_beta=5.0,
        I_noise_amplitude=2.0,
        init_weight_scale=0.1,
    )

    t0 = time.time()
    result = train_weights(config, tracker=run_tracker)
    wall = time.time() - t0

    # Save weights
    weight_path = rec_dir / "trained_weights.npz"
    np.savez_compressed(
        weight_path,
        W_exc=np.array(result.W_exc),
        W_inh=np.array(result.W_inh),
    )

    # Save loss history
    with open(rec_dir / "loss_history.json", "w") as f:
        json.dump(result.loss_history, f, indent=2, default=str)

    # Save config + targets metadata
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump({
            "source_file": str(filepath),
            "condition": condition,
            "recording_id": rec_id,
            "targets": targets,
            "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        }, f, indent=2, default=str)

    final = result.loss_history[-1]
    initial = result.loss_history[0]

    return {
        "name": name,
        "condition": condition,
        "rec_id": rec_id,
        "status": "OK",
        "target_fr_hz": targets["target_fr"],
        "target_burst_per_min": targets["target_burst"],
        "real_n_units": targets["n_units"],
        "real_duration_s": targets["duration_s"],
        "initial_fr_hz": initial.get("mean_fr_hz", float("nan")),
        "final_fr_hz": final.get("mean_fr_hz", float("nan")),
        "initial_loss": initial.get("total", float("nan")),
        "final_loss": final.get("total", float("nan")),
        "wall_time_s": wall,
        "output_dir": str(rec_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train BL-1 on all Sharf 2022 recordings")
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_BASE))
    parser.add_argument("--no-trackio", action="store_true", help="Disable trackio logging")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(SHARF_DIR.glob("*.h5"))
    if not h5_files:
        print(f"ERROR: No .h5 files in {SHARF_DIR}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("  BL-1 Full Dataset Training — Sharf 2022 Organoid")
    print("=" * 78)
    print(f"  Recordings:  {len(h5_files)}")
    print(f"  Neurons:     {args.n_neurons}")
    print(f"  Epochs:      {args.n_epochs}")
    print(f"  Output:      {output_dir}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  Trackio:     {'enabled' if not args.no_trackio else 'disabled'}")
    print("=" * 78)

    all_results = []
    t_total = time.time()

    for i, f in enumerate(h5_files):
        condition, rec_id = classify_recording(f.name)
        run_name = f"sharf/{condition}/{rec_id}"
        print(f"\n[{i+1}/{len(h5_files)}] {run_name}")

        # Initialize trackio run for this recording
        run_tracker = None
        if not args.no_trackio:
            try:
                run_tracker = trackio.init(
                    project="bl1-sharf-2022",
                    name=run_name,
                    dir=str(output_dir / "trackio"),
                    config={
                        "dataset": "sharf_2022",
                        "condition": condition,
                        "recording_id": rec_id,
                        "source_file": f.name,
                        "n_neurons": args.n_neurons,
                        "n_epochs": args.n_epochs,
                    },
                )
            except Exception as e:
                print(f"  trackio init failed: {e}")

        try:
            r = train_one(f, args.n_neurons, args.n_epochs, output_dir, run_tracker)
            all_results.append(r)
            if r["status"] == "OK":
                print(f"  target FR={r['target_fr_hz']:.3f} Hz → sim FR={r['final_fr_hz']:.2f} Hz "
                      f"| loss {r['initial_loss']:.3f}→{r['final_loss']:.3f} "
                      f"| {r['wall_time_s']:.0f}s")
            else:
                print(f"  SKIPPED: {r.get('reason', 'unknown')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"name": f.stem, "condition": condition,
                                "rec_id": rec_id, "status": "ERROR", "reason": str(e)})
        finally:
            if run_tracker is not None:
                try:
                    trackio.finish()
                except Exception:
                    pass

    total_wall = time.time() - t_total

    # --- Save aggregate results ---
    summary_path = output_dir / f"summary_{timestamp}.json"

    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    with open(summary_path, "w") as f:
        json.dump([{k: to_native(v) for k, v in r.items()} for r in all_results],
                  f, indent=2, default=str)

    # CSV
    csv_path = output_dir / f"summary_{timestamp}.csv"
    ok_results = [r for r in all_results if r["status"] == "OK"]
    if ok_results:
        fields = [k for k in ok_results[0].keys() if k != "output_dir"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ok_results)

    # --- Summary ---
    print("\n" + "=" * 78)
    print("  Training Summary")
    print("=" * 78)
    n_ok = sum(1 for r in all_results if r["status"] == "OK")
    n_skip = sum(1 for r in all_results if r["status"] == "SKIP")
    n_err = sum(1 for r in all_results if r["status"] == "ERROR")
    print(f"  Completed: {n_ok}  Skipped: {n_skip}  Errors: {n_err}")

    if ok_results:
        # Per-condition breakdown
        conditions = sorted(set(r["condition"] for r in ok_results))
        for cond in conditions:
            cond_results = [r for r in ok_results if r["condition"] == cond]
            frs = [r["final_fr_hz"] for r in cond_results]
            tgts = [r["target_fr_hz"] for r in cond_results]
            print(f"\n  {cond} ({len(cond_results)} recordings):")
            print(f"    Target FR:   {min(tgts):.3f} - {max(tgts):.3f} Hz")
            print(f"    Achieved FR: {min(frs):.3f} - {max(frs):.3f} Hz")

    print(f"\n  Total time: {total_wall/60:.1f} min")
    print(f"  Results:    {output_dir}")
    print(f"  Summary:    {summary_path}")
    print(f"  CSV:        {csv_path}")
    if not args.no_trackio:
        print(f"  Trackio:    {output_dir / 'trackio'}")
        print(f"  Dashboard:  trackio dashboard --logdir {output_dir / 'trackio'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
