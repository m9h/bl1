#!/usr/bin/env python3
"""M0 connectivity smoke test for the FinalSpark Neuroplatform.

Run this *inside the FinalSpark-provided environment* (where the ``neuroplatform``
package is available) with your account token in the environment::

    export FINALSPARK_TOKEN=...        # provided by FinalSpark
    python experiments/finalspark_smoke.py --fs-name fs300 --minutes 5

What it does (no stimulation — read-only and safe):

1. Connects via ``Experiment(token)`` and lists the electrodes available to you.
2. Pulls the last ``--minutes`` of spike counts and spike events from the Spike DB.
3. Prints a per-electrode summary and saves two plots:
   - ``finalspark_raster.png``   — spike events (time vs electrode)
   - ``finalspark_counts.png``   — per-electrode total spike counts

This validates auth, electrode discovery, and the data path before any of the
stimulation / closed-loop milestones (M1+). It does not import or depend on the
BL-1 package, so it runs in a minimal FinalSpark environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone


def _require_neuroplatform():
    """Import the FinalSpark SDK with a clear message if it is unavailable."""
    try:
        import neuroplatform  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: the 'neuroplatform' package is not importable.\n"
            "This script must run inside the FinalSpark-provided environment "
            "(the package is not on PyPI). See "
            "https://finalspark-np.github.io/np-docs/welcome.html"
        )
    return neuroplatform


def main() -> int:
    p = argparse.ArgumentParser(description="FinalSpark Neuroplatform M0 smoke test")
    p.add_argument(
        "--fs-name",
        default=os.environ.get("FINALSPARK_FS_NAME"),
        help="Experiment identifier (e.g. 'fs300'); or set FINALSPARK_FS_NAME",
    )
    p.add_argument("--minutes", type=float, default=5.0, help="Lookback window (docs recommend ~5)")
    p.add_argument("--outdir", default=".", help="Directory for the saved plots")
    args = p.parse_args()

    token = os.environ.get("FINALSPARK_TOKEN")
    if not token:
        sys.exit("ERROR: set FINALSPARK_TOKEN in the environment (do not hard-code it).")
    if not args.fs_name:
        sys.exit("ERROR: pass --fs-name or set FINALSPARK_FS_NAME (e.g. 'fs300').")

    np_sdk = _require_neuroplatform()
    from neuroplatform import Database, Experiment

    # --- connect ----------------------------------------------------------------
    exp = Experiment(token)
    exp.start()
    try:
        electrodes = list(getattr(exp, "electrodes", []) or [])
        print(f"Connected. {len(electrodes)} electrode(s) available: {electrodes}")

        # --- pull recent activity (read-only) -----------------------------------
        stop = datetime.now(timezone.utc)
        start = stop - timedelta(minutes=args.minutes)
        db = Database()

        counts = db.get_spike_count(start, stop, args.fs_name)
        events = db.get_spike_event(start, stop, args.fs_name)
        print(f"get_spike_count -> {getattr(counts, 'shape', '?')} rows")
        print(f"get_spike_event -> {getattr(events, 'shape', '?')} rows over {args.minutes} min")
    finally:
        exp.stop()
        print("Experiment stopped (clean exit).")

    _plot(events, counts, args.outdir)
    print(f"Smoke test OK. SDK version: {getattr(np_sdk, '__version__', 'unknown')}")
    return 0


def _plot(events, counts, outdir: str) -> None:
    """Best-effort plots; never fail the smoke test on a plotting issue."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # Raster: spike events (Time vs channel)
    try:
        if events is not None and len(events) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(events["Time"], events["channel"], s=2, c="k")
            ax.set(xlabel="Time", ylabel="Electrode", title="FinalSpark spike events")
            fig.tight_layout()
            fig.savefig(f"{outdir}/finalspark_raster.png", dpi=120)
            plt.close(fig)
            print(f"Wrote {outdir}/finalspark_raster.png")
    except Exception as e:  # noqa: BLE001 — diagnostic script, surface and continue
        print(f"raster plot skipped: {e}")

    # Per-electrode total counts
    try:
        if counts is not None and len(counts) > 0:
            per_electrode = counts.drop(columns=["Time"], errors="ignore").sum(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            per_electrode.plot.bar(ax=ax)
            ax.set(xlabel="Electrode", ylabel="Spike count", title="FinalSpark per-electrode counts")
            fig.tight_layout()
            fig.savefig(f"{outdir}/finalspark_counts.png", dpi=120)
            plt.close(fig)
            print(f"Wrote {outdir}/finalspark_counts.png")
    except Exception as e:  # noqa: BLE001
        print(f"counts plot skipped: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
