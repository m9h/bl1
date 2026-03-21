#!/usr/bin/env python3
"""Submit BL-1 simulation jobs to the Neurosciences Gateway (NSG).

NSG provides free GPU compute at SDSC Expanse for neuroscience research.
This script packages a BL-1 simulation as a job and submits it via the
CIPRES REST API.

Setup:
    1. Create an account at https://nsgr.sdsc.edu:8443/restusers
    2. Create an application (Developer → Application Management → Create New)
       - Select DIRECT authentication type
       - Save your Application ID
    3. Set environment variables:
       export NSG_USERNAME=your_username
       export NSG_PASSWORD=your_password
       export NSG_APPKEY=your_application_id

Usage:
    python scripts/nsg_submit.py --list-tools          # show available tools
    python scripts/nsg_submit.py --validate             # validate without submitting
    python scripts/nsg_submit.py --submit               # submit job
    python scripts/nsg_submit.py --status JOB_HANDLE    # check job status
    python scripts/nsg_submit.py --list-jobs             # list all jobs
    python scripts/nsg_submit.py --download JOB_HANDLE   # download results

Tools of interest:
    GPU_PY_EXPANSE      - Python on Expanse GPUs (custom scripts)
    PYTORCH_PY_EXPANSE  - PyTorch Python on Expanse GPUs
    PY_EXPANSE          - Python in Singularity on Expanse (CPU)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# NSG connection
# ---------------------------------------------------------------------------

NSG_BASE_URL = "https://nsgr.sdsc.edu:8443/cipresrest/v1"

# Default tool — GPU Python on Expanse
DEFAULT_TOOL = "GPU_PY_EXPANSE"


def get_client():
    """Create an authenticated CIPRES client from environment variables."""
    try:
        from python_cipres.client import Client
    except ImportError:
        print("ERROR: python_cipres not installed. Run: pip install python_cipres")
        sys.exit(1)

    username = os.environ.get("NSG_USERNAME")
    password = os.environ.get("NSG_PASSWORD")
    appkey = os.environ.get("NSG_APPKEY")

    if not all([username, password, appkey]):
        print("ERROR: Set NSG_USERNAME, NSG_PASSWORD, and NSG_APPKEY environment variables.")
        print("       Register at https://nsgr.sdsc.edu:8443/restusers")
        sys.exit(1)

    return Client(
        appname="bl1_simulator",
        appID=appkey,
        username=username,
        password=password,
        baseUrl=NSG_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Job packaging
# ---------------------------------------------------------------------------

# The Python script that will run on Expanse GPUs
JOB_SCRIPT = '''\
#!/usr/bin/env python3
"""BL-1 validation job for NSG Expanse GPU."""

import subprocess
import sys

# Install BL-1 dependencies into the job environment
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "jax[cuda12]", "equinox", "matplotlib", "h5py", "pyyaml", "numpy",
])

import time
import jax
import jax.numpy as jnp
import numpy as np

print("=" * 72)
print("BL-1 NSG Validation Job")
print("=" * 72)
print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print()

# Import bl1 (installed from the uploaded zip)
sys.path.insert(0, ".")
from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState, create_synapse_state,
    ampa_step, gaba_a_step, nmda_step, compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step
from bl1.validation.comparison import compute_culture_statistics
from bl1.validation.datasets import compare_statistics

# --- Wagenaar-calibrated parameters ---
CONFIGS = [
    {"n_neurons": 5000,   "duration_ms": 60_000.0,  "label": "5K/60s"},
    {"n_neurons": 10000,  "duration_ms": 60_000.0,  "label": "10K/60s"},
    {"n_neurons": 50000,  "duration_ms": 10_000.0,  "label": "50K/10s"},
    {"n_neurons": 100000, "duration_ms": 5_000.0,   "label": "100K/5s"},
]

DT = 0.5
G_EXC, G_INH = 0.12, 0.36
NMDA_RATIO = 0.37
U_EXC, TAU_REC = 0.30, 800.0
SEED = 42

for cfg in CONFIGS:
    N = cfg["n_neurons"]
    DUR = cfg["duration_ms"]
    print(f"\\n{'='*72}")
    print(f"  {cfg['label']}: {N} neurons, {DUR/1000:.0f}s")
    print(f"{'='*72}")

    key = jax.random.PRNGKey(SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    t0 = time.perf_counter()
    positions = place_neurons(k1, N, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, N)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=G_EXC, g_inh=G_INH,
    )
    syn = create_synapse_state(N)
    W_ampa = W_exc * (1.0 - NMDA_RATIO)
    W_nmda = W_exc * NMDA_RATIO

    U = jnp.where(is_exc, U_EXC, 0.04)
    tau_rec = jnp.where(is_exc, TAU_REC, 100.0)
    tau_fac = jnp.where(is_exc, 0.001, 1000.0)
    stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
    stp_state = init_stp_state(N, stp_params)

    n_steps = int(DUR / DT)
    I_noise = 1.0 + 3.0 * jax.random.normal(k4, (n_steps, N))
    build_time = time.perf_counter() - t0
    print(f"  Build: {build_time:.1f}s")

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
    sim_time = time.perf_counter() - t0
    rt_factor = (DUR / 1000.0) / sim_time

    raster = np.asarray(spikes)
    fr = float(raster.sum()) / (N * DUR / 1000.0)

    print(f"  Sim:   {sim_time:.1f}s  ({rt_factor:.1f}x realtime)")
    print(f"  FR:    {fr:.2f} Hz")
    print(f"  Spikes: {int(raster.sum()):,}")

    # Bio validation for 5K run
    if N == 5000:
        stats = compute_culture_statistics(raster, dt_ms=DT, burst_threshold_std=1.5)
        results = compare_statistics(stats, "wagenaar_2006")
        n_pass = sum(1 for r in results.values() if r["in_range"] is True)
        n_total = sum(1 for r in results.values() if r["in_range"] is not None)
        print(f"  Wagenaar validation: {n_pass}/{n_total} metrics PASS")
        for k, r in sorted(results.items()):
            if r["in_range"] is not None:
                status = "PASS" if r["in_range"] else "FAIL"
                print(f"    {k}: {r['sim_value']:.4g} [{status}]")

    # Save results
    np.savez_compressed(f"results_{N}.npz", spikes=raster[:min(10000, n_steps)],
                        fr=fr, sim_time=sim_time, rt_factor=rt_factor)

print("\\n" + "="*72)
print("All benchmarks complete.")
print("="*72)
'''


def package_job(output_dir: Path) -> Path:
    """Package BL-1 source + job script into a zip for upload."""
    import zipfile

    project_root = Path(__file__).resolve().parent.parent
    zip_path = output_dir / "bl1_job.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add the job script
        zf.writestr("run_job.py", JOB_SCRIPT)

        # Add bl1 source package
        src_dir = project_root / "src" / "bl1"
        for f in src_dir.rglob("*.py"):
            arcname = str(f.relative_to(project_root / "src"))
            zf.writestr(arcname, f.read_text())

    print(f"  Job packaged: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Submit BL-1 jobs to NSG (SDSC Expanse GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-tools", action="store_true",
                       help="List available NSG tools")
    group.add_argument("--list-jobs", action="store_true",
                       help="List your submitted jobs")
    group.add_argument("--validate", action="store_true",
                       help="Validate job without submitting")
    group.add_argument("--submit", action="store_true",
                       help="Submit job to NSG")
    group.add_argument("--status", metavar="JOB_HANDLE",
                       help="Check status of a job")
    group.add_argument("--download", metavar="JOB_HANDLE",
                       help="Download results of a completed job")
    group.add_argument("--delete", metavar="JOB_HANDLE",
                       help="Delete/cancel a job")

    parser.add_argument("--tool", default=DEFAULT_TOOL,
                        help=f"NSG tool ID (default: {DEFAULT_TOOL})")
    parser.add_argument("--runtime", type=float, default=1.0,
                        help="Max walltime in hours (default: 1.0)")
    parser.add_argument("--output-dir", default="results/nsg",
                        help="Local directory for results")

    args = parser.parse_args()

    if args.list_tools:
        # Use curl since we just need to see what's available
        import subprocess
        username = os.environ.get("NSG_USERNAME", "")
        password = os.environ.get("NSG_PASSWORD", "")
        appkey = os.environ.get("NSG_APPKEY", "")
        result = subprocess.run([
            "curl", "-s", "-u", f"{username}:{password}",
            "-H", f"cipres-appkey:{appkey}",
            f"{NSG_BASE_URL}/tool",
        ], capture_output=True, text=True)
        # Parse tool names from XML
        import re
        tools = re.findall(r"<toolId>([^<]+)</toolId>", result.stdout)
        print("Available NSG tools:")
        for t in sorted(tools):
            print(f"  {t}")
        return

    client = get_client()

    if args.list_jobs:
        jobs = client.listJobs()
        if not jobs:
            print("No jobs found.")
        else:
            print(f"{'Handle':<40s} {'Stage':<15s} {'Terminal':<10s}")
            print("-" * 65)
            for j in jobs:
                j.update()
                print(f"{j.jobHandle:<40s} {j.jobStage:<15s} {str(j.terminalStage):<10s}")
        return

    if args.status:
        from python_cipres.client import JobStatus
        job = JobStatus(client, jobUrl=f"{NSG_BASE_URL}/job/{os.environ['NSG_USERNAME']}/{args.status}")
        job.update()
        job.show(messages=True)
        return

    if args.download:
        from python_cipres.client import JobStatus
        job = JobStatus(client, jobUrl=f"{NSG_BASE_URL}/job/{os.environ['NSG_USERNAME']}/{args.download}")
        job.update()
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading results to {out_dir}/...")
        job.downloadResults(directory=str(out_dir))
        print("Done.")
        return

    if args.delete:
        from python_cipres.client import JobStatus
        job = JobStatus(client, jobUrl=f"{NSG_BASE_URL}/job/{os.environ['NSG_USERNAME']}/{args.delete}")
        job.delete()
        print(f"Job {args.delete} deleted.")
        return

    # --- Submit or validate ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = package_job(tmpdir)

        vParams = {
            "toolId": args.tool,
            "runtime_": str(args.runtime),
        }
        inputParams = {
            "infile_": str(zip_path),
        }
        metadata = {
            "statusEmail": "true",
        }

        if args.validate:
            print("Validating job...")
            try:
                job = client.validateJob(vParams, inputParams, metadata)
                print(f"  Valid! Command: {job.commandline}")
            except Exception as e:
                print(f"  Validation failed: {e}")
            return

        if args.submit:
            print(f"Submitting job to NSG ({args.tool}, {args.runtime}h walltime)...")
            job = client.submitJob(vParams, inputParams, metadata)
            print(f"  Submitted! Job handle: {job.jobHandle}")
            print(f"  Status URL: {job.jobUrl}")
            print(f"\n  Check status:   python scripts/nsg_submit.py --status {job.jobHandle}")
            print(f"  Download:       python scripts/nsg_submit.py --download {job.jobHandle}")
            print(f"  Delete/cancel:  python scripts/nsg_submit.py --delete {job.jobHandle}")
            return


if __name__ == "__main__":
    main()
