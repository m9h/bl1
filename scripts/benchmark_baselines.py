#!/usr/bin/env python3
"""BL-1 Benchmark Baselines

Runs BL-1 simulations at various network scales and saves results to a
Markdown table suitable for committing to the repo.

For each neuron count the script:
    1. Builds the network (placement + population + connectivity)
    2. Runs a JIT warmup (100 steps)
    3. Times the full simulation (default 5 s simulated time at dt=0.5 ms)
    4. Records build time, JIT time, wall time, realtime factor,
       spike count, mean firing rate, and peak GPU memory (if available)

Usage:
    python scripts/benchmark_baselines.py
    python scripts/benchmark_baselines.py --sizes 1000 5000 10000
    python scripts/benchmark_baselines.py --duration-ms 2000 --output-dir results/
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import socket
import time

import jax
import jax.numpy as jnp

from bl1.core.integrator import simulate
from bl1.core.izhikevich import create_population
from bl1.core.synapses import create_synapse_state
from bl1.network.topology import build_connectivity, place_neurons

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
DEFAULT_DURATION_MS = 5000.0
DEFAULT_DT = 0.5
WARMUP_STEPS = 100


def _gpu_name() -> str:
    """Return the GPU device name, or 'cpu' if no GPU is available."""
    devices = jax.devices()
    for d in devices:
        if d.platform == "gpu":
            return str(d.device_kind)
    return "cpu"


def _cuda_version() -> str:
    """Best-effort retrieval of the CUDA version visible to JAX."""
    try:
        # jaxlib exposes CUDA version on GPU builds
        import jaxlib  # noqa: F401

        ver = getattr(jaxlib, "cuda_version", None)
        if ver is not None:
            return ver
    except Exception:
        pass
    # Fall back to environment / nvidia-smi
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return f"CUDA_HOME={cuda_home}"
    return "N/A"


def _peak_gpu_memory_mb() -> str:
    """Return peak GPU memory in MB if JAX exposes it, else 'N/A'."""
    try:
        backend = jax.lib.xla_bridge.get_backend()
        for device in backend.devices():
            if device.platform == "gpu":
                stats = device.memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    return f"{stats['peak_bytes_in_use'] / 1e6:.0f}"
    except Exception:
        pass
    return "N/A"


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def run_single(
    n_neurons: int,
    duration_ms: float,
    dt: float,
) -> dict:
    """Run a full benchmark for one network size and return a results dict."""
    n_steps = int(duration_ms / dt)
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k_drive = jax.random.split(key, 4)

    # --- Network build ---------------------------------------------------
    t0 = time.perf_counter()
    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _delays = build_connectivity(
        k3,
        positions,
        is_exc,
        lambda_um=200.0,
        p_max=0.21,
        g_exc=0.05,
        g_inh=0.20,
    )
    # Ensure arrays are materialised before stopping the clock
    jax.block_until_ready((W_exc, W_inh))
    build_time = time.perf_counter() - t0

    syn_state = create_synapse_state(n_neurons)

    # Background noise current
    I_external = jax.random.normal(k_drive, (n_steps, n_neurons)) * 3.0

    # --- JIT warmup (compile) -------------------------------------------
    warmup_I = I_external[:WARMUP_STEPS]
    t0 = time.perf_counter()
    warmup_result = simulate(
        params,
        state,
        syn_state,
        None,  # stdp_state
        W_exc,
        W_inh,
        warmup_I,
        dt=dt,
    )
    warmup_result.spike_history.block_until_ready()
    jit_time = time.perf_counter() - t0

    # --- Timed simulation -----------------------------------------------
    t0 = time.perf_counter()
    result = simulate(
        params,
        state,
        syn_state,
        None,  # stdp_state
        W_exc,
        W_inh,
        I_external,
        dt=dt,
    )
    result.spike_history.block_until_ready()
    sim_wall = time.perf_counter() - t0

    # --- Metrics ---------------------------------------------------------
    total_spikes = int(jnp.sum(result.spike_history))
    sim_ms = n_steps * dt
    realtime_factor = sim_ms / (sim_wall * 1000.0) if sim_wall > 0 else float("inf")
    mean_rate = total_spikes / (n_neurons * sim_ms / 1000.0) if sim_ms > 0 else 0.0
    peak_mem = _peak_gpu_memory_mb()

    # --- NNZ info --------------------------------------------------------
    nnz_exc = int(W_exc.data.shape[0])
    nnz_inh = int(W_inh.data.shape[0])

    return {
        "n_neurons": n_neurons,
        "build_s": build_time,
        "jit_s": jit_time,
        "sim_s": sim_wall,
        "realtime_x": realtime_factor,
        "total_spikes": total_spikes,
        "mean_rate_hz": mean_rate,
        "peak_gpu_mb": peak_mem,
        "nnz_exc": nnz_exc,
        "nnz_inh": nnz_inh,
        "n_steps": n_steps,
        "duration_ms": sim_ms,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COLUMNS = [
    ("Neurons", "n_neurons", ">8", ","),
    ("NNZ(E+I)", "_nnz_total", ">12", ","),
    ("Build(s)", "build_s", ">9", ".2f"),
    ("JIT(s)", "jit_s", ">8", ".2f"),
    ("Sim(s)", "sim_s", ">8", ".2f"),
    ("RT Factor", "realtime_x", ">10", ".2f"),
    ("Spikes", "total_spikes", ">10", ","),
    ("Rate(Hz)", "mean_rate_hz", ">9", ".1f"),
    ("GPU MB", "peak_gpu_mb", ">8", "s"),
]


def _format_row(r: dict) -> str:
    """Format a single result dict as a pipe-delimited Markdown row."""
    r["_nnz_total"] = r["nnz_exc"] + r["nnz_inh"]
    parts = []
    for _header, key, align, fmt in _COLUMNS:
        val = r[key]
        if fmt == "s":
            cell = f"{val:{align}}"
        elif fmt == ",":
            cell = f"{val:{align}{fmt}}"
        else:
            cell = f"{val:{align}{fmt}}"
        parts.append(cell)
    return "| " + " | ".join(parts) + " |"


def _header_row() -> str:
    parts = []
    for header, _key, align, _fmt in _COLUMNS:
        width = int(align.lstrip("<>^"))
        parts.append(f"{header:>{width}}")
    return "| " + " | ".join(parts) + " |"


def _separator_row() -> str:
    parts = []
    for _header, _key, align, _fmt in _COLUMNS:
        width = int(align.lstrip("<>^"))
        parts.append("-" * width)
    return "|-" + "-|-".join(parts) + "-|"


def format_table(results: list[dict]) -> str:
    """Build a complete Markdown table from a list of result dicts."""
    lines = [_header_row(), _separator_row()]
    for r in results:
        lines.append(_format_row(r))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report(results: list[dict], duration_ms: float, dt: float) -> str:
    """Build the full Markdown report string."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    gpu = _gpu_name()
    jax_ver = jax.__version__
    cuda_ver = _cuda_version()
    backend = jax.default_backend()
    hostname = socket.gethostname()
    n_steps = int(duration_ms / dt)

    lines = [
        f"# BL-1 Benchmark Baselines",
        "",
        f"- **Date**: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Host**: {hostname}",
        f"- **GPU**: {gpu}",
        f"- **JAX version**: {jax_ver}",
        f"- **JAX backend**: {backend}",
        f"- **CUDA**: {cuda_ver}",
        f"- **Sim duration**: {duration_ms:.0f} ms ({n_steps:,} steps at dt={dt} ms)",
        f"- **JIT warmup**: {WARMUP_STEPS} steps",
        "",
        "## Results",
        "",
        format_table(results),
        "",
        "## System Info",
        "",
        f"- **OS**: {platform.system()} {platform.release()}",
        f"- **Python**: {platform.python_version()}",
        f"- **CPU**: {platform.processor() or platform.machine()}",
        f"- **JAX devices**: {jax.devices()}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run BL-1 benchmark baselines across network scales"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Neuron counts to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=DEFAULT_DURATION_MS,
        help="Simulated time in ms (default: %(default)s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Integration timestep in ms (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for the Markdown output file (default: %(default)s)",
    )
    args = parser.parse_args()

    sizes = sorted(args.sizes)
    duration_ms = args.duration_ms
    dt = args.dt
    n_steps = int(duration_ms / dt)

    # Print preamble
    print("=" * 72)
    print("BL-1 Benchmark Baselines")
    print("=" * 72)
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  JAX devices : {jax.devices()}")
    print(f"  GPU         : {_gpu_name()}")
    print(f"  Duration    : {duration_ms:.0f} ms = {n_steps:,} steps @ dt={dt} ms")
    print(f"  Sizes       : {sizes}")
    print(f"  Warmup      : {WARMUP_STEPS} steps")
    print("=" * 72)

    results: list[dict] = []

    for n in sizes:
        print(f"\n--- N = {n:,} neurons ---")
        try:
            r = run_single(n, duration_ms, dt)
            results.append(r)
            print(f"  Build     : {r['build_s']:.2f} s")
            print(f"  JIT       : {r['jit_s']:.2f} s")
            print(f"  Sim       : {r['sim_s']:.2f} s")
            print(f"  RT factor : {r['realtime_x']:.2f}x")
            print(f"  Spikes    : {r['total_spikes']:,}")
            print(f"  Rate      : {r['mean_rate_hz']:.1f} Hz")
            print(f"  NNZ (E/I) : {r['nnz_exc']:,} / {r['nnz_inh']:,}")
            print(f"  GPU mem   : {r['peak_gpu_mb']} MB")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({
                "n_neurons": n,
                "build_s": 0.0,
                "jit_s": 0.0,
                "sim_s": 0.0,
                "realtime_x": 0.0,
                "total_spikes": 0,
                "mean_rate_hz": 0.0,
                "peak_gpu_mb": "ERR",
                "nnz_exc": 0,
                "nnz_inh": 0,
                "n_steps": n_steps,
                "duration_ms": duration_ms,
            })

    # Print summary table to stdout
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(format_table(results))

    # Write Markdown report
    hostname = socket.gethostname()
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"benchmark_{hostname}_{date_str}.md")

    report = build_report(results, duration_ms, dt)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
