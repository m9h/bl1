#!/usr/bin/env python3
"""Long-duration stability validation for BL-1 cortical culture simulator.

Runs an extended simulation in chunks and monitors for:
- Memory leaks (GPU and CPU RSS growing over time)
- Numerical instability (NaN/Inf in neuron states or spike history)
- Performance degradation (wall time per chunk increasing)
- Biological plausibility (firing rate in realistic range)

Each chunk is 10 seconds of simulated time (20,000 steps at dt=0.5 ms).
Results are logged to stdout and saved to a CSV file.

Usage:
    python scripts/validate_stability.py
    python scripts/validate_stability.py --duration-hours 0.5 --n-neurons 5000
    python scripts/validate_stability.py --duration-hours 2 --output-dir /tmp/results
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bl1.core.integrator import simulate
from bl1.core.izhikevich import NeuronState, create_population
from bl1.core.synapses import SynapseState, create_synapse_state
from bl1.network.topology import build_connectivity, place_neurons
from bl1.plasticity.stp import STPParams, create_stp_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT_MS = 0.5
CHUNK_DURATION_MS = 10_000.0  # 10 seconds per chunk
STEPS_PER_CHUNK = int(CHUNK_DURATION_MS / DT_MS)  # 20,000 steps

# Network parameters (from wagenaar_calibrated.yaml)
SUBSTRATE_UM = (3000.0, 3000.0)
LAMBDA_UM = 200.0
P_MAX = 0.02
G_EXC = 0.50
G_INH = 2.00
BACKGROUND_MEAN = 1.0
BACKGROUND_STD = 3.0

# STP parameters matching the calibrated config
STP_EXC_TAU_REC = 3000.0  # 3 second recovery -- sets IBI timescale

# Pass/fail thresholds
MAX_MEMORY_GROWTH_FACTOR = 2.0
MAX_FIRING_RATE_HZ = 100.0
MIN_FIRING_RATE_HZ = 0.0  # fail if exactly 0 (dead network)
MAX_WALL_TIME_GROWTH_FACTOR = 3.0

# CSV column headers
CSV_HEADERS = [
    "chunk",
    "wall_time_s",
    "cumulative_sim_time_s",
    "total_spikes",
    "mean_firing_rate_hz",
    "v_min",
    "v_max",
    "v_has_nan",
    "v_has_inf",
    "spike_has_nan",
    "cpu_rss_mb",
    "gpu_memory_mb",
]


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def get_cpu_rss_mb() -> float:
    """Return current process RSS in megabytes (cross-platform via resource module)."""
    # ru_maxrss is in bytes on Linux, kilobytes on macOS
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)  # bytes -> MB on macOS
    else:
        return usage.ru_maxrss / 1024  # KB -> MB on Linux


def get_gpu_memory_mb() -> float | None:
    """Return GPU memory usage in MB if available, else None.

    Attempts to read from the first JAX device's memory_stats(). Falls
    back gracefully when running on CPU-only or when the backend does
    not expose memory statistics.
    """
    try:
        device = jax.local_devices()[0]
        if device.platform == "cpu":
            return None
        stats = device.memory_stats()
        if stats and "bytes_in_use" in stats:
            return stats["bytes_in_use"] / (1024 * 1024)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Network setup
# ---------------------------------------------------------------------------


def build_network(
    n_neurons: int,
    seed: int = 42,
) -> tuple:
    """Build the full network: neurons, synapses, weights, STP params.

    Returns
    -------
    params : IzhikevichParams
    init_state : NeuronState
    syn_state : SynapseState
    W_exc : BCOO
    W_inh : BCOO
    stp_params : STPParams
    is_excitatory : Array
    prng_key : Array (for noise generation)
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k_noise = jax.random.split(key, 4)

    # Place neurons on 2D substrate
    positions = place_neurons(k1, n_neurons, SUBSTRATE_UM)

    # Create neuron population (80/20 E/I)
    params, init_state, is_excitatory = create_population(k2, n_neurons)

    # Build distance-dependent connectivity
    W_exc, W_inh, _delays = build_connectivity(
        k3,
        positions,
        is_excitatory,
        lambda_um=LAMBDA_UM,
        p_max=P_MAX,
        g_exc=G_EXC,
        g_inh=G_INH,
    )

    # Create synapse state
    syn_state = create_synapse_state(n_neurons)

    # Create STP parameters with calibrated excitatory recovery
    stp_params = create_stp_params(n_neurons, is_excitatory)
    # Override tau_rec for excitatory neurons to match calibrated config
    tau_rec_calibrated = jnp.where(is_excitatory, STP_EXC_TAU_REC, stp_params.tau_rec)
    stp_params = STPParams(
        U=stp_params.U,
        tau_rec=tau_rec_calibrated,
        tau_fac=stp_params.tau_fac,
    )

    n_exc_syn = W_exc.data.shape[0] if hasattr(W_exc, "data") else 0
    n_inh_syn = W_inh.data.shape[0] if hasattr(W_inh, "data") else 0
    print(f"  Excitatory synapses: {n_exc_syn:,}")
    print(f"  Inhibitory synapses: {n_inh_syn:,}")

    return params, init_state, syn_state, W_exc, W_inh, stp_params, is_excitatory, k_noise


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def run_stability_test(
    duration_hours: float,
    n_neurons: int,
    output_dir: str,
    seed: int = 42,
) -> bool:
    """Run the long-duration stability test.

    Returns True if all checks pass, False otherwise.
    """
    total_sim_time_ms = duration_hours * 3600 * 1000  # hours -> ms
    n_chunks = int(total_sim_time_ms / CHUNK_DURATION_MS)
    if n_chunks < 1:
        n_chunks = 1

    print("=" * 72)
    print("BL-1 Long-Duration Stability Test")
    print("=" * 72)
    print(f"  Duration:       {duration_hours:.2f} hours ({n_chunks} chunks)")
    print(f"  Neurons:        {n_neurons:,}")
    print(f"  Steps/chunk:    {STEPS_PER_CHUNK:,}")
    print(f"  dt:             {DT_MS} ms")
    print(f"  Chunk duration: {CHUNK_DURATION_MS / 1000:.1f} s simulated time")
    print(f"  Total sim time: {total_sim_time_ms / 1000:.0f} s")
    print(f"  STP:            enabled (tau_rec_exc={STP_EXC_TAU_REC} ms)")
    print()

    # --- Build network ---
    print("Building network...")
    t_build_start = time.perf_counter()
    (
        params,
        neuron_state,
        syn_state,
        W_exc,
        W_inh,
        stp_params,
        is_excitatory,
        k_noise,
    ) = build_network(n_neurons, seed=seed)
    t_build = time.perf_counter() - t_build_start
    print(f"  Network built in {t_build:.1f} s")
    print()

    # --- Prepare output CSV ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"stability_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADERS)

    print(f"Logging to: {csv_path}")
    print()

    # --- Tracking state ---
    first_chunk_wall_time: float | None = None
    any_nan_detected = False
    any_inf_detected = False
    initial_cpu_rss: float | None = None
    initial_gpu_mem: float | None = None
    max_cpu_rss = 0.0
    max_gpu_mem = 0.0
    firing_rates: list[float] = []
    wall_times: list[float] = []

    # Header for live output
    print(
        f"{'Chunk':>6} | {'Wall(s)':>8} | {'CumSim(s)':>10} | "
        f"{'Spikes':>10} | {'Rate(Hz)':>9} | {'Vmin':>8} | {'Vmax':>8} | "
        f"{'NaN':>4} | {'Inf':>4} | {'CPU(MB)':>9} | {'GPU(MB)':>9}"
    )
    print("-" * 120)

    # --- Simulation loop (chunk by chunk) ---
    for chunk_idx in range(n_chunks):
        # Generate noise current for this chunk
        k_noise, k_chunk = jax.random.split(k_noise)
        I_external = BACKGROUND_MEAN + BACKGROUND_STD * jax.random.normal(
            k_chunk, (STEPS_PER_CHUNK, n_neurons)
        )

        # Run one chunk via the integrator
        t_start = time.perf_counter()
        result = simulate(
            params=params,
            init_state=neuron_state,
            syn_state=syn_state,
            stdp_state=None,
            W_exc=W_exc,
            W_inh=W_inh,
            I_external=I_external,
            dt=DT_MS,
            plasticity_fn=None,
            stp_params=stp_params,
        )

        # Block until computation is done (JAX is async by default)
        spike_history = result.spike_history
        spike_history.block_until_ready()

        wall_time = time.perf_counter() - t_start

        # --- Carry forward state for next chunk ---
        neuron_state = result.final_neuron_state
        syn_state = result.final_syn_state
        # W_exc is unchanged (no plasticity)

        # --- Compute diagnostics ---
        cumulative_sim_time_s = (chunk_idx + 1) * CHUNK_DURATION_MS / 1000.0

        # Spike statistics
        spike_count = int(jnp.sum(spike_history).item())
        chunk_duration_s = CHUNK_DURATION_MS / 1000.0
        mean_rate_hz = spike_count / (n_neurons * chunk_duration_s) if n_neurons > 0 else 0.0

        # Membrane voltage checks
        v_arr = neuron_state.v
        v_min = float(jnp.min(v_arr).item())
        v_max = float(jnp.max(v_arr).item())
        v_has_nan = bool(jnp.any(jnp.isnan(v_arr)).item())
        v_has_inf = bool(jnp.any(jnp.isinf(v_arr)).item())

        # Spike history NaN check (should not happen for bool, but check float path)
        spike_has_nan = bool(jnp.any(jnp.isnan(spike_history.astype(jnp.float32))).item())

        if v_has_nan or spike_has_nan:
            any_nan_detected = True
        if v_has_inf:
            any_inf_detected = True

        # Memory
        cpu_rss = get_cpu_rss_mb()
        gpu_mem = get_gpu_memory_mb()

        if initial_cpu_rss is None:
            initial_cpu_rss = cpu_rss
        if initial_gpu_mem is None and gpu_mem is not None:
            initial_gpu_mem = gpu_mem

        max_cpu_rss = max(max_cpu_rss, cpu_rss)
        if gpu_mem is not None:
            max_gpu_mem = max(max_gpu_mem, gpu_mem)

        # Timing
        if first_chunk_wall_time is None:
            first_chunk_wall_time = wall_time
        wall_times.append(wall_time)
        firing_rates.append(mean_rate_hz)

        # --- Log to CSV ---
        row = [
            chunk_idx + 1,
            f"{wall_time:.3f}",
            f"{cumulative_sim_time_s:.1f}",
            spike_count,
            f"{mean_rate_hz:.3f}",
            f"{v_min:.2f}",
            f"{v_max:.2f}",
            int(v_has_nan),
            int(v_has_inf),
            int(spike_has_nan),
            f"{cpu_rss:.1f}",
            f"{gpu_mem:.1f}" if gpu_mem is not None else "N/A",
        ]
        writer.writerow(row)
        csv_file.flush()

        # --- Log to stdout ---
        gpu_str = f"{gpu_mem:9.1f}" if gpu_mem is not None else "      N/A"
        nan_str = "YES" if (v_has_nan or spike_has_nan) else "no"
        inf_str = "YES" if v_has_inf else "no"
        print(
            f"{chunk_idx + 1:>6} | {wall_time:>8.2f} | {cumulative_sim_time_s:>10.1f} | "
            f"{spike_count:>10,} | {mean_rate_hz:>9.3f} | {v_min:>8.2f} | {v_max:>8.2f} | "
            f"{nan_str:>4} | {inf_str:>4} | {cpu_rss:>9.1f} | {gpu_str}"
        )

    csv_file.close()

    # --- Final summary and pass/fail ---
    print()
    print("=" * 72)
    print("STABILITY TEST SUMMARY")
    print("=" * 72)

    failures: list[str] = []

    # Check 1: NaN/Inf
    if any_nan_detected:
        failures.append("NaN detected in neuron state or spike history")
    if any_inf_detected:
        failures.append("Inf detected in membrane voltage")

    # Check 2: Memory growth
    if initial_cpu_rss is not None and initial_cpu_rss > 0:
        cpu_growth = max_cpu_rss / initial_cpu_rss
        print(f"  CPU RSS:   {initial_cpu_rss:.1f} MB -> {max_cpu_rss:.1f} MB (growth: {cpu_growth:.2f}x)")
        if cpu_growth > MAX_MEMORY_GROWTH_FACTOR:
            failures.append(
                f"CPU memory grew {cpu_growth:.2f}x (threshold: {MAX_MEMORY_GROWTH_FACTOR}x)"
            )
    else:
        print("  CPU RSS:   could not be measured")

    if initial_gpu_mem is not None and initial_gpu_mem > 0:
        gpu_growth = max_gpu_mem / initial_gpu_mem
        print(f"  GPU mem:   {initial_gpu_mem:.1f} MB -> {max_gpu_mem:.1f} MB (growth: {gpu_growth:.2f}x)")
        if gpu_growth > MAX_MEMORY_GROWTH_FACTOR:
            failures.append(
                f"GPU memory grew {gpu_growth:.2f}x (threshold: {MAX_MEMORY_GROWTH_FACTOR}x)"
            )
    else:
        print("  GPU mem:   not available (CPU backend or no stats)")

    # Check 3: Firing rate
    if firing_rates:
        min_rate = min(firing_rates)
        max_rate = max(firing_rates)
        mean_rate = np.mean(firing_rates)
        print(f"  Firing rate: min={min_rate:.3f} Hz, max={max_rate:.3f} Hz, mean={mean_rate:.3f} Hz")

        # Use rates after the first chunk to avoid judging the warmup period
        post_warmup_rates = firing_rates[1:] if len(firing_rates) > 1 else firing_rates
        if any(r <= MIN_FIRING_RATE_HZ for r in post_warmup_rates):
            failures.append(
                f"Firing rate dropped to 0 Hz (dead network) in at least one post-warmup chunk"
            )
        if any(r > MAX_FIRING_RATE_HZ for r in post_warmup_rates):
            failures.append(
                f"Firing rate exceeded {MAX_FIRING_RATE_HZ} Hz in at least one post-warmup chunk"
            )

    # Check 4: Wall time growth
    if first_chunk_wall_time is not None and first_chunk_wall_time > 0 and len(wall_times) > 1:
        # Compare last few chunks to first (skip first which includes JIT compilation)
        # Use second chunk as baseline if available
        baseline_time = wall_times[1] if len(wall_times) > 2 else wall_times[0]
        last_few = wall_times[-3:] if len(wall_times) >= 3 else wall_times[-1:]
        max_recent = max(last_few)
        if baseline_time > 0:
            time_growth = max_recent / baseline_time
            print(
                f"  Wall time: baseline={baseline_time:.2f} s, "
                f"recent_max={max_recent:.2f} s (growth: {time_growth:.2f}x)"
            )
            if time_growth > MAX_WALL_TIME_GROWTH_FACTOR:
                failures.append(
                    f"Wall time grew {time_growth:.2f}x (threshold: {MAX_WALL_TIME_GROWTH_FACTOR}x)"
                )
    else:
        print("  Wall time: only one chunk, cannot assess growth")

    print(f"  NaN detected:  {'YES' if any_nan_detected else 'no'}")
    print(f"  Inf detected:  {'YES' if any_inf_detected else 'no'}")
    print(f"  Results CSV:   {csv_path}")
    print()

    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        print()
        return False
    else:
        print("RESULT: PASS")
        print("  All stability checks passed.")
        print()
        return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BL-1 long-duration stability validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=1.0,
        help="Total simulated duration in hours",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=10000,
        help="Number of neurons in the network",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    passed = run_stability_test(
        duration_hours=args.duration_hours,
        n_neurons=args.n_neurons,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
