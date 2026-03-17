"""BL-1 Performance Benchmark

Profiles simulation wall-clock time at various network sizes:
- 1K, 5K, 10K, 50K, 100K neurons
- With and without STDP plasticity
- Reports: setup time, JIT compile time, simulation time, spikes/sec

Usage:
    python benchmarks/profile_scale.py
    python benchmarks/profile_scale.py --n-neurons 100000 --duration-ms 5000
"""

import argparse
import time

import jax
import jax.numpy as jnp

from bl1.core.izhikevich import IzhikevichParams, NeuronState, create_population
from bl1.core.synapses import SynapseState, create_synapse_state
from bl1.core.integrator import simulate
from bl1.network.topology import place_neurons, build_connectivity
from bl1.plasticity.stdp import STDPParams, STDPState, init_stdp_state, stdp_update


def benchmark_network_creation(key, n_neurons):
    """Time network creation (placement + connectivity)."""
    k1, k2, k3 = jax.random.split(key, 3)

    t0 = time.perf_counter()
    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    t_place = time.perf_counter() - t0

    t0 = time.perf_counter()
    params, state, is_exc = create_population(k2, n_neurons)
    t_pop = time.perf_counter() - t0

    t0 = time.perf_counter()
    W_exc, W_inh, delays = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=0.05, g_inh=0.20,
    )
    t_conn = time.perf_counter() - t0

    print(f"  Placement:    {t_place:.3f}s")
    print(f"  Population:   {t_pop:.3f}s")
    print(f"  Connectivity: {t_conn:.3f}s")
    print(f"  W_exc nnz:    {W_exc.data.shape[0]:,}")
    print(f"  W_inh nnz:    {W_inh.data.shape[0]:,}")

    return positions, params, state, is_exc, W_exc, W_inh


def benchmark_simulation(
    params, state, syn_state, W_exc, W_inh, n_steps, dt=0.5,
    with_stdp=False, is_exc=None,
):
    """Time the core simulation loop."""
    N = state.v.shape[0]

    # Random external current (small background noise)
    key = jax.random.PRNGKey(99)
    I_external = jax.random.normal(key, (n_steps, N)) * 3.0

    # Plasticity setup
    # The integrator's plasticity_fn signature:
    #   (stdp_state, spikes, W_exc) -> (stdp_state, W_exc)
    if with_stdp:
        stdp_params = STDPParams()
        stdp_state = init_stdp_state(N)

        def plasticity_fn(stdp_st, spikes, w_exc):
            return stdp_update(stdp_st, stdp_params, spikes, w_exc, is_exc, dt)
    else:
        stdp_state = None
        plasticity_fn = None

    # JIT compile (first call with a small warmup)
    warmup_steps = min(100, n_steps)
    t0 = time.perf_counter()
    result = simulate(
        params, state, syn_state, stdp_state,
        W_exc, W_inh, I_external[:warmup_steps],
        dt=dt, plasticity_fn=plasticity_fn,
    )
    # Block until computation is done
    result.spike_history.block_until_ready()
    t_jit = time.perf_counter() - t0

    # Actual timed run
    t0 = time.perf_counter()
    result = simulate(
        params, state, syn_state, stdp_state,
        W_exc, W_inh, I_external,
        dt=dt, plasticity_fn=plasticity_fn,
    )
    result.spike_history.block_until_ready()
    t_sim = time.perf_counter() - t0

    total_spikes = int(jnp.sum(result.spike_history))
    sim_time_ms = n_steps * dt

    print(f"  JIT compile:  {t_jit:.3f}s ({warmup_steps} steps warmup)")
    print(f"  Simulation:   {t_sim:.3f}s ({n_steps} steps = {sim_time_ms / 1000:.1f}s simulated)")
    print(f"  Realtime factor: {sim_time_ms / (t_sim * 1000):.2f}x")
    print(f"  Total spikes: {total_spikes:,}")
    if sim_time_ms > 0 and N > 0:
        print(f"  Mean rate:    {total_spikes / (N * sim_time_ms / 1000):.1f} Hz")

    return t_jit, t_sim, total_spikes


def run_benchmarks(sizes=None, duration_ms=5000, dt=0.5):
    """Run benchmarks across multiple network sizes."""
    if sizes is None:
        sizes = [1000, 5000, 10000]

    n_steps = int(duration_ms / dt)

    print("=" * 70)
    print("BL-1 Performance Benchmark")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Duration: {duration_ms}ms = {n_steps} steps at dt={dt}ms")
    print("=" * 70)

    results = []

    for N in sizes:
        print(f"\n{'=' * 50}")
        print(f"Network size: {N:,} neurons")
        print(f"{'=' * 50}")

        key = jax.random.PRNGKey(42)

        print("\n[1] Network creation:")
        positions, params, state, is_exc, W_exc, W_inh = benchmark_network_creation(key, N)

        syn_state = create_synapse_state(N)

        print("\n[2] Simulation WITHOUT STDP:")
        t_jit, t_sim, spikes = benchmark_simulation(
            params, state, syn_state, W_exc, W_inh, n_steps, dt,
            with_stdp=False,
        )
        results.append({
            "N": N, "stdp": False, "jit_s": t_jit, "sim_s": t_sim,
            "spikes": spikes, "steps": n_steps,
        })

        # Only run STDP benchmark for smaller networks (it's slower)
        if N <= 10000:
            print("\n[3] Simulation WITH STDP:")
            t_jit2, t_sim2, spikes2 = benchmark_simulation(
                params, state, syn_state, W_exc, W_inh, n_steps, dt,
                with_stdp=True, is_exc=is_exc,
            )
            results.append({
                "N": N, "stdp": True, "jit_s": t_jit2, "sim_s": t_sim2,
                "spikes": spikes2, "steps": n_steps,
            })

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'N':>8} {'STDP':>5} {'JIT(s)':>8} {'Sim(s)':>8} {'RT Factor':>10} {'Spikes':>10}")
    print("-" * 60)
    for r in results:
        sim_ms = r["steps"] * dt
        rt = sim_ms / (r["sim_s"] * 1000) if r["sim_s"] > 0 else float("inf")
        print(
            f"{r['N']:>8,} {'Yes' if r['stdp'] else 'No':>5} "
            f"{r['jit_s']:>8.2f} {r['sim_s']:>8.2f} {rt:>10.2f}x {r['spikes']:>10,}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BL-1 Performance Benchmark")
    parser.add_argument(
        "--n-neurons", type=int, nargs="+", default=[1000, 5000, 10000],
        help="Network sizes to benchmark",
    )
    parser.add_argument(
        "--duration-ms", type=float, default=5000,
        help="Simulation duration in ms",
    )
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    run_benchmarks(sizes=args.n_neurons, duration_ms=args.duration_ms, dt=args.dt)
