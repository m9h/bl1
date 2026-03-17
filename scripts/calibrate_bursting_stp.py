"""Calibrate spontaneous bursting with STP on GPU.

Uses Tsodyks-Markram short-term plasticity to enable burst termination.
Without STP, excitatory cascades don't terminate and the network either
stays silent or goes seizure-like. With STP, excitatory depression
terminates bursts after 100-500ms, producing the burst/quiescence
cycle characteristic of real cortical cultures.

Target ranges (Wagenaar 2006, mature cultures DIV 21-35):
- IBI: 3-30 seconds (mean ~10 s)
- Burst duration: 100-1000 ms
- Burst rate: 3-20 per minute
- Recruitment: 20-80%
- Spontaneous firing between bursts: 0.1-2 Hz

Usage:
    python scripts/calibrate_bursting_stp.py

    # Or on Modal A100:
    modal run benchmarks/modal_benchmark.py --calibrate-stp
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.plasticity.stp import create_stp_params, init_stp_state, stp_step


def run_with_stp(
    n_neurons: int = 5000,
    g_exc: float = 0.08,
    g_inh: float = 0.32,
    p_max: float = 0.21,
    background_mean: float = 1.0,
    background_std: float = 3.0,
    duration_ms: float = 30000,
    dt: float = 0.5,
    seed: int = 42,
) -> dict:
    """Run simulation WITH Tsodyks-Markram STP and return burst statistics.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (keep <= 5000 for tractable sweeps).
    g_exc : float
        Excitatory synaptic conductance -- needs to be strong enough for
        cascades but STP depression will terminate them.
    g_inh : float
        Inhibitory synaptic conductance (typically 3.5-5x g_exc).
    p_max : float
        Maximum connection probability at distance zero.
    background_mean : float
        Mean tonic depolarising current.  Near but below threshold so
        noise-driven fluctuations can nucleate bursts.
    background_std : float
        Standard deviation of per-neuron per-timestep Gaussian noise.
        Must be large enough to push near-threshold neurons over.
    duration_ms : float
        Simulation duration in milliseconds.
    dt : float
        Integration timestep in ms.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Burst statistics and simulation metadata.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # --- Build network ---
    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=p_max, g_exc=g_exc, g_inh=g_inh,
    )
    syn = create_synapse_state(n_neurons)

    # --- STP: excitatory depression (U=0.5, tau_rec=800ms) ---
    stp_params = create_stp_params(n_neurons, is_exc)
    stp_state = init_stp_state(n_neurons, stp_params)

    n_steps = int(duration_ms / dt)

    # --- Background noise current ---
    I_noise = background_mean + background_std * jax.random.normal(
        k4, (n_steps, n_neurons),
    )

    # --- Simulation loop with STP via jax.lax.scan ---
    def step_fn(carry, I_t):
        ns, ss, stp_st = carry

        # Synaptic current from existing conductances
        I_syn = compute_synaptic_current(ss, ns.v)
        I_total = I_syn + I_t

        # Izhikevich neuron update
        ns = izhikevich_step(ns, params, I_total, dt)

        # STP modulates presynaptic spike amplitude
        stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, dt)

        # Update conductances with STP-scaled spikes
        new_ampa = ampa_step(ss.g_ampa, scale, W_exc, dt)
        new_gaba = gaba_a_step(ss.g_gaba_a, scale, W_inh, dt)

        ss = SynapseState(
            g_ampa=new_ampa,
            g_gaba_a=new_gaba,
            g_nmda_rise=ss.g_nmda_rise,
            g_nmda_decay=ss.g_nmda_decay,
            g_gaba_b_rise=ss.g_gaba_b_rise,
            g_gaba_b_decay=ss.g_gaba_b_decay,
        )
        return (ns, ss, stp_st), ns.spikes

    t0 = time.perf_counter()
    (_, _, _), spikes = jax.lax.scan(
        step_fn, (state, syn, stp_state), I_noise,
    )
    spikes.block_until_ready()
    wall_time = time.perf_counter() - t0

    # --- Analysis ---
    raster = np.asarray(spikes)
    bursts = detect_bursts(raster, dt_ms=dt, threshold_std=2.0, min_duration_ms=50.0)
    stats = burst_statistics(bursts)

    total_spikes = float(raster.sum())
    mean_rate = total_spikes / (n_neurons * duration_ms / 1000.0)

    return {
        "n_bursts": len(bursts),
        "burst_rate_per_min": len(bursts) / (duration_ms / 60000.0),
        **stats,
        "mean_firing_rate": mean_rate,
        "wall_time_s": wall_time,
        "params": {
            "g_exc": g_exc,
            "g_inh": g_inh,
            "p_max": p_max,
            "background_mean": background_mean,
            "background_std": background_std,
        },
    }


def score_against_wagenaar(r: dict) -> float:
    """Score a result against Wagenaar 2006 target ranges (lower is better).

    Penalties are squared deviations from acceptable ranges, normalised
    so that each criterion contributes roughly equally.
    """
    s = 0.0

    # Burst rate: 3-20 per minute
    br = r["burst_rate_per_min"]
    if br < 3:
        s += (3 - br) ** 2
    elif br > 20:
        s += (br - 20) ** 2

    # IBI: 3000-30000 ms (3-30 s)
    ibi = r.get("ibi_mean", float("nan"))
    if not np.isnan(ibi):
        if ibi < 3000:
            s += ((3000 - ibi) / 1000) ** 2
        elif ibi > 30000:
            s += ((ibi - 30000) / 1000) ** 2
    else:
        s += 100  # no IBI means < 2 bursts detected

    # Burst duration: 100-1000 ms
    dur = r.get("duration_mean", float("nan"))
    if not np.isnan(dur):
        if dur < 100:
            s += ((100 - dur) / 100) ** 2
        elif dur > 1000:
            s += ((dur - 1000) / 100) ** 2
    else:
        s += 50

    # Recruitment: 0.2-0.8
    rec = r.get("recruitment_mean", float("nan"))
    if not np.isnan(rec):
        if rec < 0.2:
            s += ((0.2 - rec) / 0.1) ** 2
        elif rec > 0.8:
            s += ((rec - 0.8) / 0.1) ** 2

    # Mean firing rate: penalise seizure-like (>5 Hz) or silent (<0.05 Hz)
    mfr = r.get("mean_firing_rate", 0.0)
    if mfr > 5.0:
        s += ((mfr - 5.0) / 1.0) ** 2
    elif mfr < 0.05:
        s += 10

    return s


def sweep():
    """Run parameter sweep with STP enabled.

    Grid:
    - g_exc: [0.06, 0.08, 0.10, 0.12] -- strong enough for cascade
    - g_inh: g_exc * [3.5, 4.0, 5.0] -- E/I ratio
    - background_mean: [0.5, 1.0, 2.0] -- near but below threshold
    - background_std: [2.5, 3.0, 3.5] -- noise to trigger cascades
    - p_max: [0.15, 0.21] -- connectivity density
    """
    print("=" * 72)
    print("BL-1 Bursting Calibration WITH STP")
    print("Wagenaar et al. (2006) targets")
    print("=" * 72)
    print()

    g_exc_values = [0.06, 0.08, 0.10, 0.12]
    gi_mult_values = [3.5, 4.0, 5.0]
    bg_mean_values = [0.5, 1.0, 2.0]
    bg_std_values = [2.5, 3.0, 3.5]
    p_max_values = [0.15, 0.21]

    grid = list(itertools.product(
        g_exc_values, gi_mult_values, bg_mean_values,
        bg_std_values, p_max_values,
    ))

    print(f"Total parameter combinations: {len(grid)}")
    print()

    results: list[dict] = []

    for i, (g_exc, gi_mult, bg_mean, bg_std, p_max) in enumerate(grid, 1):
        g_inh = g_exc * gi_mult
        tag = (
            f"[{i}/{len(grid)}] g_exc={g_exc:.3f}, g_inh={g_inh:.3f}, "
            f"bg={bg_mean:.1f}+/-{bg_std:.1f}, p_max={p_max:.2f}"
        )
        print(tag, end=" ... ", flush=True)

        try:
            result = run_with_stp(
                n_neurons=5000,
                g_exc=g_exc,
                g_inh=g_inh,
                p_max=p_max,
                background_mean=bg_mean,
                background_std=bg_std,
                duration_ms=30000,
                seed=42,
            )
            results.append(result)

            ibi_str = (
                f"{result['ibi_mean']:.0f}"
                if not np.isnan(result.get("ibi_mean", float("nan")))
                else "N/A"
            )
            dur_str = (
                f"{result['duration_mean']:.0f}"
                if not np.isnan(result.get("duration_mean", float("nan")))
                else "N/A"
            )
            rec_str = (
                f"{result['recruitment_mean']:.2f}"
                if not np.isnan(result.get("recruitment_mean", float("nan")))
                else "N/A"
            )
            print(
                f"bursts={result['n_bursts']:3d}, "
                f"rate={result['burst_rate_per_min']:5.1f}/min, "
                f"IBI={ibi_str:>6s}ms, "
                f"dur={dur_str:>5s}ms, "
                f"rec={rec_str}, "
                f"MFR={result['mean_firing_rate']:.2f}Hz, "
                f"wall={result['wall_time_s']:.1f}s"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("\nNo successful simulations. Check the BL-1 installation.")
        return []

    # --- Rank results ---
    print()
    print("=" * 72)
    print("TOP 5 MATCHES TO WAGENAAR 2006 (with STP)")
    print("=" * 72)

    scored = sorted(results, key=score_against_wagenaar)
    for rank, r in enumerate(scored[:5], 1):
        sc = score_against_wagenaar(r)
        ibi = r.get("ibi_mean", float("nan"))
        dur = r.get("duration_mean", float("nan"))
        rec = r.get("recruitment_mean", float("nan"))
        print(f"\n  #{rank}  score = {sc:.2f}")
        print(f"    Params:      {r['params']}")
        print(f"    Burst rate:  {r['burst_rate_per_min']:.1f} /min")
        print(
            f"    IBI:         {'%.0f' % ibi if not np.isnan(ibi) else 'N/A'} ms"
        )
        print(
            f"    Duration:    {'%.0f' % dur if not np.isnan(dur) else 'N/A'} ms"
        )
        print(
            f"    Recruitment: {'%.2f' % rec if not np.isnan(rec) else 'N/A'}"
        )
        print(f"    Mean rate:   {r['mean_firing_rate']:.2f} Hz")

    # --- Report best parameters ---
    best = scored[0]
    bp = best["params"]
    print()
    print("=" * 72)
    print("RECOMMENDED CONFIG (configs/wagenaar_calibrated.yaml):")
    print(f"  g_exc:            {bp['g_exc']}")
    print(f"  g_inh:            {bp['g_inh']}")
    print(f"  p_max:            {bp['p_max']}")
    print(f"  background.mean:  {bp['background_mean']}")
    print(f"  background.std:   {bp['background_std']}")
    print(f"  stp.enabled:      true")
    print(f"  stp.U_exc:        0.5")
    print(f"  stp.tau_rec_exc:  800.0")
    print("=" * 72)

    return scored


if __name__ == "__main__":
    sweep()
