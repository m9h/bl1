"""Calibrate BL-1 spontaneous bursting to match Wagenaar et al. (2006).

Sweeps key parameters (g_exc, background_noise, p_max) and reports
burst statistics for each combination.

Target ranges (Wagenaar 2006, mature cultures DIV 21-35):
- IBI: 3-30 seconds (mean ~10 s)
- Burst duration: 100-1000 ms
- Burst rate: 3-20 per minute
- Recruitment: 20-80%
- Spontaneous firing between bursts: 0.1-2 Hz

Usage:
    python scripts/calibrate_bursting.py
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from bl1.core.izhikevich import create_population, izhikevich_step, NeuronState
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.analysis.bursts import detect_bursts, burst_statistics


def run_culture_simulation(
    n_neurons: int = 5000,
    g_exc: float = 0.05,
    g_inh: float = 0.20,
    p_max: float = 0.15,
    lambda_um: float = 200.0,
    background_mean: float = 3.5,
    background_std: float = 2.5,
    duration_ms: float = 30000,
    dt: float = 0.5,
    seed: int = 42,
) -> dict:
    """Run a spontaneous activity simulation and return burst stats.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (keep <= 5000 for tractable sweeps).
    g_exc : float
        Excitatory synaptic conductance.
    g_inh : float
        Inhibitory synaptic conductance (typically 4x g_exc).
    p_max : float
        Maximum connection probability at distance zero.
    lambda_um : float
        Spatial decay constant for connection probability (um).
    background_mean : float
        Mean tonic depolarising current injected into all neurons.
    background_std : float
        Standard deviation of per-neuron per-timestep Gaussian noise.
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
    W_exc, W_inh, _delays = build_connectivity(
        k3,
        positions,
        is_exc,
        lambda_um=lambda_um,
        p_max=p_max,
        g_exc=g_exc,
        g_inh=g_inh,
    )
    syn_state = create_synapse_state(n_neurons)

    n_steps = int(duration_ms / dt)

    # --- Background noise current ---
    I_noise = background_mean + background_std * jax.random.normal(
        k4, (n_steps, n_neurons)
    )

    # --- Simulation loop via jax.lax.scan ---
    def step_fn(carry, I_t):
        ns, ss = carry

        # Synaptic current from existing conductances
        I_syn = compute_synaptic_current(ss, ns.v)
        I_total = I_syn + I_t

        # Izhikevich neuron update
        ns = izhikevich_step(ns, params, I_total, dt)

        # Update conductances with new spikes
        spikes_f = ns.spikes.astype(jnp.float32)
        new_ampa = ampa_step(ss.g_ampa, spikes_f, W_exc, dt)
        new_gaba = gaba_a_step(ss.g_gaba_a, spikes_f, W_inh, dt)

        ss = SynapseState(
            g_ampa=new_ampa,
            g_gaba_a=new_gaba,
            g_nmda_rise=ss.g_nmda_rise,
            g_nmda_decay=ss.g_nmda_decay,
            g_gaba_b_rise=ss.g_gaba_b_rise,
            g_gaba_b_decay=ss.g_gaba_b_decay,
        )
        return (ns, ss), ns.spikes

    t0 = time.perf_counter()
    (final_ns, final_ss), spike_history = jax.lax.scan(
        step_fn, (state, syn_state), I_noise
    )
    spike_history.block_until_ready()
    wall_time = time.perf_counter() - t0

    # --- Analysis ---
    raster = np.asarray(spike_history)

    bursts = detect_bursts(raster, dt_ms=dt, threshold_std=2.0, min_duration_ms=50.0)
    stats = burst_statistics(bursts)

    # Mean firing rate (Hz)
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

    # Mean firing rate: 0.1-2 Hz (penalise seizure-like or silent regimes)
    mfr = r.get("mean_firing_rate", 0.0)
    if mfr > 5.0:
        s += ((mfr - 5.0) / 1.0) ** 2
    elif mfr < 0.05:
        s += 10

    return s


def parameter_sweep():
    """Sweep key parameters and find the best match to Wagenaar 2006."""
    print("=" * 72)
    print("BL-1 Bursting Calibration — Wagenaar et al. (2006) targets")
    print("=" * 72)
    print()

    # Parameter grid (kept small for tractability)
    g_exc_values = [0.03, 0.05, 0.07]
    bg_mean_values = [2.0, 3.5, 5.0]
    bg_std_values = [1.5, 2.5]
    p_max_values = [0.15, 0.21]

    combos = list(
        itertools.product(g_exc_values, bg_mean_values, bg_std_values, p_max_values)
    )
    print(f"Total parameter combinations: {len(combos)}")
    print()

    results: list[dict] = []

    for i, (g_exc, bg_mean, bg_std, p_max) in enumerate(combos, 1):
        tag = (
            f"[{i}/{len(combos)}] g_exc={g_exc:.3f}, "
            f"bg_mean={bg_mean:.1f}, bg_std={bg_std:.1f}, p_max={p_max:.2f}"
        )
        print(tag)

        try:
            result = run_culture_simulation(
                n_neurons=5000,
                g_exc=g_exc,
                g_inh=g_exc * 4,
                p_max=p_max,
                background_mean=bg_mean,
                background_std=bg_std,
                duration_ms=30000,  # 30 seconds
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
                f"  Bursts: {result['n_bursts']:3d}, "
                f"Rate: {result['burst_rate_per_min']:5.1f}/min, "
                f"IBI: {ibi_str:>6s} ms, "
                f"Duration: {dur_str:>5s} ms, "
                f"Recruitment: {rec_str}, "
                f"MFR: {result['mean_firing_rate']:.2f} Hz, "
                f"Wall: {result['wall_time_s']:.1f}s"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("\nNo successful simulations. Check the BL-1 installation.")
        return

    # --- Rank results ---
    print()
    print("=" * 72)
    print("TOP 5 MATCHES TO WAGENAAR 2006")
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
    print("=" * 72)


if __name__ == "__main__":
    parameter_sweep()
