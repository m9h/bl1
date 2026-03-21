"""Train bl1 synaptic weights via differentiable optimization.

Usage:
    python scripts/train_culture.py                          # default config
    python scripts/train_culture.py --n-neurons 5000 --n-epochs 100
    python scripts/train_culture.py --n-neurons 10000 --n-epochs 200 --lr 5e-4
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments corresponding to TrainingConfig fields."""
    p = argparse.ArgumentParser(
        description="Train BL-1 synaptic weights toward Wagenaar 2006 targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Network
    p.add_argument("--n-neurons", type=int, default=5000,
                    help="Number of neurons in the culture.")
    p.add_argument("--ei-ratio", type=float, default=0.8,
                    help="Fraction of excitatory neurons.")

    # Simulation
    p.add_argument("--sim-duration-ms", type=float, default=2000.0,
                    help="Simulation duration per epoch (ms).")
    p.add_argument("--dt", type=float, default=0.5,
                    help="Integration timestep (ms).")

    # Optimizer
    p.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                    dest="learning_rate",
                    help="Adam learning rate.")
    p.add_argument("--n-epochs", type=int, default=100,
                    help="Number of training epochs.")

    # Targets
    p.add_argument("--target-firing-rate-hz", type=float, default=1.4,
                    help="Target mean firing rate (Hz), Wagenaar 2006.")
    p.add_argument("--target-burst-rate-per-min", type=float, default=8.0,
                    help="Target network burst rate (bursts/min), Wagenaar 2006.")

    # Loss weights
    p.add_argument("--w-firing-rate", type=float, default=1.0,
                    help="Weight for firing rate loss.")
    p.add_argument("--w-burst-rate", type=float, default=1.0,
                    help="Weight for burst rate loss.")
    p.add_argument("--w-synchrony", type=float, default=0.5,
                    help="Weight for synchrony loss.")
    p.add_argument("--w-weight-reg", type=float, default=0.01,
                    help="Weight for weight regularization loss.")

    # Surrogate gradient
    p.add_argument("--surrogate-beta", type=float, default=10.0,
                    help="Sharpness of SuperSpike surrogate gradient.")

    # Network topology
    p.add_argument("--lambda-um", type=float, default=200.0,
                    help="Connectivity length constant (um).")
    p.add_argument("--p-max", type=float, default=0.02,
                    help="Maximum connection probability.")
    p.add_argument("--g-exc-init", type=float, default=0.50,
                    help="Initial excitatory conductance.")
    p.add_argument("--g-inh-init", type=float, default=2.00,
                    help="Initial inhibitory conductance.")

    # External drive
    p.add_argument("--I-noise-amplitude", type=float, default=5.0,
                    help="Amplitude of external noise current.")

    # Real data
    p.add_argument("--from-recording", type=str, default=None,
                    help="Path to NWB or HDF5 recording. Extracts firing rate "
                         "and burst rate as training targets automatically.")

    # Misc
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    p.add_argument("--validate", action="store_true",
                    help="Run a validation simulation after training and "
                         "compare to Wagenaar targets.")
    p.add_argument("--output-dir", type=str, default="results",
                    help="Directory to save trained weights.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for training."""
    args = parse_args(argv)

    from bl1.training.trainer import TrainingConfig, train_weights

    # --- Extract targets from real recording if provided ---
    target_fr = args.target_firing_rate_hz
    target_burst = args.target_burst_rate_per_min

    if args.from_recording:
        print(f"Loading recording: {args.from_recording}")
        from bl1.validation.loaders import (
            load_nwb_spike_trains,
            load_maxwell_h5,
            compute_recording_statistics,
        )
        ext = args.from_recording.rsplit(".", 1)[-1].lower()
        if ext == "nwb":
            rec = load_nwb_spike_trains(args.from_recording)
        else:
            rec = load_maxwell_h5(args.from_recording)

        # Auto-detect sample indices (>24h = likely not seconds)
        if rec["duration_s"] > 86400:
            sr = rec.get("sampling_rate", 20000.0)
            rec["spike_times"] = [st / sr for st in rec["spike_times"]]
            rec["duration_s"] /= sr
            print(f"  Converted sample indices at {sr:.0f} Hz")

        # Find actual activity window (some recordings start late)
        active_trains = [st for st in rec["spike_times"] if len(st) > 0]
        if not active_trains:
            print("  WARNING: No spikes found in recording, using default targets.")
            target_fr = args.target_firing_rate_hz
            target_burst = args.target_burst_rate_per_min
        else:
            t_min = float(min(st.min() for st in active_trains))
            t_max = float(max(st.max() for st in active_trains))
            actual_dur = t_max - t_min
            use_dur = min(actual_dur, 120.0)
            # Trim and shift to t=0
            trimmed_times = []
            for st in rec["spike_times"]:
                mask = (st >= t_min) & (st <= t_min + use_dur)
                trimmed_times.append(st[mask] - t_min)
            trimmed = {
                "spike_times": trimmed_times,
                "duration_s": use_dur,
                "n_units": rec["n_units"],
            }
            stats = compute_recording_statistics(trimmed, dt_ms=0.5, burst_threshold_std=1.5)
            target_fr = stats["mean_firing_rate_hz"]
            target_burst = stats["burst_rate_per_min"]
            print(f"  Activity window: {t_min:.1f}s - {t_min+use_dur:.1f}s ({use_dur:.1f}s)")
            print(f"  Units: {rec['n_units']}")
            print(f"  Extracted targets: FR={target_fr:.2f} Hz, bursts={target_burst:.1f}/min")
            print()

    config = TrainingConfig(
        n_neurons=args.n_neurons,
        ei_ratio=args.ei_ratio,
        sim_duration_ms=args.sim_duration_ms,
        dt=args.dt,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        target_firing_rate_hz=target_fr,
        target_burst_rate_per_min=target_burst,
        w_firing_rate=args.w_firing_rate,
        w_burst_rate=args.w_burst_rate,
        w_synchrony=args.w_synchrony,
        w_weight_reg=args.w_weight_reg,
        surrogate_beta=args.surrogate_beta,
        lambda_um=args.lambda_um,
        p_max=args.p_max,
        g_exc_init=args.g_exc_init,
        g_inh_init=args.g_inh_init,
        I_noise_amplitude=args.I_noise_amplitude,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("BL-1 Differentiable Training")
    print("=" * 72)
    print(f"  Neurons:      {config.n_neurons}")
    print(f"  Epochs:       {config.n_epochs}")
    print(f"  Sim duration: {config.sim_duration_ms} ms")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Seed:         {config.seed}")
    print()

    result = train_weights(config)

    # -----------------------------------------------------------------------
    # Final loss breakdown
    # -----------------------------------------------------------------------
    final = result.loss_history[-1]
    initial = result.loss_history[0]

    print()
    print("=" * 72)
    print("Final Loss Breakdown")
    print("=" * 72)
    for key in sorted(final.keys()):
        if key in ("epoch", "wall_time_s"):
            continue
        print(f"  {key:25s}: {final[key]:10.4f}")

    # -----------------------------------------------------------------------
    # Save weights
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_path = output_dir / f"trained_weights_{timestamp}.npz"

    np.savez(
        weight_path,
        W_exc=np.array(result.W_exc),
        W_inh=np.array(result.W_inh),
    )
    print(f"\nWeights saved to: {weight_path}")

    # -----------------------------------------------------------------------
    # Comparison: initial vs final metrics
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Training Comparison: Initial vs Final")
    print("=" * 72)
    print(f"  {'Metric':<30s} {'Initial':>12s} {'Final':>12s} {'Target':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    if "mean_fr_hz" in initial and "mean_fr_hz" in final:
        print(f"  {'Firing rate (Hz)':<30s} "
              f"{initial['mean_fr_hz']:12.2f} "
              f"{final['mean_fr_hz']:12.2f} "
              f"{config.target_firing_rate_hz:12.2f}")

    if "burst_rate_per_min" in initial and "burst_rate_per_min" in final:
        print(f"  {'Burst rate (/min)':<30s} "
              f"{initial['burst_rate_per_min']:12.1f} "
              f"{final['burst_rate_per_min']:12.1f} "
              f"{config.target_burst_rate_per_min:12.1f}")

    print(f"  {'Total loss':<30s} "
          f"{initial['total']:12.4f} "
          f"{final['total']:12.4f} "
          f"{'0.0':>12s}")

    # -----------------------------------------------------------------------
    # Optional validation simulation
    # -----------------------------------------------------------------------
    if args.validate:
        print()
        print("=" * 72)
        print("Validation Simulation")
        print("=" * 72)

        _run_validation(result, config)


def _run_validation(result, config):
    """Run a longer validation simulation with trained weights and compare to targets."""
    from bl1.core.integrator import simulate
    from bl1.core.izhikevich import NeuronState, create_population
    from bl1.core.synapses import create_synapse_state
    from bl1.network.topology import build_connectivity, place_neurons

    # Use a different seed for validation
    val_seed = config.seed + 9999
    key = jax.random.PRNGKey(val_seed)
    key_pop, key_pos, key_conn, key_noise, key_state = jax.random.split(key, 5)

    # Recreate network structure (same as training but different seed for noise)
    params, init_state_template, is_excitatory = create_population(
        key_pop, config.n_neurons, ei_ratio=config.ei_ratio
    )

    syn_state = create_synapse_state(config.n_neurons)

    # Longer validation duration (5 seconds)
    val_duration_ms = 5000.0
    n_steps = int(val_duration_ms / config.dt)

    I_external = (
        config.I_noise_amplitude
        * jax.random.normal(key_noise, shape=(n_steps, config.n_neurons))
    )

    v_init = -65.0 + 5.0 * jax.random.normal(
        key_state, shape=(config.n_neurons,)
    )
    init_state = NeuronState(
        v=v_init,
        u=init_state_template.u,
        spikes=jnp.zeros(config.n_neurons, dtype=jnp.float32),
    )

    print("  Running 5-second validation simulation...")
    sim_result = simulate(
        params=params,
        init_state=init_state,
        syn_state=syn_state,
        stdp_state=None,
        W_exc=result.W_exc,
        W_inh=result.W_inh,
        I_external=I_external,
        dt=config.dt,
        plasticity_fn=None,
        surrogate=False,  # Use hard threshold for validation
    )

    spike_history = sim_result.spike_history
    T, N = spike_history.shape
    sim_s = T * config.dt / 1000.0

    # Compute metrics
    total_spikes = float(jnp.sum(spike_history))
    mean_fr = total_spikes / N / sim_s
    active_frac = float(jnp.mean(jnp.sum(spike_history, axis=0) > 0))

    # Wagenaar targets
    target_fr = config.target_firing_rate_hz
    target_burst = config.target_burst_rate_per_min

    print(f"  Validation duration: {val_duration_ms:.0f} ms ({sim_s:.1f} s)")
    print(f"  Mean firing rate:    {mean_fr:.2f} Hz  (target: {target_fr:.1f} Hz)")
    print(f"  Active neurons:      {active_frac*100:.1f}%")
    print(f"  Total spikes:        {int(total_spikes):,}")

    # Check against Wagenaar targets
    fr_ok = abs(mean_fr - target_fr) / max(target_fr, 0.1) < 0.5
    print()
    if fr_ok:
        print("  [PASS] Firing rate within 50% of Wagenaar target")
    else:
        print("  [WARN] Firing rate deviates >50% from Wagenaar target")


if __name__ == "__main__":
    main()
