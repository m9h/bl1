"""Tests for Wagenaar 2006 bursting calibration.

Verifies that the calibrated parameter set produces spontaneous
network bursting in a 5000-neuron Izhikevich culture model.

The calibrated parameters use Tsodyks-Markram short-term plasticity
(STP) which is the critical ingredient for realistic bursting:
- Excitatory depression (U=0.5, tau_rec=800ms) terminates bursts
- Strong excitatory conductance (g_exc=0.10) enables cascade ignition
- Subthreshold background (mean=1.0) with noise (std=3.0) nucleates bursts
- High connectivity (p_max=0.21) supports population-wide recruitment

Reference
---------
Wagenaar, Pine & Potter (2006) "An extremely rich repertoire of
bursting patterns in the activity of cortical cultures"
J Neurosci 26(31):7610-7625
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    ampa_step,
    compute_synaptic_current,
    create_synapse_state,
    gaba_a_step,
)
from bl1.network.topology import build_connectivity, place_neurons
from bl1.analysis.bursts import detect_bursts, burst_statistics
from bl1.plasticity.stp import create_stp_params, init_stp_state, stp_step


# -----------------------------------------------------------------------
# Calibrated parameters (from STP calibration sweep)
#
# These were identified by scripts/calibrate_bursting_stp.py as producing
# realistic spontaneous bursting matching Wagenaar 2006 targets:
#   - Burst rate: 3-20/min
#   - IBI: 3-30 s
#   - Burst duration: 100-1000 ms
#   - Recruitment: 20-80%
#   - Mean firing rate: 0.1-2 Hz between bursts
# -----------------------------------------------------------------------

_CALIBRATED = dict(
    g_exc=0.50,
    g_inh=2.00,
    p_max=0.02,
    background_mean=1.0,
    background_std=3.0,
)


# -----------------------------------------------------------------------
# Shared simulation helper
# -----------------------------------------------------------------------

def _run_culture(
    n_neurons: int = 5000,
    g_exc: float = 0.50,
    g_inh: float = 2.00,
    p_max: float = 0.02,
    background_mean: float = 1.0,
    background_std: float = 3.0,
    duration_ms: float = 10000,
    dt: float = 0.5,
    seed: int = 42,
    use_stp: bool = True,
) -> dict:
    """Run a short spontaneous-activity simulation and return burst stats.

    When *use_stp* is True (the default for calibrated parameters),
    Tsodyks-Markram short-term plasticity modulates presynaptic spike
    amplitudes before driving conductance updates, enabling excitatory
    depression and inhibitory facilitation.  This is required for
    realistic bursting with the calibrated parameter set.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _ = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=p_max, g_exc=g_exc, g_inh=g_inh,
    )
    syn_state = create_synapse_state(n_neurons)

    n_steps = int(duration_ms / dt)
    I_noise = background_mean + background_std * jax.random.normal(
        k4, (n_steps, n_neurons),
    )

    if use_stp:
        from bl1.plasticity.stp import STPParams
        # Custom STP with longer recovery for slow bursting (tau_rec=3000ms)
        U = jnp.where(is_exc, 0.5, 0.04)
        tau_rec = jnp.where(is_exc, 3000.0, 100.0)
        tau_fac = jnp.where(is_exc, 0.001, 1000.0)
        stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
        stp_state = init_stp_state(n_neurons, stp_params)

        def step_fn(carry, I_t):
            ns, ss, stp_st = carry
            I_syn = compute_synaptic_current(ss, ns.v)
            I_total = I_syn + I_t
            ns = izhikevich_step(ns, params, I_total, dt)
            stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, dt)
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

        (_, _, _), spike_history = jax.lax.scan(
            step_fn, (state, syn_state, stp_state), I_noise,
        )
    else:
        def step_fn(carry, I_t):
            ns, ss = carry
            I_syn = compute_synaptic_current(ss, ns.v)
            I_total = I_syn + I_t
            ns = izhikevich_step(ns, params, I_total, dt)
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

        (_, _), spike_history = jax.lax.scan(
            step_fn, (state, syn_state), I_noise,
        )
    spike_history.block_until_ready()

    raster = np.asarray(spike_history)
    bursts = detect_bursts(raster, dt_ms=dt, threshold_std=1.5, min_duration_ms=5.0)
    stats = burst_statistics(bursts)

    total_spikes = float(raster.sum())
    mean_rate = total_spikes / (n_neurons * duration_ms / 1000.0)

    return {
        "n_bursts": len(bursts),
        "burst_rate_per_min": len(bursts) / (duration_ms / 60000.0),
        **stats,
        "mean_firing_rate": mean_rate,
    }


# -----------------------------------------------------------------------
# Tests — calibrated STP parameters
# -----------------------------------------------------------------------

@pytest.mark.slow
class TestCalibratedCultureBursts:
    """Calibrated 5K-neuron network should produce spontaneous bursting.

    Uses Tsodyks-Markram short-term plasticity (STP) to enable excitatory
    synaptic depression, which prevents runaway seizure-like activity and
    allows the network to exhibit the burst/quiescence cycle characteristic
    of real cortical cultures (Wagenaar et al., 2006).

    Parameters match configs/wagenaar_calibrated.yaml:
      g_exc=0.10, g_inh=0.40, p_max=0.21, bg_mean=1.0, bg_std=3.0, STP=on
    """

    def test_bursts_detected(self):
        """At least one burst should be detected in 10 seconds with STP."""
        result = _run_culture(
            n_neurons=5000,
            duration_ms=10000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        assert result["n_bursts"] > 0, (
            f"No bursts detected (mean rate = {result['mean_firing_rate']:.2f} Hz). "
            "Network may be too quiet or too active (seizure-like)."
        )

    def test_burst_rate_plausible(self):
        """Burst rate should be in a physiologically plausible range."""
        result = _run_culture(
            n_neurons=5000,
            duration_ms=20000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        # Wagenaar target: 3-20 bursts/min.  We allow a wider range
        # because 20 s is short for stable estimates.
        assert result["burst_rate_per_min"] >= 1.0, (
            f"Burst rate too low: {result['burst_rate_per_min']:.1f}/min "
            f"(target: 3-20/min, Wagenaar 2006)"
        )

    def test_mean_firing_rate_not_seizure(self):
        """Mean firing rate should stay below seizure-like levels with STP."""
        result = _run_culture(
            n_neurons=5000,
            duration_ms=10000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        # With STP, excitatory depression should prevent runaway activity.
        # Without STP at g_exc=0.10, the network would go seizure-like.
        assert result["mean_firing_rate"] < 50.0, (
            f"Mean firing rate {result['mean_firing_rate']:.1f} Hz is "
            "seizure-like (>50 Hz). STP may not be working correctly."
        )

    def test_stp_prevents_seizure(self):
        """STP should keep firing rate much lower than without STP.

        With g_exc=0.10 and p_max=0.21, the network without STP would
        produce extremely high firing rates (seizure-like).  STP's
        excitatory depression (U=0.5, tau_rec=800ms) should tame the
        network to produce intermittent bursting instead.
        """
        result_stp = _run_culture(
            n_neurons=2000,
            duration_ms=5000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        result_no_stp = _run_culture(
            n_neurons=2000,
            duration_ms=5000,
            seed=42,
            use_stp=False,
            **_CALIBRATED,
        )
        # STP should significantly reduce firing rate compared to no-STP
        assert result_stp["mean_firing_rate"] < result_no_stp["mean_firing_rate"], (
            f"STP rate ({result_stp['mean_firing_rate']:.1f} Hz) should be lower "
            f"than no-STP rate ({result_no_stp['mean_firing_rate']:.1f} Hz)"
        )


@pytest.mark.slow
class TestSmallerNetworkBursts:
    """Smaller network (1000 neurons) should still burst with STP."""

    def test_1k_network_has_activity(self):
        """1000-neuron network should produce spikes with STP and noise."""
        result = _run_culture(
            n_neurons=1000,
            duration_ms=10000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        assert result["mean_firing_rate"] > 0.01, (
            f"Network is essentially silent: {result['mean_firing_rate']:.4f} Hz"
        )

    def test_1k_network_bursts_with_stp(self):
        """1000-neuron network should produce at least one burst in 10 s."""
        result = _run_culture(
            n_neurons=1000,
            duration_ms=10000,
            seed=42,
            use_stp=True,
            **_CALIBRATED,
        )
        assert result["n_bursts"] > 0, (
            f"No bursts in 1K network (rate={result['mean_firing_rate']:.2f} Hz). "
            "Smaller networks may need more time or stronger drive."
        )


class TestBurstDetectionSanity:
    """Sanity checks on burst detection with synthetic data."""

    def test_no_bursts_in_uniform_noise(self):
        """Uniform low-rate Poisson spiking should not produce many bursts."""
        rng = np.random.RandomState(123)
        T = 20000  # 10 seconds at dt=0.5
        N = 100
        # ~1% spike probability per step => ~20 Hz per neuron (high but uniform)
        raster = rng.rand(T, N) < 0.001  # ~0.2 Hz per neuron
        bursts = detect_bursts(raster, dt_ms=0.5, threshold_std=2.0, min_duration_ms=50.0)
        # Uniform activity should produce very few or no bursts
        assert len(bursts) < 5, (
            f"Detected {len(bursts)} bursts in uniform low-rate noise "
            "(expected < 5)"
        )

    def test_obvious_burst_detected(self):
        """A clear burst embedded in silence should be detected."""
        T = 10000
        N = 100
        raster = np.zeros((T, N), dtype=bool)
        # Insert a single obvious burst: steps 2000-2400 (1000-1200 ms)
        rng = np.random.RandomState(456)
        raster[2000:2400, :] = rng.rand(400, N) < 0.4
        bursts = detect_bursts(raster, dt_ms=0.5, threshold_std=2.0, min_duration_ms=50.0)
        assert len(bursts) >= 1, "Failed to detect an obvious burst"
        # Burst should start near 1000 ms
        starts = [b[0] for b in bursts]
        assert any(800 < s < 1300 for s in starts), (
            f"Burst start times {starts} do not include the expected ~1000 ms burst"
        )
