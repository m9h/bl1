# %% [markdown]
# # Notebook 2: Network Bursting in Simulated Cortical Culture
#
# Characterises spontaneous synchronous bursting in a simulated cortical
# culture, a hallmark of dissociated neuronal networks grown on MEAs
# (Wagenaar et al. 2006, Kamioka et al. 1996).
#
# We build a 10,000-neuron culture with distance-dependent connectivity,
# drive it with small random background current (to mimic channel noise and
# spontaneous synaptic release), and analyse the resulting burst dynamics.
#
# Key questions:
# - Does the network produce synchronous population bursts?
# - What are the burst statistics (IBI, duration, recruitment)?
# - How does the E/I ratio affect burst rate?

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bl1.network.types import Culture
from bl1.core.izhikevich import IzhikevichParams, NeuronState, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    gaba_b_step,
    nmda_step,
    compute_synaptic_current,
)
from bl1.core.integrator import simulate, SimulationResult
from bl1.analysis.bursts import detect_bursts, burst_statistics

print("JAX devices:", jax.devices())
key = jax.random.PRNGKey(42)

# %% [markdown]
# ## Build Network
#
# `Culture.create` handles neuron placement on a 3x3 mm substrate,
# Izhikevich parameter assignment (80% excitatory: RS/IB/CH, 20%
# inhibitory: FS/LTS), and distance-dependent connectivity where
# connection probability decays exponentially with inter-somatic distance
# (lambda = 200 um, p_max = 0.21).

# %%
%%time

N_NEURONS = 10_000
key, k_culture = jax.random.split(key)

net_params, culture_state, izh_params = Culture.create(
    k_culture,
    n_neurons=N_NEURONS,
    ei_ratio=0.8,
)

# --- Connectivity statistics ---
W_exc = net_params.W_exc
W_inh = net_params.W_inh

n_exc_synapses = int(W_exc.data.shape[0])
n_inh_synapses = int(W_inh.data.shape[0])
n_exc_neurons = int(jnp.sum(net_params.is_excitatory))
n_inh_neurons = N_NEURONS - n_exc_neurons

print(f"Network: {N_NEURONS} neurons ({n_exc_neurons} E, {n_inh_neurons} I)")
print(f"Excitatory synapses: {n_exc_synapses:,} "
      f"({n_exc_synapses / N_NEURONS:.1f} per neuron)")
print(f"Inhibitory synapses: {n_inh_synapses:,} "
      f"({n_inh_synapses / N_NEURONS:.1f} per neuron)")
print(f"Connection density: {(n_exc_synapses + n_inh_synapses) / (N_NEURONS**2) * 100:.2f}%")

# %% [markdown]
# ## Simulate Spontaneous Activity
#
# We run 5 seconds of simulation with small Gaussian background noise
# (mean=0, std=3) to mimic the stochastic fluctuations that drive
# spontaneous activity in real cultures.  The noise level is chosen to be
# subthreshold for isolated neurons but sufficient to occasionally push
# neurons past threshold when combined with recurrent synaptic input.
#
# The simulation uses `jax.lax.scan` under the hood for efficient
# GPU-accelerated time stepping at dt=0.5 ms.

# %%
%%time

DURATION_MS = 5000.0
DT = 0.5
n_steps = int(DURATION_MS / DT)

# Generate background noise current: I ~ N(0, 3)
key, k_noise = jax.random.split(key)
I_external = 3.0 * jax.random.normal(k_noise, shape=(n_steps, N_NEURONS))

# Initial neuron state
v0 = jnp.full(N_NEURONS, -65.0)
u0 = izh_params.b * v0
init_state = NeuronState(
    v=v0,
    u=u0,
    spikes=jnp.zeros(N_NEURONS, dtype=jnp.bool_),
)

# Initial synapse state
syn_state = create_synapse_state(N_NEURONS)

# For the integrator, we need dense weight matrices.
# Convert BCOO to dense for the simulate function.
W_exc_dense = W_exc.todense()
W_inh_dense = W_inh.todense()

# Run simulation (no plasticity for spontaneous activity analysis)
result = simulate(
    params=izh_params,
    init_state=init_state,
    syn_state=syn_state,
    stdp_state=None,
    W_exc=W_exc_dense,
    W_inh=W_inh_dense,
    I_external=I_external,
    dt=DT,
    plasticity_fn=None,
)

spike_raster = np.asarray(result.spike_history)
total_spikes = spike_raster.sum()
print(f"Simulation complete: {n_steps} steps, {total_spikes:,} total spikes")
print(f"Mean firing rate: {total_spikes / (N_NEURONS * DURATION_MS / 1000):.2f} Hz")

# %% [markdown]
# ## Raster Plot
#
# A spike raster shows each neuron's spike times as a dot.  For
# visibility we subsample 500 neurons.  Synchronous population bursts
# appear as vertical bands of dense spiking across many neurons
# simultaneously.

# %%
# Subsample neurons for visibility
n_display = 500
key, k_sub = jax.random.split(key)
display_idx = np.sort(np.random.default_rng(42).choice(N_NEURONS, n_display, replace=False))

# Downsample in time (every 2 steps = 1 ms resolution) for plotting speed
ds = 2
raster_ds = spike_raster[::ds, :][:, display_idx]
t_ds = np.arange(raster_ds.shape[0]) * DT * ds / 1000  # seconds

fig, ax = plt.subplots(figsize=(12, 5))
spike_t, spike_n = np.where(raster_ds)
ax.scatter(t_ds[spike_t], spike_n, s=0.1, c="black", alpha=0.5, rasterized=True)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron index (subsampled)")
ax.set_title(f"Spike Raster ({n_display} of {N_NEURONS} neurons)")
ax.set_xlim(0, DURATION_MS / 1000)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Population Firing Rate
#
# The instantaneous population firing rate is the total number of spikes
# per time bin, smoothed with a Gaussian kernel.  Population bursts appear
# as sharp peaks in this trace.

# %%
# Compute population spike count per ms
bin_ms = 1.0
steps_per_bin = int(bin_ms / DT)
n_bins = spike_raster.shape[0] // steps_per_bin
pop_rate = spike_raster[:n_bins * steps_per_bin].reshape(n_bins, steps_per_bin, -1).sum(axis=(1, 2))

# Smooth with Gaussian kernel (sigma=10 ms)
from scipy.ndimage import gaussian_filter1d
pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float), sigma=10)

t_rate = np.arange(n_bins) * bin_ms / 1000  # seconds

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t_rate, pop_rate_smooth, "k-", linewidth=0.5)
ax.fill_between(t_rate, pop_rate_smooth, alpha=0.3, color="steelblue")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Spikes / ms")
ax.set_title("Population Firing Rate (Gaussian smoothed, sigma=10 ms)")
ax.set_xlim(0, DURATION_MS / 1000)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Burst Detection
#
# We use threshold-crossing on the population spike count to detect
# bursts.  A burst begins when the rate exceeds mean + 2*std and ends
# when it drops below the mean.  Only bursts lasting at least 50 ms are
# retained.

# %%
bursts = detect_bursts(spike_raster, dt_ms=DT, threshold_std=2.0, min_duration_ms=50.0)

print(f"Detected {len(bursts)} bursts in {DURATION_MS/1000:.0f} s")

# Plot burst onsets/offsets on the rate trace
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t_rate, pop_rate_smooth, "k-", linewidth=0.5)
ax.fill_between(t_rate, pop_rate_smooth, alpha=0.2, color="steelblue")

for start_ms, end_ms, n_spk, frac in bursts:
    ax.axvspan(start_ms / 1000, end_ms / 1000, alpha=0.3, color="red",
               label="Burst" if start_ms == bursts[0][0] else None)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Spikes / ms")
ax.set_title(f"Detected Bursts (n={len(bursts)}, red shading)")
ax.set_xlim(0, DURATION_MS / 1000)
if bursts:
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Print individual burst details
if bursts:
    print(f"\n{'Burst':<7} {'Start (s)':>10} {'End (s)':>10} {'Duration (ms)':>14} "
          f"{'Spikes':>8} {'Recruitment':>12}")
    print("-" * 70)
    for i, (s, e, n_spk, frac) in enumerate(bursts):
        print(f"{i+1:<7} {s/1000:>10.3f} {e/1000:>10.3f} {e-s:>14.1f} "
              f"{n_spk:>8} {frac:>12.2%}")

# %% [markdown]
# ## Burst Statistics
#
# Summary statistics from the detected bursts, including inter-burst
# interval (IBI), duration, and recruitment fraction.

# %%
stats = burst_statistics(bursts)

print("Burst Statistics:")
print(f"  IBI mean:            {stats['ibi_mean']:.1f} ms ({stats['ibi_mean']/1000:.2f} s)")
print(f"  IBI CV:              {stats['ibi_cv']:.3f}")
print(f"  Duration mean:       {stats['duration_mean']:.1f} ms")
print(f"  Recruitment mean:    {stats['recruitment_mean']:.2%}")
print(f"  Burst rate:          {stats['burst_rate']:.2f} Hz")

# %% [markdown]
# ## Comparison with Published Data
#
# Wagenaar et al. (2006) report the following for mature dissociated
# cortical cultures (DIV 14-28):
#
# | Metric          | Published Range   | Our Simulation    |
# |-----------------|-------------------|-------------------|
# | IBI             | 5-30 s            | see above         |
# | Burst duration  | 200-500 ms        | see above         |
# | Recruitment     | 20-80%            | see above         |
# | Burst rate      | 0.03-0.2 Hz       | see above         |
#
# Our 10K neuron network with default parameters should show bursting
# behaviour, though the exact statistics may differ from published values
# due to (1) smaller network size, (2) simplified synapse kinetics
# compared to real cultures, and (3) absence of intrinsic noise channels.
# The key qualitative feature -- synchronous population bursts -- should
# be present.

# %% [markdown]
# ## E/I Ratio Effect on Bursting
#
# The excitatory/inhibitory balance is a key determinant of network
# dynamics.  More inhibition (lower E/I ratio) should suppress bursting
# by increasing the effective threshold for regenerative excitatory
# cascades.  Here we compare three E/I ratios.

# %%
%%time

ei_ratios = [0.7, 0.8, 0.9]
results_ei = {}

for ei in ei_ratios:
    key, k_c, k_n = jax.random.split(key, 3)

    net_p, _, izh_p = Culture.create(k_c, n_neurons=N_NEURONS, ei_ratio=ei)

    # Build dense weight matrices
    W_e = net_p.W_exc.todense()
    W_i = net_p.W_inh.todense()

    # Background noise
    I_ext = 3.0 * jax.random.normal(k_n, shape=(n_steps, N_NEURONS))

    v0 = jnp.full(N_NEURONS, -65.0)
    u0 = izh_p.b * v0
    init_s = NeuronState(v=v0, u=u0, spikes=jnp.zeros(N_NEURONS, dtype=jnp.bool_))
    syn_s = create_synapse_state(N_NEURONS)

    res = simulate(
        params=izh_p,
        init_state=init_s,
        syn_state=syn_s,
        stdp_state=None,
        W_exc=W_e,
        W_inh=W_i,
        I_external=I_ext,
        dt=DT,
        plasticity_fn=None,
    )

    raster = np.asarray(res.spike_history)
    bs = detect_bursts(raster, dt_ms=DT, threshold_std=2.0, min_duration_ms=50.0)
    stats_ei = burst_statistics(bs)
    n_exc = int(jnp.sum(net_p.is_excitatory))
    results_ei[ei] = {
        "n_bursts": len(bs),
        "burst_rate": stats_ei["burst_rate"],
        "ibi_mean": stats_ei["ibi_mean"],
        "mean_rate_hz": raster.sum() / (N_NEURONS * DURATION_MS / 1000),
        "n_exc": n_exc,
        "n_inh": N_NEURONS - n_exc,
    }
    print(f"E/I ratio {ei}: {len(bs)} bursts, "
          f"rate={stats_ei['burst_rate']:.3f} Hz, "
          f"mean firing={results_ei[ei]['mean_rate_hz']:.2f} Hz "
          f"({n_exc}E / {N_NEURONS - n_exc}I)")

# %%
# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Burst count
ax = axes[0]
ax.bar([str(e) for e in ei_ratios], [results_ei[e]["n_bursts"] for e in ei_ratios],
       color=["#4CAF50", "#2196F3", "#F44336"])
ax.set_xlabel("E/I Ratio")
ax.set_ylabel("Number of Bursts")
ax.set_title("Burst Count (5 s)")

# Burst rate
ax = axes[1]
ax.bar([str(e) for e in ei_ratios], [results_ei[e]["burst_rate"] for e in ei_ratios],
       color=["#4CAF50", "#2196F3", "#F44336"])
ax.set_xlabel("E/I Ratio")
ax.set_ylabel("Burst Rate (Hz)")
ax.set_title("Burst Rate")

# Mean firing rate
ax = axes[2]
ax.bar([str(e) for e in ei_ratios], [results_ei[e]["mean_rate_hz"] for e in ei_ratios],
       color=["#4CAF50", "#2196F3", "#F44336"])
ax.set_xlabel("E/I Ratio")
ax.set_ylabel("Mean Firing Rate (Hz)")
ax.set_title("Population Firing Rate")

fig.suptitle("Effect of E/I Ratio on Network Bursting", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Observations
#
# 1. **Spontaneous bursting** emerges from the interaction of excitatory
#    recurrence and background noise, even without structured external
#    input -- matching the behaviour of real dissociated cultures.
#
# 2. **Burst statistics** should fall in a biologically plausible range,
#    though exact values depend on network size and parameter tuning.
#
# 3. **E/I ratio effect**: Higher E/I ratios (more excitation, less
#    inhibition) should produce more frequent and larger bursts.  Lower
#    E/I ratios (more inhibition) should suppress bursting and reduce
#    overall firing rates.  This recapitulates the well-known role of
#    GABAergic inhibition in regulating cortical excitability.
