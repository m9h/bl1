# %% [markdown]
# # Notebook 4: Criticality and Avalanche Dynamics
#
# Analyses whether the simulated cortical culture operates near the
# critical point -- a dynamical regime characterised by a branching ratio
# sigma ~ 1.0 and power-law distributed avalanche sizes and durations.
#
# Criticality is considered a hallmark of healthy cortical networks
# (Beggs & Plenz 2003, 2004) and confers optimal information processing
# capabilities, maximal dynamic range, and the ability to transition
# rapidly between activity states.
#
# Key analyses:
# 1. Branching ratio across bin sizes
# 2. Avalanche size distribution (expected power law with alpha ~ -1.5)
# 3. Avalanche duration distribution (expected power law with beta ~ -2.0)
# 4. Size-duration scaling (crackling noise relation)
# 5. Pharmacological perturbation (bicuculline = GABA_A block -> supercritical)

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter1d

from bl1.network.types import Culture
from bl1.core.izhikevich import IzhikevichParams, NeuronState
from bl1.core.synapses import create_synapse_state
from bl1.core.integrator import simulate
from bl1.analysis.criticality import branching_ratio, avalanche_size_distribution

print("JAX devices:", jax.devices())
key = jax.random.PRNGKey(42)

# %% [markdown]
# ## Build Network and Simulate Spontaneous Activity
#
# We create a 10,000-neuron culture and simulate 5 seconds of spontaneous
# activity driven by background noise.  The noise amplitude is tuned to
# produce moderate firing rates (~1-5 Hz mean) consistent with the
# subcritical-to-critical regime.

# %%
%%time

N_NEURONS = 10_000
DURATION_MS = 5000.0
DT = 0.5
n_steps = int(DURATION_MS / DT)

key, k_culture, k_noise = jax.random.split(key, 3)

net_params, culture_state, izh_params = Culture.create(
    k_culture,
    n_neurons=N_NEURONS,
    ei_ratio=0.8,
)

# Dense weight matrices for the integrator
W_exc_dense = net_params.W_exc.todense()
W_inh_dense = net_params.W_inh.todense()

# Background noise current
I_external = 3.0 * jax.random.normal(k_noise, shape=(n_steps, N_NEURONS))

# Initial state
v0 = jnp.full(N_NEURONS, -65.0)
u0 = izh_params.b * v0
init_state = NeuronState(v=v0, u=u0, spikes=jnp.zeros(N_NEURONS, dtype=jnp.bool_))
syn_state = create_synapse_state(N_NEURONS)

# Simulate
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
mean_rate = total_spikes / (N_NEURONS * DURATION_MS / 1000)
print(f"Simulation complete: {total_spikes:,} spikes, mean rate = {mean_rate:.2f} Hz")

# %% [markdown]
# ## Branching Ratio
#
# The branching ratio sigma is the average ratio of activity in
# consecutive time bins:
#
#     sigma = <n_spikes[t+1] / n_spikes[t]>
#
# - sigma ~ 1.0: critical (balanced propagation)
# - sigma < 1.0: subcritical (activity dies out)
# - sigma > 1.0: supercritical (runaway excitation)
#
# We compute sigma for multiple bin sizes (2, 4, 8, 16 ms) as the
# branching ratio is sensitive to temporal resolution.

# %%
%%time

bin_sizes_ms = [2.0, 4.0, 8.0, 16.0]
sigmas = []

for bin_ms in bin_sizes_ms:
    sigma = branching_ratio(spike_raster, dt_ms=DT, bin_ms=bin_ms)
    sigmas.append(sigma)
    print(f"Bin size {bin_ms:5.1f} ms: sigma = {sigma:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bin_sizes_ms, sigmas, "ko-", markersize=8, linewidth=2)
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Critical (sigma=1)")
ax.fill_between([0, 20], 0.9, 1.1, color="red", alpha=0.1, label="Near-critical zone")
ax.set_xlabel("Bin Size (ms)")
ax.set_ylabel("Branching Ratio (sigma)")
ax.set_title("Branching Ratio vs. Temporal Bin Size")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(bin_sizes_ms) + 2)
ax.set_ylim(0, max(max(sigmas) * 1.3, 1.5))

for bs, s in zip(bin_sizes_ms, sigmas):
    ax.annotate(f"{s:.3f}", (bs, s), textcoords="offset points",
                xytext=(10, 10), fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Avalanche Detection
#
# A neuronal avalanche is a contiguous sequence of active time bins
# (at least one spike per bin) bounded by silent bins.  We extract
# avalanche sizes (total spikes) and durations (number of active bins)
# using a standard 4 ms bin width.

# %%
%%time

BIN_MS = 4.0
sizes, durations = avalanche_size_distribution(spike_raster, dt_ms=DT, bin_ms=BIN_MS)

print(f"Detected {len(sizes)} avalanches (bin={BIN_MS} ms)")
if len(sizes) > 0:
    print(f"  Size range: [{sizes.min():.0f}, {sizes.max():.0f}]")
    print(f"  Duration range: [{durations.min()}, {durations.max()}] bins "
          f"([{durations.min() * BIN_MS:.0f}, {durations.max() * BIN_MS:.0f}] ms)")
    print(f"  Mean size: {sizes.mean():.1f}, Median size: {np.median(sizes):.1f}")
    print(f"  Mean duration: {durations.mean():.1f} bins "
          f"({durations.mean() * BIN_MS:.1f} ms)")

# %% [markdown]
# ## Avalanche Size Distribution
#
# At criticality, the avalanche size distribution follows a power law:
#
#     P(s) ~ s^alpha,   alpha ~ -1.5 (mean-field universality class)
#
# We plot the distribution on log-log axes and fit a power law exponent
# using maximum likelihood estimation on the tail (sizes > 10).

# %%
fig, ax = plt.subplots(figsize=(8, 6))

if len(sizes) > 10:
    # Log-binned histogram for cleaner log-log plots
    log_bins = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 30)
    counts, bin_edges = np.histogram(sizes, bins=log_bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    # Normalise to probability density
    bin_widths = np.diff(bin_edges)
    pdf = counts / (counts.sum() * bin_widths)

    # Filter out zero bins for log-log
    mask = pdf > 0
    ax.scatter(bin_centers[mask], pdf[mask], c="steelblue", s=30, alpha=0.7,
               label="Data", zorder=3)

    # Power law fit on log-log scale (sizes > threshold)
    fit_threshold = max(10, np.percentile(sizes, 25))
    fit_sizes = sizes[sizes >= fit_threshold]
    if len(fit_sizes) > 20:
        # MLE power law exponent: alpha_hat = 1 + n / sum(log(x/x_min))
        x_min = fit_threshold
        n = len(fit_sizes)
        alpha_mle = 1 + n / np.sum(np.log(fit_sizes / x_min))
        print(f"Power law fit (MLE, x_min={x_min:.0f}):")
        print(f"  alpha = {-alpha_mle:.3f} (expected ~ -1.5)")

        # Plot fitted power law
        x_fit = np.logspace(np.log10(x_min), np.log10(sizes.max()), 50)
        # P(s) ~ s^(-alpha) => normalise so it passes through data
        y_fit = x_fit ** (-alpha_mle)
        # Scale to match the data magnitude
        scale_idx = np.argmin(np.abs(bin_centers[mask] - x_min * 2))
        if scale_idx < len(pdf[mask]) and pdf[mask][scale_idx] > 0:
            y_fit *= pdf[mask][scale_idx] / (x_min * 2) ** (-alpha_mle)
        ax.plot(x_fit, y_fit, "r--", linewidth=2,
                label=f"Power law fit: alpha={-alpha_mle:.2f}")
        ax.axvline(x_min, color="gray", linestyle=":", alpha=0.5,
                   label=f"Fit threshold: {x_min:.0f}")

    # Reference line for alpha = -1.5
    x_ref = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50)
    y_ref = x_ref ** (-1.5)
    y_ref *= pdf[mask][len(pdf[mask])//3] / x_ref[len(x_ref)//3] ** (-1.5)
    ax.plot(x_ref, y_ref, "k:", alpha=0.3, label="Reference: alpha=-1.5")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Avalanche Size (total spikes)")
ax.set_ylabel("Probability Density")
ax.set_title("Avalanche Size Distribution")
ax.legend()
ax.grid(True, alpha=0.2, which="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Avalanche Duration Distribution
#
# The duration distribution should also follow a power law at criticality:
#
#     P(T) ~ T^beta,   beta ~ -2.0 (mean-field theory)

# %%
fig, ax = plt.subplots(figsize=(8, 6))

if len(durations) > 10:
    unique_durations, dur_counts = np.unique(durations, return_counts=True)
    dur_pdf = dur_counts / dur_counts.sum()

    ax.scatter(unique_durations, dur_pdf, c="darkorange", s=30, alpha=0.7,
               label="Data", zorder=3)

    # Power law fit (MLE) for durations > 2
    fit_durations = durations[durations >= 2]
    if len(fit_durations) > 20:
        d_min = 2
        n_d = len(fit_durations)
        beta_mle = 1 + n_d / np.sum(np.log(fit_durations / d_min))
        print(f"Duration power law fit (MLE, T_min={d_min}):")
        print(f"  beta = {-beta_mle:.3f} (expected ~ -2.0)")

        # Plot fitted line
        t_fit = np.logspace(np.log10(d_min), np.log10(durations.max()), 50)
        y_fit = t_fit ** (-beta_mle)
        # Scale
        ref_idx = np.argmin(np.abs(unique_durations - d_min))
        if ref_idx < len(dur_pdf) and dur_pdf[ref_idx] > 0:
            y_fit *= dur_pdf[ref_idx] / d_min ** (-beta_mle)
        ax.plot(t_fit, y_fit, "r--", linewidth=2,
                label=f"Power law fit: beta={-beta_mle:.2f}")

    # Reference line for beta = -2.0
    t_ref = np.logspace(0, np.log10(durations.max()), 50)
    y_ref = t_ref ** (-2.0)
    y_ref *= dur_pdf[0] / 1.0 ** (-2.0) * 0.5
    ax.plot(t_ref, y_ref, "k:", alpha=0.3, label="Reference: beta=-2.0")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"Avalanche Duration (bins, 1 bin = {BIN_MS} ms)")
ax.set_ylabel("Probability")
ax.set_title("Avalanche Duration Distribution")
ax.legend()
ax.grid(True, alpha=0.2, which="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Size vs. Duration Scaling (Crackling Noise Relation)
#
# At criticality, mean avalanche size scales with duration as:
#
#     <S>(T) ~ T^gamma
#
# The crackling noise relation connects the three exponents:
#
#     gamma = (alpha - 1) / (beta - 1)
#
# For mean-field criticality: gamma = (1.5 - 1) / (2.0 - 1) = 0.5 / 1.0 = 1.5
# (i.e., <S> ~ T^1.5 -- though in 2D cortical cultures the exponent may
# differ due to the dimensionality of the system).

# %%
fig, ax = plt.subplots(figsize=(8, 6))

if len(sizes) > 10 and len(durations) > 10:
    unique_durs = np.unique(durations)
    mean_sizes = []
    valid_durs = []

    for d in unique_durs:
        mask = durations == d
        if mask.sum() >= 3:  # require at least 3 avalanches per duration
            mean_sizes.append(np.mean(sizes[mask]))
            valid_durs.append(d)

    valid_durs = np.array(valid_durs)
    mean_sizes = np.array(mean_sizes)

    if len(valid_durs) > 3:
        ax.scatter(valid_durs, mean_sizes, c="purple", s=40, alpha=0.7,
                   label="Data", zorder=3)

        # Log-log linear fit
        log_d = np.log10(valid_durs.astype(float))
        log_s = np.log10(mean_sizes)
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_d, log_s)
        print(f"Size-duration scaling:")
        print(f"  gamma = {slope:.3f} (expected ~ 1.5 for mean-field)")
        print(f"  R^2 = {r_value**2:.4f}")

        # Plot fit
        d_fit = np.logspace(np.log10(valid_durs.min()),
                            np.log10(valid_durs.max()), 50)
        s_fit = 10**(intercept) * d_fit ** slope
        ax.plot(d_fit, s_fit, "r--", linewidth=2,
                label=f"Fit: gamma={slope:.2f} (R2={r_value**2:.3f})")

        # Reference gamma = 1.5
        s_ref = 10**(intercept) * d_fit ** 1.5
        ax.plot(d_fit, s_ref, "k:", alpha=0.3, label="Reference: gamma=1.5")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"Avalanche Duration (bins, 1 bin = {BIN_MS} ms)")
ax.set_ylabel("Mean Avalanche Size")
ax.set_title("Avalanche Size vs. Duration Scaling")
ax.legend()
ax.grid(True, alpha=0.2, which="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pharmacological Perturbation: Bicuculline (GABA_A Block)
#
# Bicuculline is a GABA_A receptor antagonist that blocks inhibitory
# synaptic transmission.  In real cultures, bicuculline application
# shifts the network from near-critical to supercritical dynamics:
# the branching ratio increases above 1, avalanche distributions deviate
# from power laws, and large runaway bursts dominate.
#
# We simulate this by zeroing the inhibitory weight matrix (W_inh = 0)
# and comparing the resulting branching ratio to the control condition.

# %%
%%time

print("Running bicuculline (GABA_A block) simulation...")

key, k_bic_noise = jax.random.split(key)
I_external_bic = 3.0 * jax.random.normal(k_bic_noise, shape=(n_steps, N_NEURONS))

# Zero out inhibitory weights (simulating complete GABA_A block)
W_inh_blocked = jnp.zeros_like(W_inh_dense)

# Fresh initial state
init_state_bic = NeuronState(
    v=jnp.full(N_NEURONS, -65.0),
    u=izh_params.b * jnp.full(N_NEURONS, -65.0),
    spikes=jnp.zeros(N_NEURONS, dtype=jnp.bool_),
)
syn_state_bic = create_synapse_state(N_NEURONS)

result_bic = simulate(
    params=izh_params,
    init_state=init_state_bic,
    syn_state=syn_state_bic,
    stdp_state=None,
    W_exc=W_exc_dense,
    W_inh=W_inh_blocked,
    I_external=I_external_bic,
    dt=DT,
    plasticity_fn=None,
)

spike_raster_bic = np.asarray(result_bic.spike_history)
total_spikes_bic = spike_raster_bic.sum()
mean_rate_bic = total_spikes_bic / (N_NEURONS * DURATION_MS / 1000)

print(f"Bicuculline: {total_spikes_bic:,} spikes, mean rate = {mean_rate_bic:.2f} Hz")

# Compute branching ratios for both conditions
print("\nBranching ratio comparison:")
print(f"{'Bin (ms)':<10} {'Control':>10} {'Bicuculline':>12} {'Delta':>8}")
print("-" * 42)

sigmas_bic = []
for bin_ms in bin_sizes_ms:
    sigma_ctrl = branching_ratio(spike_raster, dt_ms=DT, bin_ms=bin_ms)
    sigma_bic = branching_ratio(spike_raster_bic, dt_ms=DT, bin_ms=bin_ms)
    sigmas_bic.append(sigma_bic)
    delta = sigma_bic - sigma_ctrl
    print(f"{bin_ms:<10.1f} {sigma_ctrl:>10.4f} {sigma_bic:>12.4f} {delta:>+8.4f}")

# %%
# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Branching ratio comparison
ax = axes[0]
ax.plot(bin_sizes_ms, sigmas, "bo-", markersize=8, linewidth=2, label="Control")
ax.plot(bin_sizes_ms, sigmas_bic, "rs-", markersize=8, linewidth=2, label="Bicuculline")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Critical")
ax.fill_between([0, 20], 0.9, 1.1, color="gray", alpha=0.1)
ax.set_xlabel("Bin Size (ms)")
ax.set_ylabel("Branching Ratio (sigma)")
ax.set_title("Branching Ratio: Control vs. Bicuculline")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(bin_sizes_ms) + 2)

# Raster comparison (first 2 seconds, 500 neurons)
n_display = 500
display_idx = np.sort(np.random.default_rng(42).choice(N_NEURONS, n_display, replace=False))
t_limit_steps = int(2000 / DT)

ax = axes[1]
ds = 2
raster_ctrl_ds = spike_raster[:t_limit_steps:ds, :][:, display_idx]
spike_t, spike_n = np.where(raster_ctrl_ds)
t_ds = np.arange(raster_ctrl_ds.shape[0]) * DT * ds / 1000
ax.scatter(t_ds[spike_t], spike_n, s=0.1, c="blue", alpha=0.3, rasterized=True)
ax.set_ylabel("Neuron")
ax.set_title("Control (first 2 s)")
ax.set_xlim(0, 2)

ax = axes[2]
raster_bic_ds = spike_raster_bic[:t_limit_steps:ds, :][:, display_idx]
spike_t, spike_n = np.where(raster_bic_ds)
t_ds = np.arange(raster_bic_ds.shape[0]) * DT * ds / 1000
ax.scatter(t_ds[spike_t], spike_n, s=0.1, c="red", alpha=0.3, rasterized=True)
ax.set_ylabel("Neuron")
ax.set_title("Bicuculline (first 2 s)")
ax.set_xlim(0, 2)

fig.suptitle("Effect of GABA_A Blockade on Network Dynamics", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Avalanche distributions comparison
sizes_bic, durations_bic = avalanche_size_distribution(
    spike_raster_bic, dt_ms=DT, bin_ms=BIN_MS
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Size distributions
ax = axes[0]
if len(sizes) > 5:
    log_bins = np.logspace(0, np.log10(max(sizes.max(), sizes_bic.max() if len(sizes_bic) > 0 else 1) + 1), 30)
    counts_ctrl, _ = np.histogram(sizes, bins=log_bins)
    pdf_ctrl = counts_ctrl / counts_ctrl.sum()
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])
    mask_c = pdf_ctrl > 0
    ax.scatter(bin_centers[mask_c], pdf_ctrl[mask_c], c="blue", s=20, alpha=0.6, label="Control")

if len(sizes_bic) > 5:
    counts_bic, _ = np.histogram(sizes_bic, bins=log_bins)
    pdf_bic = counts_bic / counts_bic.sum()
    mask_b = pdf_bic > 0
    ax.scatter(bin_centers[mask_b], pdf_bic[mask_b], c="red", s=20, alpha=0.6, label="Bicuculline")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Avalanche Size")
ax.set_ylabel("Probability")
ax.set_title("Avalanche Size Distribution")
ax.legend()
ax.grid(True, alpha=0.2, which="both")

# Duration distributions
ax = axes[1]
if len(durations) > 5:
    ud, uc = np.unique(durations, return_counts=True)
    ax.scatter(ud, uc / uc.sum(), c="blue", s=20, alpha=0.6, label="Control")

if len(durations_bic) > 5:
    ud_b, uc_b = np.unique(durations_bic, return_counts=True)
    ax.scatter(ud_b, uc_b / uc_b.sum(), c="red", s=20, alpha=0.6, label="Bicuculline")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Avalanche Duration (bins)")
ax.set_ylabel("Probability")
ax.set_title("Avalanche Duration Distribution")
ax.legend()
ax.grid(True, alpha=0.2, which="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary of Criticality Metrics
#
# A concise table comparing the control and bicuculline conditions
# across all criticality measures.

# %%
# Compute summary metrics
def compute_power_law_exponent(values, x_min):
    """MLE power law exponent."""
    filtered = values[values >= x_min]
    if len(filtered) < 10:
        return float("nan")
    n = len(filtered)
    return -(1 + n / np.sum(np.log(filtered / x_min)))


# Control metrics
sigma_ctrl_4ms = branching_ratio(spike_raster, dt_ms=DT, bin_ms=4.0)
alpha_ctrl = compute_power_law_exponent(sizes, max(10, np.percentile(sizes, 25))) if len(sizes) > 10 else float("nan")
beta_ctrl = compute_power_law_exponent(durations.astype(float), 2.0) if len(durations) > 10 else float("nan")

# Bicuculline metrics
sigma_bic_4ms = branching_ratio(spike_raster_bic, dt_ms=DT, bin_ms=4.0)
alpha_bic = compute_power_law_exponent(sizes_bic, max(10, np.percentile(sizes_bic, 25)) if len(sizes_bic) > 10 else 10) if len(sizes_bic) > 10 else float("nan")
beta_bic = compute_power_law_exponent(durations_bic.astype(float), 2.0) if len(durations_bic) > 10 else float("nan")

print("=" * 65)
print(f"{'Metric':<25} {'Theory':>10} {'Control':>12} {'Bicuculline':>12}")
print("=" * 65)
print(f"{'Branching ratio (4ms)':<25} {'~1.0':>10} {sigma_ctrl_4ms:>12.4f} {sigma_bic_4ms:>12.4f}")
print(f"{'Size exponent (alpha)':<25} {'~-1.5':>10} {alpha_ctrl:>12.3f} {alpha_bic:>12.3f}")
print(f"{'Duration exponent (beta)':<25} {'~-2.0':>10} {beta_ctrl:>12.3f} {beta_bic:>12.3f}")
print(f"{'Mean firing rate (Hz)':<25} {'1-5':>10} {mean_rate:>12.2f} {mean_rate_bic:>12.2f}")
print(f"{'N avalanches':<25} {'':>10} {len(sizes):>12} {len(sizes_bic):>12}")
print("=" * 65)

print("\nInterpretation:")
if sigma_ctrl_4ms > 0.8 and sigma_ctrl_4ms < 1.2:
    print("  Control network: NEAR-CRITICAL (sigma within 0.8-1.2)")
elif sigma_ctrl_4ms < 0.8:
    print("  Control network: SUBCRITICAL (sigma < 0.8)")
else:
    print("  Control network: SUPERCRITICAL (sigma > 1.2)")

if sigma_bic_4ms > sigma_ctrl_4ms:
    print("  Bicuculline shifted network toward SUPERCRITICAL dynamics (expected)")
else:
    print("  Bicuculline effect was not as expected (may need parameter tuning)")

# %% [markdown]
# ## Key Observations
#
# 1. **Branching ratio**: A healthy cortical culture should operate near
#    sigma = 1.0 (critical).  The exact value depends on network parameters,
#    noise levels, and the bin size used for analysis.  Bin sizes of 2-8 ms
#    are most commonly used in the literature.
#
# 2. **Avalanche size distribution**: At criticality, sizes follow a power
#    law with exponent alpha ~ -1.5 (mean-field universality class).
#    Deviations indicate subcritical (steeper falloff) or supercritical
#    (excess large avalanches) dynamics.
#
# 3. **Avalanche duration distribution**: Durations should follow a power
#    law with beta ~ -2.0.  The size-duration scaling exponent gamma
#    should satisfy the crackling noise relation:
#    gamma = (alpha - 1) / (beta - 1).
#
# 4. **Bicuculline perturbation**: Removing inhibition pushes the network
#    supercritical (sigma > 1), consistent with experimental observations
#    that GABA_A blockade produces epileptiform activity with runaway
#    excitation and large synchronous bursts.
#
# 5. **Scaling note**: With 10K neurons and 5 s of simulation, statistical
#    estimates may be noisy.  For publication-quality criticality analysis,
#    use 100K+ neurons and 60+ seconds of recording to obtain robust
#    power-law fits spanning multiple decades.
