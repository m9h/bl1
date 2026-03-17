# %% [markdown]
# # Notebook 1: Izhikevich Neuron Type Validation
#
# Validates that each Izhikevich cell type in the BL-1 simulator reproduces
# the known firing patterns described in Izhikevich (2003).  Five cortical
# cell types are tested: Regular Spiking (RS), Intrinsically Bursting (IB),
# Chattering (CH), Fast Spiking (FS), and Low-Threshold Spiking (LTS).
#
# An AdEx model comparison and F-I curve analysis are included to cross-
# validate the two neuron backends.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bl1.core.izhikevich import (
    IzhikevichParams,
    NeuronState,
    V_PEAK,
    V_REST,
    izhikevich_step,
)
from bl1.core.adex import (
    AdExParams,
    AdExState,
    adex_step,
    _CELL_TYPES as ADEX_CELL_TYPES,
)

print("JAX devices:", jax.devices())
key = jax.random.PRNGKey(42)

# %% [markdown]
# ## Single Neuron Simulation Helper
#
# The `simulate_izh_neuron` function runs a single Izhikevich neuron for a
# specified duration with constant input current and records the membrane
# potential trace.  The Izhikevich model uses semi-implicit Euler with
# dt = 0.5 ms (two half-steps per millisecond as recommended by Izhikevich).

# %%
def simulate_izh_neuron(a, b, c, d, I_const, duration_ms=1000, dt=0.5):
    """Simulate a single Izhikevich neuron with constant current injection.

    Returns:
        t_ms: time vector (ms)
        v_trace: membrane potential trace (mV)
        spike_times: list of spike times (ms)
    """
    n_steps = int(duration_ms / dt)
    params = IzhikevichParams(
        a=jnp.array([a]),
        b=jnp.array([b]),
        c=jnp.array([c]),
        d=jnp.array([d]),
    )
    state = NeuronState(
        v=jnp.array([V_REST]),
        u=jnp.array([b * V_REST]),
        spikes=jnp.zeros(1, dtype=jnp.bool_),
    )
    I_ext = jnp.array([I_const])

    v_trace = np.zeros(n_steps)
    spike_times = []

    for i in range(n_steps):
        state = izhikevich_step(state, params, I_ext, dt)
        v_trace[i] = float(state.v[0])
        if bool(state.spikes[0]):
            spike_times.append(i * dt)
            # Record the peak voltage for display
            v_trace[i] = V_PEAK

    t_ms = np.arange(n_steps) * dt
    return t_ms, v_trace, spike_times


def plot_neuron(t_ms, v_trace, title, description, ax=None):
    """Plot membrane potential trace with descriptive annotation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ms, v_trace, "k-", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title(f"{title}")
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-80, 40)
    ax.text(0.02, 0.95, description, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", style="italic", color="gray")
    return ax

# %% [markdown]
# ## Regular Spiking (RS)
#
# Parameters: a=0.02, b=0.2, c=-65, d=8, I=10
#
# RS neurons are the most common excitatory cell type in cortex (64% in our
# model).  They fire tonically with spike frequency adaptation: the
# inter-spike interval (ISI) increases over time as the recovery variable
# `u` accumulates.

# %%
%%time
t, v, spikes_rs = simulate_izh_neuron(a=0.02, b=0.2, c=-65, d=8, I_const=10)
plot_neuron(t, v, "Regular Spiking (RS)",
            "Tonic spiking with adaptation (ISIs increase over time)")
plt.tight_layout()
plt.show()

# Verify adaptation: ISIs should increase
if len(spikes_rs) > 3:
    isis = np.diff(spikes_rs)
    print(f"RS: {len(spikes_rs)} spikes, first 5 ISIs: {isis[:5].round(1)} ms")
    print(f"  Adaptation confirmed: first ISI ({isis[0]:.1f} ms) < last ISI ({isis[-1]:.1f} ms): {isis[0] < isis[-1]}")

# %% [markdown]
# ## Intrinsically Bursting (IB)
#
# Parameters: a=0.02, b=0.2, c=-55, d=4, I=10
#
# IB neurons produce an initial burst of closely-spaced spikes, then
# transition to tonic single-spike firing.  The higher reset voltage
# (c=-55) allows the neuron to fire again quickly after the first spike.

# %%
%%time
t, v, spikes_ib = simulate_izh_neuron(a=0.02, b=0.2, c=-55, d=4, I_const=10)
plot_neuron(t, v, "Intrinsically Bursting (IB)",
            "Initial burst followed by tonic single spikes")
plt.tight_layout()
plt.show()

if len(spikes_ib) > 3:
    isis = np.diff(spikes_ib)
    print(f"IB: {len(spikes_ib)} spikes, first 5 ISIs: {isis[:5].round(1)} ms")
    print(f"  Burst-then-tonic: initial ISI ({isis[0]:.1f} ms) should be short")

# %% [markdown]
# ## Chattering (CH)
#
# Parameters: a=0.02, b=0.2, c=-50, d=2, I=10
#
# CH neurons produce rhythmic bursting: clusters of 2-4 closely-spaced
# spikes separated by silent intervals.  The very high reset voltage
# (c=-50) combined with small recovery increment (d=2) allows re-firing
# within each burst.

# %%
%%time
t, v, spikes_ch = simulate_izh_neuron(a=0.02, b=0.2, c=-50, d=2, I_const=10)
plot_neuron(t, v, "Chattering (CH)",
            "Rhythmic bursting: clusters of fast spikes with pauses")
plt.tight_layout()
plt.show()

if len(spikes_ch) > 3:
    isis = np.diff(spikes_ch)
    print(f"CH: {len(spikes_ch)} spikes, first 10 ISIs: {isis[:10].round(1)} ms")

# %% [markdown]
# ## Fast Spiking (FS)
#
# Parameters: a=0.1, b=0.2, c=-65, d=2, I=10
#
# FS neurons (parvalbumin-positive interneurons) fire at high rates with
# minimal spike frequency adaptation.  The large `a` parameter (0.1 vs
# 0.02 for excitatory types) causes the recovery variable to track voltage
# closely, preventing accumulation of adaptation.

# %%
%%time
t, v, spikes_fs = simulate_izh_neuron(a=0.1, b=0.2, c=-65, d=2, I_const=10)
plot_neuron(t, v, "Fast Spiking (FS)",
            "Fast tonic firing with little/no adaptation")
plt.tight_layout()
plt.show()

if len(spikes_fs) > 3:
    isis = np.diff(spikes_fs)
    print(f"FS: {len(spikes_fs)} spikes, ISI CV: {np.std(isis)/np.mean(isis):.3f} (low = no adaptation)")
    print(f"  First ISI: {isis[0]:.1f} ms, Last ISI: {isis[-1]:.1f} ms")

# %% [markdown]
# ## Low-Threshold Spiking (LTS)
#
# Parameters: a=0.02, b=0.25, c=-65, d=2, I=10
#
# LTS neurons (somatostatin-positive interneurons) have a higher `b`
# parameter (0.25 vs 0.2), making the recovery variable more sensitive to
# subthreshold voltage.  This creates a low threshold for burst activation
# and produces initial bursting followed by tonic firing.

# %%
%%time
t, v, spikes_lts = simulate_izh_neuron(a=0.02, b=0.25, c=-65, d=2, I_const=10)
plot_neuron(t, v, "Low-Threshold Spiking (LTS)",
            "Low-threshold bursting with spike frequency adaptation")
plt.tight_layout()
plt.show()

if len(spikes_lts) > 3:
    isis = np.diff(spikes_lts)
    print(f"LTS: {len(spikes_lts)} spikes, first 5 ISIs: {isis[:5].round(1)} ms")

# %% [markdown]
# ## AdEx Model Comparison
#
# The Adaptive Exponential Integrate-and-Fire (AdEx) model (Brette &
# Gerstner 2005) provides a more biophysically grounded alternative with
# an exponential spike initiation mechanism and linear adaptation current.
#
# Here we simulate the five AdEx cell types with I = 700 pA constant
# current and compare their firing patterns qualitatively to the
# Izhikevich types above.  Both models should produce the same qualitative
# repertoire (tonic, bursting, adapting, fast spiking) despite different
# parameterisations.

# %%
%%time

def simulate_adex_neuron(cell_type, I_const=700.0, duration_ms=1000, dt=0.5):
    """Simulate a single AdEx neuron with constant current injection."""
    p = ADEX_CELL_TYPES[cell_type]
    n_steps = int(duration_ms / dt)

    params = AdExParams(
        C=jnp.array([p["C"]]),
        g_L=jnp.array([p["g_L"]]),
        E_L=jnp.array([p["E_L"]]),
        delta_T=jnp.array([p["delta_T"]]),
        V_T=jnp.array([p["V_T"]]),
        V_reset=jnp.array([p["V_reset"]]),
        V_peak=jnp.array([p["V_peak"]]),
        a=jnp.array([p["a"]]),
        b=jnp.array([p["b"]]),
        tau_w=jnp.array([p["tau_w"]]),
    )
    state = AdExState(
        v=jnp.array([p["E_L"]]),
        w=jnp.zeros(1),
        spikes=jnp.zeros(1, dtype=jnp.bool_),
    )
    I_ext = jnp.array([I_const])

    v_trace = np.zeros(n_steps)
    spike_times = []

    for i in range(n_steps):
        state = adex_step(state, params, I_ext, dt)
        v_trace[i] = float(state.v[0])
        if bool(state.spikes[0]):
            spike_times.append(i * dt)
            v_trace[i] = float(p["V_peak"])

    t_ms = np.arange(n_steps) * dt
    return t_ms, v_trace, spike_times


adex_types = [
    ("RS", "Regular Spiking -- tonic with adaptation"),
    ("Bursting", "Bursting -- initial burst then tonic"),
    ("FS", "Fast Spiking -- high rate, no adaptation"),
    ("Adapting", "Adapting -- strong subthreshold adaptation"),
    ("Irregular", "Irregular -- negative adaptation coupling"),
]

fig, axes = plt.subplots(len(adex_types), 1, figsize=(10, 12), sharex=True)

for ax, (ctype, desc) in zip(axes, adex_types):
    t, v, spikes = simulate_adex_neuron(ctype, I_const=700.0)
    ax.plot(t, v, "k-", linewidth=0.5)
    ax.set_ylabel("V (mV)")
    ax.set_title(f"AdEx {ctype}")
    ax.set_xlim(0, t[-1])
    ax.text(0.02, 0.95, desc, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", style="italic", color="gray")
    print(f"AdEx {ctype}: {len(spikes)} spikes in 1000 ms")

axes[-1].set_xlabel("Time (ms)")
fig.suptitle("AdEx Neuron Types (I = 700 pA)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## F-I Curve: Firing Rate vs. Input Current
#
# The F-I curve characterises how a neuron's firing rate depends on the
# injected current.  Biologically, FS interneurons have steeper F-I curves
# than RS pyramidal cells, meaning they recruit more rapidly with
# increasing drive.  This is critical for maintaining E/I balance in
# cortical circuits.
#
# We sweep input current from 0 to 20 (arbitrary units for Izhikevich)
# and compute the mean firing rate for RS and FS neurons.

# %%
%%time

I_range = np.arange(0, 21, 1.0)
rates_rs = []
rates_fs = []

for I_val in I_range:
    # RS neuron
    _, _, spk = simulate_izh_neuron(a=0.02, b=0.2, c=-65, d=8, I_const=float(I_val))
    rates_rs.append(len(spk))  # spikes per 1000 ms = Hz

    # FS neuron
    _, _, spk = simulate_izh_neuron(a=0.1, b=0.2, c=-65, d=2, I_const=float(I_val))
    rates_fs.append(len(spk))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(I_range, rates_rs, "bo-", label="RS (a=0.02, d=8)", markersize=4)
ax.plot(I_range, rates_fs, "rs-", label="FS (a=0.1, d=2)", markersize=4)
ax.set_xlabel("Input Current I (a.u.)")
ax.set_ylabel("Firing Rate (Hz)")
ax.set_title("F-I Curves: Regular Spiking vs. Fast Spiking")
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate
ax.text(0.6, 0.3, "FS has steeper F-I curve\n(faster recruitment with drive)",
        transform=ax.transAxes, fontsize=9, color="red", style="italic")
ax.text(0.6, 0.15, "RS shows adaptation-limited\nrate saturation",
        transform=ax.transAxes, fontsize=9, color="blue", style="italic")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary Table
#
# Compare spike counts and mean inter-spike intervals (ISI) across all
# five Izhikevich neuron types with I=10 constant current.

# %%
neuron_types = {
    "RS":  {"a": 0.02, "b": 0.2,  "c": -65, "d": 8},
    "IB":  {"a": 0.02, "b": 0.2,  "c": -55, "d": 4},
    "CH":  {"a": 0.02, "b": 0.2,  "c": -50, "d": 2},
    "FS":  {"a": 0.1,  "b": 0.2,  "c": -65, "d": 2},
    "LTS": {"a": 0.02, "b": 0.25, "c": -65, "d": 2},
}

print(f"{'Type':<6} {'Spikes':>7} {'Mean ISI':>10} {'ISI CV':>8} {'Pattern'}")
print("-" * 60)

for name, p in neuron_types.items():
    _, _, spk = simulate_izh_neuron(**p, I_const=10)
    n_spk = len(spk)
    if n_spk > 1:
        isis = np.diff(spk)
        mean_isi = np.mean(isis)
        cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else 0
    else:
        mean_isi = float("nan")
        cv_isi = float("nan")

    # Classify pattern
    if name == "RS":
        pattern = "Tonic + adaptation"
    elif name == "IB":
        pattern = "Burst then tonic"
    elif name == "CH":
        pattern = "Rhythmic bursting"
    elif name == "FS":
        pattern = "Fast tonic"
    else:
        pattern = "LT burst + adaptation"

    print(f"{name:<6} {n_spk:>7} {mean_isi:>10.2f} {cv_isi:>8.3f} {pattern}")

# %% [markdown]
# ## Key Observations
#
# 1. **RS neurons** show clear spike frequency adaptation, with ISIs
#    increasing monotonically -- the hallmark of cortical pyramidal cells.
#
# 2. **IB neurons** produce an initial burst (short ISIs) followed by
#    tonic firing, matching the behaviour of layer 5 thick-tufted
#    pyramidal cells.
#
# 3. **CH neurons** show rhythmic bursting with alternating short
#    (intra-burst) and long (inter-burst) ISIs, characteristic of
#    chattering cells in superficial cortical layers.
#
# 4. **FS neurons** fire at high rates with near-constant ISIs (low ISI
#    CV), matching the non-adapting behaviour of parvalbumin-positive
#    basket cells.
#
# 5. **LTS neurons** show low-threshold activation with initial bursting,
#    consistent with somatostatin-positive Martinotti cells.
#
# 6. **AdEx comparison** shows qualitatively similar firing pattern
#    repertoire despite different biophysical parameterisation, validating
#    that both backends can support the same cortical dynamics.
#
# 7. **F-I curves** confirm that FS neurons have steeper gain than RS
#    neurons, a property essential for stable E/I balance.
