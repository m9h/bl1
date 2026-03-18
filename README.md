# BL-1: In-Silico Cortical Culture Simulator

![Python 3.10+](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![JAX](https://img.shields.io/badge/accelerator-JAX-orange)

BL-1 is a JAX-based framework for simulating dissociated cortical cultures growing on multi-electrode arrays (MEAs), inspired by the DishBrain system (Kagan et al. 2022). It combines biologically detailed spiking neuron models with conductance-based synapses, four timescales of synaptic plasticity, virtual MEA recording and stimulation, and closed-loop game experiments -- all within a fully differentiable, JIT-compiled simulation loop built on `jax.lax.scan`. BL-1 enables researchers to run in-silico replications of biological intelligence experiments at scale, with GPU acceleration and support for gradient-based optimization through surrogate gradients.

---

## Key Features

### Neuron Models
- **Izhikevich** (2003) -- five cortical cell types: Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH), Fast Spiking (FS), Low-Threshold Spiking (LTS)
- **Adaptive Exponential (AdEx)** integrate-and-fire model
- **Hybrid populations** mixing Izhikevich and AdEx neurons
- **Jaxley adapter** for morphologically detailed compartmental neurons (optional)

### Conductance-Based Synapses
- **AMPA** -- fast excitatory (tau = 2 ms)
- **NMDA** -- slow excitatory with Mg2+ voltage-dependent block (dual-exponential kinetics)
- **GABA_A** -- fast inhibitory (tau = 6 ms)
- **GABA_B** -- slow inhibitory (dual-exponential kinetics)

### Plasticity (4 Timescales)
- **Short-term plasticity (STP)** -- Tsodyks-Markram depression and facilitation
- **Spike-timing-dependent plasticity (STDP)** -- asymmetric Hebbian learning with configurable time constants
- **Homeostatic scaling** -- firing-rate-based synaptic weight normalization
- **Structural plasticity** -- activity-dependent synapse creation and pruning

### Virtual MEA
- **CL1 64-channel** -- 8x8 grid, 200 um spacing (standard configuration)
- **MaxOne HD-MEA** -- 26,400 electrodes, 17.5 um spacing (high-density mode with sparse indexing)
- Spike detection, LFP approximation, and electrical stimulation
- Configurable detection and activation radii

### Game Environments
- **Pong** -- closed-loop paddle game with DishBrain-style FEP feedback
- **Doom** -- ViZDoom integration for 3D environments (optional `vizdoom` extra)
- Three feedback modes: Free Energy Principle (`fep`), `open_loop`, `silent`

### Analysis Toolkit
- **Criticality** -- branching ratio, neuronal avalanche size distributions
- **Burst detection** -- network burst identification, inter-burst intervals, recruitment statistics
- **Functional connectivity** -- cross-correlation, transfer entropy, small-world and rich-club coefficients
- **Information theory** -- active information storage, mutual information, integration, complexity
- **Pharmacology** -- drug effect modelling (TTX, Carbamazepine, Bicuculline, APV, CNQX)
- **Sensitivity analysis** -- parameter sweeps, fitting, firing rate and synchrony metrics

### Visualization
- Spike raster plots with E/I coloring
- Population firing rate traces
- MEA heatmaps and activity plots
- Avalanche distribution plots
- Burst overlays and ISI distributions

### Performance
- Entire simulation loop JIT-compiled via `jax.lax.scan` -- zero Python overhead
- Sparse connectivity with JAX BCOO and event-driven operations
- Fast sparse path with `segment_sum` for 50K+ neuron networks
- Pallas custom kernels for CSC event-driven synaptic input (GPU)
- GPU-ready: **5.3x realtime at 10K neurons on A100**
- Differentiable through surrogate gradients (SuperSpike, sigmoid, arctan)

---

## Installation

**From source (recommended):**

```bash
git clone https://github.com/your-org/bl1.git
cd bl1
pip install -e ".[dev]"
```

**With optional extras:**

```bash
# Development tools (pytest, hypothesis, ipykernel)
pip install -e ".[dev]"

# Code quality (ruff, mypy, vulture, radon, bandit, interrogate)
pip install -e ".[quality]"

# Documentation (sphinx, RTD theme)
pip install -e ".[docs]"

# Morphological neurons via Jaxley
pip install -e ".[jaxley]"

# Doom game environment
pip install -e ".[vizdoom]"
```

**JAX GPU support:**

BL-1 uses CPU JAX by default. For GPU acceleration, install the appropriate JAX variant for your CUDA version:

```bash
pip install -U "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

---

## Quick Start

```python
import jax
import jax.numpy as jnp

from bl1 import Culture
from bl1.core import simulate, create_synapse_state
from bl1.visualization import plot_raster

# 1. Create a culture: 200 neurons on a 3x3 mm substrate
key = jax.random.PRNGKey(42)
net_params, state, izh_params = Culture.create(
    key, n_neurons=200, ei_ratio=0.8, g_exc=0.05, g_inh=0.20,
)

# 2. Prepare synapse state and external drive
n_neurons = 200
syn_state = create_synapse_state(n_neurons)
T_ms, dt = 1000.0, 0.5               # 1 second simulation
n_steps = int(T_ms / dt)

# Background noise current (keeps the culture spontaneously active)
I_ext = 5.0 * jax.random.normal(jax.random.PRNGKey(0), (n_steps, n_neurons))

# 3. Build initial neuron state from the culture state
from bl1.core import NeuronState
neuron_state = NeuronState(v=state.v, u=state.u, spikes=state.spikes)

# 4. Run simulation
result = simulate(
    izh_params, neuron_state, syn_state, stdp_state=None,
    W_exc=net_params.W_exc, W_inh=net_params.W_inh,
    I_external=I_ext, dt=dt,
)

# 5. Plot the spike raster
n_exc = int(n_neurons * 0.8)
fig = plot_raster(result.spike_history, dt_ms=dt, ei_boundary=n_exc)
fig.savefig("raster.png", dpi=150)
```

---

## Architecture

```
                        +------------------+
                        |   Closed-Loop    |  bl1.loop
                        |   Controller     |
                        +--------+---------+
                                 |
                    encode / decode / stimulate
                                 |
                  +--------------+--------------+
                  |         Virtual MEA         |  bl1.mea
                  |  (64-ch / HD-MEA / LFP)     |
                  +--------------+--------------+
                                 |
                   record spikes | inject current
                                 |
     +---------------------------+---------------------------+
     |                    Network Layer                      |  bl1.network
     |  place_neurons  |  build_connectivity  |  Culture     |
     +---------------------------+---------------------------+
                                 |
              +------------------+------------------+
              |           Plasticity Stack          |  bl1.plasticity
              |  STP  |  STDP  |  Homeostatic  |   |
              +------------------+------------------+
                                 |
         +-----------------------+-----------------------+
         |              Core Simulation                  |  bl1.core
         |  Izhikevich / AdEx neurons                    |
         |  AMPA / NMDA / GABA_A / GABA_B synapses       |
         |  simulate() via jax.lax.scan                  |
         +-----------------------------------------------+
```

**Data flow per timestep:** Neurons fire spikes. Spikes propagate through conductance-based synapses (optionally modulated by STP). STDP and homeostatic rules update weights. The MEA records spikes and injects stimulation currents. In closed-loop mode, the controller decodes motor actions from neural activity, steps the game, encodes sensory feedback, and delivers stimulation according to the selected feedback policy.

---

## Configuration

BL-1 uses YAML configuration files to define experiment parameters. Three reference configs are included:

| Config | File | Description |
|--------|------|-------------|
| **Default** | `configs/default.yaml` | 100K neurons, standard Izhikevich parameters, 64-ch MEA |
| **Wagenaar Calibrated** | `configs/wagenaar_calibrated.yaml` | 5K neurons tuned to reproduce Wagenaar et al. (2006) spontaneous bursting (~8 bursts/min, IBI ~9s) |
| **DishBrain Pong** | `configs/dishbrain_pong.yaml` | Closed-loop Pong experiment with FEP feedback |

Load a config:

```python
import yaml

with open("configs/wagenaar_calibrated.yaml") as f:
    cfg = yaml.safe_load(f)

net_params, state, izh_params = Culture.create(
    key,
    n_neurons=cfg["culture"]["n_neurons"],
    ei_ratio=cfg["culture"]["ei_ratio"],
    lambda_um=cfg["culture"]["lambda_um"],
    p_max=cfg["culture"]["p_max"],
    g_exc=cfg["culture"]["g_exc"],
    g_inh=cfg["culture"]["g_inh"],
)
```

---

## Documentation

- **API reference:** `docs/api/` -- auto-generated Sphinx docs for all modules
- **Quick start guide:** `docs/quickstart.rst`
- **Notebooks:** `notebooks/00_quickstart.ipynb` -- interactive tutorial

Build the HTML docs locally:

```bash
make docs
# Output: docs/_build/html/index.html
```

---

## Development

BL-1 uses a Makefile-driven workflow with [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, [mypy](https://mypy-lang.org/) for type checking, and [pytest](https://pytest.org/) for testing.

```bash
# Lint and format check
make lint

# Auto-fix lint issues
make lint-fix

# Type checking
make typecheck

# Run tests (excludes slow calibration tests)
make test

# Tests with coverage report
make coverage

# Code quality metrics (docstring coverage, dead code, complexity, security)
make quality

# All checks
make all

# Benchmarks (local CPU / Modal A100)
make benchmark
make benchmark-gpu
```

---

## License

BL-1 is released under the [MIT License](LICENSE).
