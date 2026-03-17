"""Run doom-neuron training against BL-1's simulated culture on Modal.

Deploys two processes:
1. BL-1 Virtual CL1 (JAX + simulated neurons on GPU)
2. doom-neuron training server (PyTorch + VizDoom on GPU)

Architecture
------------
In a full deployment, the BL-1 virtual CL1 and doom-neuron training server
need bidirectional real-time communication (UDP-like stimulus/response loop).
Modal's container-to-container networking options include:

  - modal.Queue: async message passing between functions (good for commands,
    not ideal for high-frequency tick data).
  - modal.web_endpoint: expose one container as an HTTP endpoint the other
    can call (adds latency per tick).
  - modal.Sandbox: run both processes in a single container (simplest for
    tightly-coupled real-time communication).

For the real-time neuroscience loop (~10-100 Hz tick rate), the recommended
approach is to run both processes in a **single container** using the
Sandbox approach or subprocess orchestration, avoiding network round-trips
between containers.

The template below shows both the multi-container architecture (for reference)
and a single-container combined mode (recommended for actual use).

Usage:
    modal run benchmarks/modal_doom.py                          # combined mode
    modal run benchmarks/modal_doom.py --n-neurons 50000
    modal run benchmarks/modal_doom.py --max-episodes 50
"""

import modal

app = modal.App("bl1-doom")

# ---------------------------------------------------------------------------
# Image for BL-1 virtual CL1 (JAX on CUDA)
# ---------------------------------------------------------------------------
bl1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "equinox>=0.11.0",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "h5py>=3.10",
        "matplotlib>=3.8",
    )
    .copy_local_dir("src/bl1", "/root/bl1/src/bl1")
    .copy_local_file("pyproject.toml", "/root/bl1/pyproject.toml")
    .copy_local_file("README.md", "/root/bl1/README.md")
    .run_commands("cd /root/bl1 && pip install -e .")
)

# ---------------------------------------------------------------------------
# Image for doom-neuron training (PyTorch + VizDoom)
#
# NOTE: This image expects the doom-neuron repo to be cloned at /tmp/doom-neuron
# on the local machine before building.  If it doesn't exist the Modal build
# will fail with a clear error.
# ---------------------------------------------------------------------------
doom_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglu1-mesa", "libx11-6", "xvfb")
    .pip_install(
        "torch",
        "vizdoom",
        "tensorboard",
        "opencv-python",
        "numpy",
    )
    # doom-neuron source must be available locally at /tmp/doom-neuron
    # Clone it first:  git clone <doom-neuron-repo> /tmp/doom-neuron
    .copy_local_dir("/tmp/doom-neuron", "/root/doom-neuron")
)


# ---------------------------------------------------------------------------
# Multi-container architecture (template / reference)
# ---------------------------------------------------------------------------
# These functions show how you *would* structure two separate GPU containers.
# In practice, the real-time tick loop (10-100 Hz) requires lower latency
# than Modal's inter-container networking provides, so the combined mode
# below is preferred.


@app.function(image=bl1_image, gpu="A100", timeout=3600)
def run_virtual_cl1(
    training_host: str,
    n_neurons: int = 100_000,
    tick_frequency: int = 10,
):
    """Run BL-1 as a virtual CL1 server.

    This starts the UDP bridge that sends spike data to and receives
    stimulation from the doom-neuron training server.
    """
    import sys

    sys.path.insert(0, "/root/bl1")

    import jax

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    from bl1.compat.udp_bridge import VirtualCL1Server

    server = VirtualCL1Server(
        training_host=training_host,
        tick_frequency_hz=tick_frequency,
        n_neurons=n_neurons,
    )
    server.run()


@app.function(image=doom_image, gpu="A100", timeout=3600)
def run_doom_training(cl1_host: str, max_episodes: int = 100):
    """Run doom-neuron training server.

    Connects to the virtual CL1 to receive neural activity and send
    game-state stimulation back.
    """
    import subprocess

    subprocess.run(
        [
            "python",
            "/root/doom-neuron/training_server.py",
            "--mode",
            "train",
            "--device",
            "cuda",
            "--cl1-host",
            cl1_host,
            "--max-episodes",
            str(max_episodes),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Combined single-container mode (recommended)
# ---------------------------------------------------------------------------
# For real-time closed-loop experiments, run both BL-1 and doom-neuron
# in the same container to avoid network latency between tick steps.
# This function uses bl1_image since BL-1/JAX is the heavier dependency;
# doom-neuron dependencies would need to be added to this image for a
# true combined deployment.


@app.function(image=bl1_image, gpu="A100", timeout=3600)
def run_combined_benchmark(
    n_neurons: int = 100_000,
    duration_ms: float = 30_000,
    dt: float = 0.5,
):
    """Run a BL-1 culture simulation benchmark (no doom-neuron, JAX only).

    This validates that the Virtual CL1 can sustain the required tick rate
    for real-time doom-neuron integration.
    """
    import sys
    import time

    sys.path.insert(0, "/root/bl1")

    import jax
    import jax.numpy as jnp

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    from bl1.core.izhikevich import create_population
    from bl1.core.synapses import create_synapse_state
    from bl1.core.integrator import simulate
    from bl1.network.topology import place_neurons, build_connectivity

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    print(f"\nBuilding network: {n_neurons:,} neurons...")
    t0 = time.perf_counter()
    positions = place_neurons(k1, n_neurons, (3000.0, 3000.0))
    params, state, is_exc = create_population(k2, n_neurons)
    W_exc, W_inh, _delays = build_connectivity(
        k3, positions, is_exc,
        lambda_um=200.0, p_max=0.21, g_exc=0.05, g_inh=0.20,
    )
    syn_state = create_synapse_state(n_neurons)
    print(f"Network built in {time.perf_counter() - t0:.2f}s")

    n_steps = int(duration_ms / dt)
    I_ext = jax.random.normal(jax.random.PRNGKey(99), (n_steps, n_neurons)) * 3.0

    # JIT warmup
    print(f"\nJIT compiling (warmup)...")
    warmup_steps = min(100, n_steps)
    t0 = time.perf_counter()
    result = simulate(
        params, state, syn_state, None,
        W_exc, W_inh, I_ext[:warmup_steps],
        dt=dt, plasticity_fn=None,
    )
    result.spike_history.block_until_ready()
    jit_time = time.perf_counter() - t0
    print(f"JIT compile: {jit_time:.2f}s")

    # Timed simulation
    print(f"\nSimulating {duration_ms / 1000:.1f}s of neural activity...")
    t0 = time.perf_counter()
    result = simulate(
        params, state, syn_state, None,
        W_exc, W_inh, I_ext,
        dt=dt, plasticity_fn=None,
    )
    result.spike_history.block_until_ready()
    sim_time = time.perf_counter() - t0

    total_spikes = int(jnp.sum(result.spike_history))
    sim_ms = n_steps * dt
    rt_factor = sim_ms / (sim_time * 1000)

    print(f"\n{'=' * 60}")
    print(f"DOOM-READINESS BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Neurons:        {n_neurons:,}")
    print(f"  Sim duration:   {sim_ms / 1000:.1f}s")
    print(f"  Wall time:      {sim_time:.2f}s")
    print(f"  Realtime factor: {rt_factor:.2f}x")
    print(f"  Total spikes:   {total_spikes:,}")
    if rt_factor >= 1.0:
        print(f"  Status:         PASS - can sustain real-time for doom-neuron")
    else:
        print(f"  Status:         WARN - below real-time ({rt_factor:.2f}x)")
    print(f"{'=' * 60}")

    return {
        "n_neurons": n_neurons,
        "sim_time_s": sim_time,
        "rt_factor": rt_factor,
        "total_spikes": total_spikes,
        "jit_time_s": jit_time,
    }


@app.local_entrypoint()
def main(
    n_neurons: int = 100_000,
    max_episodes: int = 100,
    combined: bool = True,
    duration_ms: float = 30_000,
):
    """Launch BL-1 + doom-neuron training on Modal.

    Args:
        n_neurons: Number of neurons in the virtual culture.
        max_episodes: Maximum doom training episodes.
        combined: If True (default), run the combined benchmark.
        duration_ms: Simulation duration for combined benchmark.
    """
    import time as _time

    t0 = _time.perf_counter()

    if combined:
        print(f"Running doom-readiness benchmark: {n_neurons:,} neurons, "
              f"{duration_ms / 1000:.0f}s simulation")
        print("=" * 60)
        result = run_combined_benchmark.remote(
            n_neurons=n_neurons,
            duration_ms=duration_ms,
        )
        wall = _time.perf_counter() - t0
        print(f"\nTotal wall time (including Modal overhead): {wall:.1f}s")
        print(f"Realtime factor on A100: {result['rt_factor']:.2f}x")
    else:
        # Multi-container mode (template)
        print(f"Starting BL-1 virtual CL1 ({n_neurons:,} neurons) + "
              f"doom-neuron training...")
        print()
        print("NOTE: Multi-container real-time communication requires")
        print("additional networking setup. For Modal, the recommended")
        print("approach is to run both processes in a single container.")
        print()
        print("To run the combined benchmark instead:")
        print("  modal run benchmarks/modal_doom.py")
        print()
        print("For the separate-container template, the virtual CL1 and")
        print("doom-neuron server would need to be connected via:")
        print("  - modal.Queue (async, higher latency)")
        print("  - modal.web_endpoint (HTTP, ~10-50ms per tick)")
        print("  - Single-container subprocess orchestration (recommended)")

        # Launch just the BL-1 side as a demonstration
        run_virtual_cl1.remote(
            training_host="localhost",
            n_neurons=n_neurons,
        )
