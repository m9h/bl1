"""Run BL-1 benchmarks on Modal with A100 GPU.

Deploys the profiling script to a cloud A100 and streams results back.

Prerequisites:
    pip install modal
    modal setup  # one-time auth

Usage:
    modal run benchmarks/modal_benchmark.py
    modal run benchmarks/modal_benchmark.py --n-neurons 1000 5000 10000 50000 100000
"""

import modal

app = modal.App("bl1-benchmark")

# Image with JAX + CUDA + BL-1 installed
bl1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "equinox>=0.11.0",
        "matplotlib>=3.8",
        "h5py>=3.10",
        "pyyaml>=6.0",
        "numpy>=1.24",
    )
    .copy_local_dir("src/bl1", "/root/bl1_pkg/src/bl1")
    .copy_local_file("pyproject.toml", "/root/bl1_pkg/pyproject.toml")
    .copy_local_file("README.md", "/root/bl1_pkg/README.md")
    .copy_local_dir("benchmarks", "/root/bl1_pkg/benchmarks")
    .run_commands("cd /root/bl1_pkg && pip install -e .")
)


@app.function(
    image=bl1_image,
    gpu="A100",
    timeout=1800,
)
def run_benchmark(
    n_neurons_list: list[int] = [1000, 5000, 10000, 50000, 100000],
    duration_ms: float = 5000,
    dt: float = 0.5,
):
    """Run BL-1 benchmark on A100 GPU."""
    import sys
    sys.path.insert(0, "/root/bl1_pkg")

    import jax
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    from benchmarks.profile_scale import run_benchmarks
    run_benchmarks(sizes=n_neurons_list, duration_ms=duration_ms, dt=dt)


@app.function(
    image=bl1_image,
    gpu="A100",
    timeout=3600,
)
def run_calibration(
    n_neurons: int = 5000,
    duration_ms: float = 30000,
):
    """Run bursting calibration sweep on A100 GPU."""
    import sys
    sys.path.insert(0, "/root/bl1_pkg")

    import jax
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    from scripts.calibrate_bursting import parameter_sweep
    parameter_sweep()


@app.local_entrypoint()
def main(
    n_neurons: list[int] = [1000, 5000, 10000, 50000, 100000],
    duration_ms: float = 5000,
    calibrate: bool = False,
):
    if calibrate:
        print("Running bursting calibration on A100...")
        run_calibration.remote()
    else:
        print(f"Running benchmark on A100 for N={n_neurons}...")
        run_benchmark.remote(
            n_neurons_list=n_neurons,
            duration_ms=duration_ms,
        )
