"""Run BL-1 benchmarks on Modal with A100 GPU.

Deploys the profiling script to a cloud A100 and streams results back.

Prerequisites:
    pip install modal
    modal setup  # one-time auth

Usage:
    modal run benchmarks/modal_benchmark.py                     # default benchmark
    modal run benchmarks/modal_benchmark.py --calibrate         # run calibration sweep
    modal run benchmarks/modal_benchmark.py --n-neurons 100000  # specific size
    modal run benchmarks/modal_benchmark.py --test-suite        # run pytest on GPU
"""

import time

import modal

app = modal.App("bl1-benchmark")

# ---------------------------------------------------------------------------
# Container image: JAX with CUDA 12, plus BL-1 installed as an editable package.
#
# Key points:
#   - jax[cuda12] pulls in the correct jaxlib+CUDA wheels for GPU support.
#   - We copy the full project tree then `pip install -e .` so that both
#     `import bl1` and the benchmarks/ and scripts/ directories are available.
# ---------------------------------------------------------------------------
bl1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "equinox>=0.11.0",
        "matplotlib>=3.8",
        "h5py>=3.10",
        "pyyaml>=6.0",
        "numpy>=1.24",
        "pytest>=7.0",
    )
    .copy_local_dir("src/bl1", "/root/bl1_pkg/src/bl1")
    .copy_local_file("pyproject.toml", "/root/bl1_pkg/pyproject.toml")
    .copy_local_file("README.md", "/root/bl1_pkg/README.md")
    .copy_local_dir("benchmarks", "/root/bl1_pkg/benchmarks")
    .copy_local_dir("scripts", "/root/bl1_pkg/scripts")
    .copy_local_dir("tests", "/root/bl1_pkg/tests")
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
) -> dict:
    """Run BL-1 benchmark on A100 GPU and return timing results."""
    import sys

    sys.path.insert(0, "/root/bl1_pkg")

    import jax

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")

    t_start = time.perf_counter()

    from benchmarks.profile_scale import run_benchmarks

    run_benchmarks(sizes=n_neurons_list, duration_ms=duration_ms, dt=dt)

    total_time = time.perf_counter() - t_start

    return {
        "backend": backend,
        "devices": devices,
        "n_neurons_list": n_neurons_list,
        "duration_ms": duration_ms,
        "total_wall_time_s": total_time,
    }


@app.function(
    image=bl1_image,
    gpu="A100",
    timeout=3600,
)
def run_calibration(
    n_neurons: int = 5000,
    duration_ms: float = 30000,
) -> dict:
    """Run bursting calibration sweep on A100 GPU."""
    import sys

    sys.path.insert(0, "/root/bl1_pkg")

    import jax

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")

    t_start = time.perf_counter()

    from scripts.calibrate_bursting import parameter_sweep

    parameter_sweep()

    total_time = time.perf_counter() - t_start

    return {
        "backend": backend,
        "devices": devices,
        "total_wall_time_s": total_time,
    }


@app.function(
    image=bl1_image,
    gpu="A100",
    timeout=900,
)
def run_full_test_suite() -> dict:
    """Run pytest on A100 GPU to verify JAX+CUDA works with BL-1.

    Returns a dict with test results, backend info, and pass/fail status.
    """
    import subprocess
    import sys

    sys.path.insert(0, "/root/bl1_pkg")

    import jax

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")

    # Quick smoke test: JAX can actually use the GPU
    import jax.numpy as jnp

    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print(f"GPU smoke test passed: matmul result shape = {y.shape}")

    # Run the full test suite
    t_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd="/root/bl1_pkg",
        capture_output=True,
        text=True,
        timeout=600,
    )
    test_time = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("PYTEST STDOUT:")
    print("=" * 70)
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

    if result.stderr:
        print("\n" + "=" * 70)
        print("PYTEST STDERR (last 2000 chars):")
        print("=" * 70)
        print(result.stderr[-2000:])

    return {
        "backend": backend,
        "devices": devices,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "test_wall_time_s": test_time,
        "stdout_tail": result.stdout[-3000:],
    }


@app.local_entrypoint()
def main(
    n_neurons: int = 0,
    duration_ms: float = 5000,
    calibrate: bool = False,
    test_suite: bool = False,
):
    """Local entrypoint that dispatches to the appropriate remote function.

    Args:
        n_neurons: Number of neurons for benchmark (0 = use default list).
        duration_ms: Simulation duration in ms.
        calibrate: If True, run calibration sweep instead of benchmark.
        test_suite: If True, run the full pytest suite on GPU.
    """
    wall_start = time.perf_counter()

    if test_suite:
        print("Running full test suite on A100 GPU...")
        print("=" * 60)
        result = run_full_test_suite.remote()
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("TEST SUITE RESULTS")
        print("=" * 60)
        print(f"  Backend:    {result['backend']}")
        print(f"  Devices:    {result['devices']}")
        print(f"  Passed:     {result['passed']}")
        print(f"  Test time:  {result['test_wall_time_s']:.1f}s (remote)")
        print(f"  Total time: {wall_total:.1f}s (including Modal overhead)")

    elif calibrate:
        print("Running bursting calibration on A100...")
        print("=" * 60)
        result = run_calibration.remote()
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"  Backend:    {result['backend']}")
        print(f"  Devices:    {result['devices']}")
        print(f"  Remote:     {result['total_wall_time_s']:.1f}s")
        print(f"  Total:      {wall_total:.1f}s (including Modal overhead)")

    else:
        if n_neurons > 0:
            sizes = [n_neurons]
        else:
            sizes = [1000, 5000, 10000, 50000, 100000]

        print(f"Running benchmark on A100 for N={sizes}...")
        print("=" * 60)
        result = run_benchmark.remote(
            n_neurons_list=sizes,
            duration_ms=duration_ms,
        )
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"  Backend:    {result['backend']}")
        print(f"  Devices:    {result['devices']}")
        print(f"  Sizes:      {result['n_neurons_list']}")
        print(f"  Duration:   {result['duration_ms']}ms")
        print(f"  Remote:     {result['total_wall_time_s']:.1f}s")
        print(f"  Total:      {wall_total:.1f}s (including Modal overhead)")
