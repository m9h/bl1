# BL-1 Benchmarks

## Local CPU

```bash
python benchmarks/profile_scale.py --n-neurons 1000 5000 10000
python benchmarks/profile_scale.py --n-neurons 100000 --duration-ms 5000
```

## Modal A100 GPU

```bash
pip install modal
modal setup                                              # one-time auth

modal run benchmarks/modal_benchmark.py                  # default benchmark (1K-100K neurons)
modal run benchmarks/modal_benchmark.py --n-neurons 100000  # specific size
modal run benchmarks/modal_benchmark.py --calibrate      # bursting calibration sweep
modal run benchmarks/modal_benchmark.py --test-suite     # run pytest on GPU
```

## Doom-Neuron Integration

```bash
modal run benchmarks/modal_doom.py                       # doom-readiness benchmark
modal run benchmarks/modal_doom.py --n-neurons 50000     # smaller network
modal run benchmarks/modal_doom.py --duration-ms 60000   # longer simulation
```

## Calibration

```bash
python scripts/calibrate_bursting.py                     # local calibration sweep
modal run benchmarks/modal_benchmark.py --calibrate      # calibration on A100
```

## What the benchmarks measure

- **profile_scale.py**: Network creation time, JIT compilation time, simulation wall-clock time, realtime factor, and spike rates across network sizes (1K-100K neurons), with and without STDP plasticity.
- **modal_benchmark.py**: Same as profile_scale but executed on an A100 GPU via Modal. Also supports running the calibration sweep and the full test suite remotely.
- **modal_doom.py**: Validates that BL-1 can sustain real-time simulation rates required for doom-neuron integration. Reports the realtime factor (must be >= 1.0x for live gameplay).
