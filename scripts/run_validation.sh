#!/usr/bin/env bash
# =============================================================================
# BL-1 Master Validation Runner
#
# Single entry point for running all validation tasks on the DGX Spark.
#
# Usage:
#   bash scripts/run_validation.sh              # full suite (stability = 1 hr)
#   bash scripts/run_validation.sh --quick      # skip stability test
#   bash scripts/run_validation.sh --stability-hours 2   # custom duration
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
QUICK=false
STABILITY_HOURS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=true
            shift
            ;;
        --stability-hours)
            STABILITY_HOURS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--quick] [--stability-hours N]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths (script may be invoked from any directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_PYTEST="$PROJECT_ROOT/.venv/bin/pytest"
RESULTS_DIR="$PROJECT_ROOT/results"

# ---------------------------------------------------------------------------
# Track results for summary table
# ---------------------------------------------------------------------------
STATUS_TESTS="SKIP"
STATUS_BENCHMARKS="SKIP"
STATUS_BIOVALIDATION="SKIP"
BIOVALIDATION_DETAIL=""
STATUS_STABILITY="SKIP"
STATUS_DOOM="SKIP"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  BL-1 VALIDATION SUITE"
echo "======================================================================"
echo "  Timestamp : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Host      : $(hostname)"
echo "  OS        : $(uname -srm)"
echo "  Python    : $VENV_PYTHON"
echo "  Project   : $PROJECT_ROOT"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
    echo "  GPU       : $GPU_NAME ($GPU_MEM)"
else
    echo "  GPU       : nvidia-smi not found"
fi
if $QUICK; then
    echo "  Mode      : QUICK (stability test skipped)"
else
    echo "  Mode      : FULL (stability = ${STABILITY_HOURS}h)"
fi
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo ">> Pre-flight checks..."

if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    echo "ERROR: .venv not found at $PROJECT_ROOT/.venv"
    echo "       Run dgx_setup.sh first to create the virtual environment."
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: $VENV_PYTHON not found or not executable."
    echo "       Run dgx_setup.sh first to create the virtual environment."
    exit 1
fi

if ! "$VENV_PYTHON" -c "import bl1" 2>/dev/null; then
    echo "ERROR: 'import bl1' failed in the venv Python."
    echo "       Run dgx_setup.sh first, or: .venv/bin/pip install -e ."
    exit 1
fi

echo "   .venv OK, bl1 importable."
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Track overall start time
SUITE_START=$(date +%s)

# =========================================================================
# Step A: Quick smoke test (< 1 min)
# =========================================================================
echo "======================================================================"
echo "  [A] QUICK SMOKE TEST  (pytest, fast tests only)"
echo "======================================================================"
echo ""

STEP_START=$(date +%s)

if "$VENV_PYTEST" "$PROJECT_ROOT/tests/" -m "not slow" -q --tb=short 2>&1; then
    STATUS_TESTS="PASS"
    echo ""
    echo "   Smoke tests PASSED."
else
    STATUS_TESTS="FAIL"
    echo ""
    echo "   ERROR: Smoke tests FAILED. Aborting validation."
    echo ""
    echo "======================================================================"
    echo "  SUMMARY (aborted)"
    echo "======================================================================"
    echo "  Tests              : FAIL"
    echo "  Benchmarks         : SKIPPED (tests failed)"
    echo "  Bio validation     : SKIPPED (tests failed)"
    echo "  Stability          : SKIPPED (tests failed)"
    echo "  Doom               : SKIPPED (tests failed)"
    echo "======================================================================"
    exit 1
fi

STEP_END=$(date +%s)
echo "   Duration: $((STEP_END - STEP_START))s"
echo ""

# =========================================================================
# Step B: GPU benchmark baselines (~10-30 min)
# =========================================================================
echo "======================================================================"
echo "  [B] GPU BENCHMARK BASELINES  (~10-30 min depending on GPU)"
echo "======================================================================"
echo ""

STEP_START=$(date +%s)

BENCHMARK_SCRIPT="$PROJECT_ROOT/scripts/benchmark_baselines.py"
if [[ -f "$BENCHMARK_SCRIPT" ]]; then
    if "$VENV_PYTHON" "$BENCHMARK_SCRIPT" 2>&1; then
        STATUS_BENCHMARKS="PASS"
        echo ""
        echo "   Benchmarks completed. Results saved to $RESULTS_DIR/"
    else
        STATUS_BENCHMARKS="FAIL"
        echo ""
        echo "   WARNING: Benchmarks failed (non-fatal, continuing)."
    fi
else
    STATUS_BENCHMARKS="SKIP"
    echo "   SKIPPED: $BENCHMARK_SCRIPT not found."
    echo "   Create this script to enable GPU benchmarking."
fi

STEP_END=$(date +%s)
echo "   Duration: $((STEP_END - STEP_START))s"
echo ""

# =========================================================================
# Step C: Biological validation (~5 min)
# =========================================================================
echo "======================================================================"
echo "  [C] BIOLOGICAL VALIDATION  (~5 min)"
echo "      30s simulation with Wagenaar-calibrated parameters"
echo "======================================================================"
echo ""

STEP_START=$(date +%s)

BIO_OUTPUT=$("$VENV_PYTHON" - 2>&1 <<'PYEOF'
"""Run a 60-second simulation with Wagenaar-calibrated parameters
(AMPA/NMDA split + STP) and compare statistics against published ranges."""

import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from bl1.core.izhikevich import create_population, izhikevich_step
from bl1.core.synapses import (
    SynapseState,
    create_synapse_state,
    ampa_step,
    gaba_a_step,
    nmda_step,
    compute_synaptic_current,
)
from bl1.network.topology import place_neurons, build_connectivity
from bl1.plasticity.stp import STPParams, init_stp_state, stp_step
from bl1.validation.comparison import compute_culture_statistics
from bl1.validation.datasets import compare_statistics

# --- Wagenaar-calibrated parameters (configs/wagenaar_calibrated.yaml) ---
N_NEURONS = 5000
DURATION_MS = 60_000.0  # 60 seconds for robust IBI statistics
DT = 0.5
G_EXC = 0.12
G_INH = 0.36
P_MAX = 0.21
NMDA_RATIO = 0.37        # Fraction of g_exc routed through NMDA
U_EXC = 0.30              # STP release probability
TAU_REC = 800.0           # STP recovery time constant (ms)
BG_MEAN = 1.0
BG_STD = 3.0
BURST_THRESHOLD_STD = 1.5  # Wagenaar-recommended burst detection threshold
SEED = 42

print(f"  Simulating {N_NEURONS} neurons for {DURATION_MS/1000:.0f}s (dt={DT}ms)...")
print(f"  AMPA/NMDA split: {1-NMDA_RATIO:.0%}/{NMDA_RATIO:.0%}, STP U={U_EXC} tau_rec={TAU_REC}ms")
t0 = time.perf_counter()

key = jax.random.PRNGKey(SEED)
k1, k2, k3, k4 = jax.random.split(key, 4)

positions = place_neurons(k1, N_NEURONS, (3000.0, 3000.0))
params, state, is_exc = create_population(k2, N_NEURONS)
W_exc, W_inh, _ = build_connectivity(
    k3, positions, is_exc,
    lambda_um=200.0, p_max=P_MAX, g_exc=G_EXC, g_inh=G_INH,
)
syn = create_synapse_state(N_NEURONS)

# Split excitatory weights: AMPA (fast, tau=2ms) + NMDA (slow, tau=100ms)
W_ampa = W_exc * (1.0 - NMDA_RATIO)
W_nmda = W_exc * NMDA_RATIO

# Custom STP params: moderate depression for excitatory, facilitation for inhibitory
U = jnp.where(is_exc, U_EXC, 0.04)
tau_rec = jnp.where(is_exc, TAU_REC, 100.0)
tau_fac = jnp.where(is_exc, 0.001, 1000.0)
stp_params = STPParams(U=U, tau_rec=tau_rec, tau_fac=tau_fac)
stp_state = init_stp_state(N_NEURONS, stp_params)

n_steps = int(DURATION_MS / DT)
I_noise = BG_MEAN + BG_STD * jax.random.normal(k4, (n_steps, N_NEURONS))

def step_fn(carry, I_t):
    ns, ss, stp_st = carry
    I_syn = compute_synaptic_current(ss, ns.v)
    ns = izhikevich_step(ns, params, I_syn + I_t, DT)
    stp_st, scale = stp_step(stp_st, stp_params, ns.spikes, DT)
    new_ampa = ampa_step(ss.g_ampa, scale, W_ampa, DT)
    new_gaba = gaba_a_step(ss.g_gaba_a, scale, W_inh, DT)
    new_nmda_rise, new_nmda_decay, _ = nmda_step(
        ss.g_nmda_rise, ss.g_nmda_decay, scale, W_nmda, DT,
    )
    ss = SynapseState(
        g_ampa=new_ampa, g_gaba_a=new_gaba,
        g_nmda_rise=new_nmda_rise, g_nmda_decay=new_nmda_decay,
        g_gaba_b_rise=ss.g_gaba_b_rise, g_gaba_b_decay=ss.g_gaba_b_decay,
    )
    return (ns, ss, stp_st), ns.spikes

(_, _, _), spikes = jax.lax.scan(step_fn, (state, syn, stp_state), I_noise)
spikes.block_until_ready()
wall_time = time.perf_counter() - t0

print(f"  Simulation complete in {wall_time:.1f}s  (realtime ratio: {DURATION_MS/1000/wall_time:.1f}x)")
print()

# --- Compute statistics and compare ---
raster = np.asarray(spikes)
stats = compute_culture_statistics(
    raster, dt_ms=DT, burst_threshold_std=BURST_THRESHOLD_STD,
)

results = compare_statistics(stats, "wagenaar_2006")

n_pass = 0
n_fail = 0
n_total = 0

print("  Metric                          Sim Value     Reference Range       Status")
print("  " + "-" * 78)

for metric_key in sorted(results.keys()):
    r = results[metric_key]
    sim_val = r["sim_value"]
    ref_range = r["ref_range"]
    in_range = r["in_range"]
    n_total += 1

    if in_range is True:
        status = "PASS"
        n_pass += 1
    elif in_range is False:
        status = "FAIL"
        n_fail += 1
    else:
        status = "N/A"
        n_total -= 1  # don't count N/A

    if ref_range is not None:
        ref_str = f"[{ref_range[0]:.3g}, {ref_range[1]:.3g}]"
    else:
        ref_str = "N/A"

    val_str = "NaN" if math.isnan(sim_val) else f"{sim_val:.4g}"
    print(f"  {metric_key:<32s}  {val_str:<12s}  {ref_str:<20s}  {status}")

print()
print(f"  Result: {n_pass}/{n_total} metrics within published ranges")

# Exit code: 0 if all passed, 1 if any failed
# Print machine-readable summary on final line
print(f"BIO_RESULT:{n_pass}:{n_total}")
if n_fail > 0:
    sys.exit(1)
PYEOF
) || true

echo "$BIO_OUTPUT"
echo ""

# Parse the machine-readable line from output
BIO_LINE=$(echo "$BIO_OUTPUT" | grep "^BIO_RESULT:" | tail -1)
if [[ -n "$BIO_LINE" ]]; then
    BIO_PASS=$(echo "$BIO_LINE" | cut -d: -f2)
    BIO_TOTAL=$(echo "$BIO_LINE" | cut -d: -f3)
    BIOVALIDATION_DETAIL="${BIO_PASS}/${BIO_TOTAL} metrics in range"
    if [[ "$BIO_PASS" == "$BIO_TOTAL" ]]; then
        STATUS_BIOVALIDATION="PASS"
    else
        STATUS_BIOVALIDATION="FAIL"
    fi
else
    STATUS_BIOVALIDATION="FAIL"
    BIOVALIDATION_DETAIL="simulation or analysis error"
fi

STEP_END=$(date +%s)
echo "   Duration: $((STEP_END - STEP_START))s"
echo ""

# =========================================================================
# Step D: Long-duration stability (default 1 hour, skipped with --quick)
# =========================================================================
echo "======================================================================"
if $QUICK; then
    echo "  [D] LONG-DURATION STABILITY  (SKIPPED -- --quick mode)"
    STATUS_STABILITY="SKIP"
else
    echo "  [D] LONG-DURATION STABILITY  (${STABILITY_HOURS}h)"
fi
echo "======================================================================"
echo ""

if ! $QUICK; then
    STEP_START=$(date +%s)

    STABILITY_SCRIPT="$PROJECT_ROOT/scripts/validate_stability.py"
    if [[ -f "$STABILITY_SCRIPT" ]]; then
        if "$VENV_PYTHON" "$STABILITY_SCRIPT" --duration-hours "$STABILITY_HOURS" 2>&1; then
            STATUS_STABILITY="PASS"
            echo ""
            echo "   Stability test PASSED."
        else
            STATUS_STABILITY="FAIL"
            echo ""
            echo "   WARNING: Stability test FAILED."
        fi
    else
        STATUS_STABILITY="SKIP"
        echo "   SKIPPED: $STABILITY_SCRIPT not found."
        echo "   Create this script to enable long-duration stability testing."
    fi

    STEP_END=$(date +%s)
    echo "   Duration: $((STEP_END - STEP_START))s"
else
    echo "   Skipped (use full mode to run: omit --quick flag)."
fi
echo ""

# =========================================================================
# Step E: Doom readiness check (optional)
# =========================================================================
echo "======================================================================"
echo "  [E] DOOM READINESS CHECK  (optional)"
echo "======================================================================"
echo ""

STEP_START=$(date +%s)

if "$VENV_PYTHON" -c "import vizdoom" 2>/dev/null; then
    echo "   vizdoom detected, running Doom tests..."
    echo ""
    if "$VENV_PYTEST" "$PROJECT_ROOT/tests/test_doom.py" -v 2>&1; then
        STATUS_DOOM="PASS"
        echo ""
        echo "   Doom tests PASSED."
    else
        STATUS_DOOM="FAIL"
        echo ""
        echo "   WARNING: Doom tests FAILED."
    fi
else
    STATUS_DOOM="SKIP"
    echo "   Skipping Doom (vizdoom not installed)."
fi

STEP_END=$(date +%s)
echo "   Duration: $((STEP_END - STEP_START))s"
echo ""

# =========================================================================
# Summary
# =========================================================================
SUITE_END=$(date +%s)
SUITE_DURATION=$((SUITE_END - SUITE_START))
SUITE_MINUTES=$((SUITE_DURATION / 60))
SUITE_SECONDS=$((SUITE_DURATION % 60))

echo "======================================================================"
echo "  VALIDATION SUMMARY"
echo "======================================================================"
echo ""
echo "  Step                    Result"
echo "  ------------------------------------------------"
printf "  %-24s %s\n" "Tests"              "$STATUS_TESTS"
if [[ "$STATUS_BENCHMARKS" == "PASS" ]]; then
    printf "  %-24s %s\n" "Benchmarks"     "PASS (results in $RESULTS_DIR/)"
elif [[ "$STATUS_BENCHMARKS" == "SKIP" ]]; then
    printf "  %-24s %s\n" "Benchmarks"     "SKIPPED"
else
    printf "  %-24s %s\n" "Benchmarks"     "$STATUS_BENCHMARKS"
fi
printf "  %-24s %s (%s)\n" "Bio validation" "$STATUS_BIOVALIDATION" "$BIOVALIDATION_DETAIL"
if [[ "$STATUS_STABILITY" == "SKIP" ]] && $QUICK; then
    printf "  %-24s %s\n" "Stability"      "SKIPPED (--quick)"
elif [[ "$STATUS_STABILITY" == "SKIP" ]]; then
    printf "  %-24s %s\n" "Stability"      "SKIPPED (script missing)"
else
    printf "  %-24s %s\n" "Stability"      "$STATUS_STABILITY"
fi
if [[ "$STATUS_DOOM" == "SKIP" ]]; then
    printf "  %-24s %s\n" "Doom"           "SKIPPED (vizdoom not installed)"
else
    printf "  %-24s %s\n" "Doom"           "$STATUS_DOOM"
fi
echo "  ------------------------------------------------"
echo "  Total time: ${SUITE_MINUTES}m ${SUITE_SECONDS}s"
echo ""
echo "======================================================================"
echo "  Done.  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================================================================"

# Exit with failure if any critical step failed
if [[ "$STATUS_TESTS" == "FAIL" || "$STATUS_BIOVALIDATION" == "FAIL" ]]; then
    exit 1
fi
