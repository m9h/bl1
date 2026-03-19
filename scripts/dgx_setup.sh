#!/usr/bin/env bash
# ===========================================================================
# dgx_setup.sh -- Prepare NVIDIA DGX Spark for bl1 validation runs
#
# Expects: DGX Spark with Blackwell GPU, CUDA 12, Python 3.10+
# Creates: .venv at project root, installs JAX (CUDA 12) + bl1 + extras
# Safe to re-run (idempotent).
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# ── Banner ─────────────────────────────────────────────────────────────────
echo "==========================================================="
echo "  bl1 DGX Spark Setup"
echo "==========================================================="
echo ""

# ── 1. Auto-detect environment ─────────────────────────────────────────────
echo "--- Environment detection ---"

# CUDA / GPU via nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)

    # Parse CUDA version from the top-right of nvidia-smi output
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")

    echo "  GPU:            $GPU_NAME"
    echo "  Driver:         $DRIVER_VERSION"
    echo "  CUDA version:   $CUDA_VERSION"
else
    echo "  WARNING: nvidia-smi not found. GPU acceleration may not work."
    GPU_NAME="(not detected)"
    DRIVER_VERSION="(not detected)"
    CUDA_VERSION="(not detected)"
fi

# Python
PYTHON_VERSION=$(python3 --version 2>&1 || echo "Python not found")
echo "  Python:         $PYTHON_VERSION"
echo ""

# ── 2. Create virtual environment (skip if it already exists) ──────────────
echo "--- Virtual environment ---"
if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists at $VENV_DIR -- reusing."
else
    echo "  Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "  Created."
fi

# Activate the venv for the rest of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  Active Python: $(which python)"
echo ""

# Upgrade pip/setuptools first (avoids build issues)
echo "--- Upgrading pip & setuptools ---"
pip install --upgrade pip setuptools wheel --quiet
echo "  Done."
echo ""

# ── 3. Install JAX with GPU support ───────────────────────────────────────
# Auto-detect CUDA version and install the appropriate JAX variant.
echo "--- Installing JAX with GPU support ---"
if [[ "$CUDA_VERSION" == 13.* ]]; then
    echo "  Detected CUDA 13 — installing jax[cuda13]"
    pip install --upgrade "jax[cuda13]" --quiet
elif [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "  Detected CUDA 12 — installing jax[cuda12]"
    pip install --upgrade "jax[cuda12]" --quiet
else
    echo "  CUDA version '$CUDA_VERSION' — trying jax[cuda13] (latest)"
    pip install --upgrade "jax[cuda13]" --quiet
fi
echo "  Done."
echo ""

# ── 4. Install bl1 in editable mode with all extras ───────────────────────
echo "--- Installing bl1 (editable, dev+quality+docs extras) ---"
pip install -e "$PROJECT_ROOT[dev,quality,docs]" --quiet
echo "  Done."
echo ""

# ── 5. Try to install vizdoom (optional -- may fail on some systems) ──────
echo "--- Installing vizdoom (optional) ---"
if pip install vizdoom --quiet 2>/dev/null; then
    VIZDOOM_STATUS="installed"
    echo "  vizdoom installed."
else
    VIZDOOM_STATUS="SKIPPED (install failed -- not critical)"
    echo "  WARNING: vizdoom install failed. This is non-critical."
    echo "  The Doom-environment integration tests will be skipped."
fi
echo ""

# ── 6. Verify installation ────────────────────────────────────────────────
echo "--- Verification ---"

echo "  JAX:"
python -c "import jax; print(f'    version:  {jax.__version__}'); print(f'    backend:  {jax.default_backend()}'); print(f'    devices:  {jax.devices()}')"

echo "  bl1:"
python -c "import bl1; print('    import OK')"

echo ""

# ── 7. Summary ─────────────────────────────────────────────────────────────
echo "==========================================================="
echo "  Setup complete"
echo "==========================================================="
echo ""
echo "  GPU:        $GPU_NAME"
echo "  CUDA:       $CUDA_VERSION"
echo "  Driver:     $DRIVER_VERSION"
echo "  Python:     $PYTHON_VERSION"
echo "  Venv:       $VENV_DIR"
echo "  vizdoom:    $VIZDOOM_STATUS"
echo ""
echo "  To activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Quick smoke test:"
echo "    python -m pytest tests/ -x -q"
echo ""
echo "==========================================================="
