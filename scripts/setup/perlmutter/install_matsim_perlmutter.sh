#!/usr/bin/env bash
# =============================================================================
# install_matsim_perlmutter.sh
#
# Full two-phase installation of matsim-agents + HydraGNN dependencies on
# NERSC Perlmutter (CUDA 12.4 / A100).
#
# This script mimics the methodology used for Frontier, providing a complete,
# isolated conda environment for matsim-agents development and execution.
#
# Phases:
# -------
# Phase 1 – Create/activate conda environment
# Phase 2 – Install HydraGNN dependencies + PyTorch + matsim-agents
#
# Usage:
# ------
#   bash install_matsim_perlmutter.sh [--skip-hydragnn] [--gpu]
#
# Flags:
# ------
#   --skip-hydragnn   Skip Phase 1 (conda env creation). Useful when the env
#                     already exists and only matsim-agents needs reinstalling.
#   --gpu             Load GPU-specific modules (craype-accel-nvidia80).
#
# Configurable variables (set via environment before calling this script):
# -------------------------------------------------------------------------
#   PROJECT_DIR      Root project directory       (default: directory of this script)
#   MATSIM_DIR       matsim-agents checkout path  (default: $PROJECT_DIR)
#   HYDRAGNN_DIR     HydraGNN checkout path       (default: parent dir of MATSIM_DIR)
#   VENV_PATH        Conda env path               (default: $MATSIM_DIR/perlmutter_venv)
#   PYTHON_VERSION   Python version               (default: 3.11)
#   TORCH_CUDA_TAG   PyTorch CUDA version tag     (default: cu124)
#   TORCH_CUDA_ARCH  GPU compute capability       (default: 8.0 for A100)
#   MAX_JOBS         Parallel build jobs          (default: 16)
#
# Examples:
# ---------
#   # First-time full installation
#   bash install_matsim_perlmutter.sh --gpu
#
#   # Reinstall in existing environment (faster)
#   bash install_matsim_perlmutter.sh --skip-hydragnn --gpu
#
# =============================================================================
set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../..}"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"

# Environment configuration
VENV_PATH="${VENV_PATH:-${MATSIM_DIR}/perlmutter_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu124}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.0}"  # A100
MAX_JOBS="${MAX_JOBS:-16}"
INSTALL_VLLM_SERVER="${INSTALL_VLLM_SERVER:-0}"

# Parse command-line arguments
SKIP_HYDRAGNN=0
USE_GPU=0

for arg in "$@"; do
    [[ "$arg" == "--skip-hydragnn" ]] && SKIP_HYDRAGNN=1
    [[ "$arg" == "--gpu" ]] && USE_GPU=1
done

# ── Helper functions ──────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Step 0: Load Perlmutter modules ───────────────────────────────────────────
log "Loading Perlmutter modules..."

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        source /usr/share/Modules/init/bash
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    die "module command not found. Ensure you're running on a Perlmutter login node."
fi

# Perform Cray "hard reset"
if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
    source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
fi

module reset
ml nersc-default/1.0 || true
ml cpe/24.07
ml PrgEnv-gnu/8.5.0
ml cray-mpich/8.1.30

if [[ "$USE_GPU" -eq 1 ]]; then
    log "Loading GPU (A100) support..."
    ml craype-accel-nvidia80
fi

ml cudatoolkit/12.4
ml gcc-native/13.2
ml cmake/3.30.2 || ml cmake/3.24.3 || true
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true

log "✓ Perlmutter modules loaded (ROCm $(if [[ "$USE_GPU" -eq 1 ]]; then echo "GPU"; else echo "CPU"; fi))."

# ── Step 1: Initialize Conda and create/activate environment ──────────────────
log "Initializing conda environment..."

if ! command -v conda >/dev/null 2>&1; then
    die "conda command not found (Miniforge module not loaded?)"
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)" 2>/dev/null || true
fi

log "Virtual environment path: ${VENV_PATH}"
log "Python version: ${PYTHON_VERSION}"

if [[ "$SKIP_HYDRAGNN" -eq 0 ]]; then
    if [[ -d "$VENV_PATH" ]]; then
        warn "Conda environment already exists at ${VENV_PATH}"
        warn "To recreate, remove it first: rm -rf ${VENV_PATH}"
    else
        log "Creating conda environment at ${VENV_PATH} with Python ${PYTHON_VERSION}..."
        conda create -y -p "$VENV_PATH" python="${PYTHON_VERSION}"
    fi
else
    warn "--skip-hydragnn set: skipping Phase 1."
    if [[ ! -d "$VENV_PATH" ]]; then
        die "Expected conda env not found at: ${VENV_PATH}\nRun without --skip-hydragnn first."
    fi
fi

conda activate "$VENV_PATH"
log "✓ Conda environment activated: $(which python)"
python --version

# ── Step 2: Install dependencies and matsim-agents ──────────────────────────
log "Installing Python packages and matsim-agents..."

# Cray libs (often helpful)
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# Force modern compiler for C++ extensions
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

log "Compiler check (must be GCC >= 9)"
echo "CC=$(which ${CC})"
echo "CXX=$(which ${CXX})"
${CXX} --version | head -n 1

# CUDA/PyTorch build environment hints
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH}"
export MAX_JOBS="${MAX_JOBS}"

if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(dirname "$(which nvcc)")")" && pwd)"
    log "CUDA_HOME=${CUDA_HOME}"
fi

# ── pip helpers ───────────────────────────────────────────────────────────────
log "Upgrading pip/setuptools/wheel..."

PIP_FLAGS=(--upgrade-strategy only-if-needed)

pip_retry() {
    local tries=3 delay=3
    for ((i=1; i<=tries; i++)); do
        if pip install "${PIP_FLAGS[@]}" "$@"; then
            return 0
        fi
        warn "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
        sleep "$delay"; delay=$((delay*2))
    done
    return 1
}

pip_retry --disable-pip-version-check -U pip setuptools wheel

# ── Install NumPy with version pin ────────────────────────────────────────────
log "Installing NumPy 1.26.4 (pinned)..."
pip_retry "numpy==1.26.4"

python -c "import numpy as np; assert np.__version__=='1.26.4', f'NumPy is {np.__version__}, expected 1.26.4'"
log "✓ NumPy 1.26.4 installed"

# ── Install core scientific Python deps ────────────────────────────────────────
log "Installing core Python packages (ninja, cython, packaging)..."
pip_retry ninja cython packaging

# ── Install PyTorch with CUDA support ─────────────────────────────────────────
log "Installing PyTorch from CUDA ${TORCH_CUDA_TAG} index..."
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
log "PyTorch index: ${PYTORCH_INDEX_URL}"

pip_retry --index-url "${PYTORCH_INDEX_URL}" torch torchvision
log "✓ PyTorch installed with CUDA support"

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# ── Install HydraGNN and dependencies ──────────────────────────────────────────
log "Installing HydraGNN dependencies (PyG, PyTorch Scatter, etc.)..."

# PyTorch Geometric (may need source build on Perlmutter to avoid GLIBC issues)
pip_retry torch-scatter torch-sparse torch-cluster torch-geometric

# HydraGNN core dependencies
pip_retry "ase>=3.22.1" h5py "setuptools>=68.0" "cmake>=3.24"

log "✓ HydraGNN dependencies installed"

# ── Install matsim-agents ─────────────────────────────────────────────────────
log "Installing matsim-agents..."

if [[ ! -d "${MATSIM_DIR}" ]]; then
    die "matsim-agents directory not found at: ${MATSIM_DIR}"
fi

cd "${MATSIM_DIR}"

if [[ -f "pyproject.toml" ]]; then
    log "Installing matsim-agents from ${MATSIM_DIR}..."
    pip_retry -e ".[dev]"
elif [[ -f "setup.py" ]]; then
    log "Installing matsim-agents from ${MATSIM_DIR} (setup.py)..."
    pip_retry -e "."
else
    warn "No pyproject.toml or setup.py found; attempting to add PYTHONPATH instead"
    export PYTHONPATH="${MATSIM_DIR}/src:${PYTHONPATH:-}"
    log "PYTHONPATH=${PYTHONPATH}"
fi

log "✓ matsim-agents installed"

# ── Ensure core runtime/test deps are present even on partial reinstalls ─────
# These are required by the test suite (tests/conftest.py imports
# langchain_core.fake_chat_models) and by local sanity checks.
log "Ensuring core runtime/test dependencies (langchain-core, pytest, pytest-cov)..."
pip_retry "langchain-core>=0.3.0" "pytest>=8.0" "pytest-cov>=5.0"

# ── Install LLM download/runtime tooling (session-learned extras) ───────────
# We explicitly install huggingface_hub so `hf` is available for resumable
# model downloads on login nodes, and transformers for local model loading.
log "Installing LLM tooling extras (huggingface_hub CLI + transformers)..."
pip_retry "huggingface_hub>=1.12"
pip_retry "transformers>=4.45"

if [[ "${INSTALL_VLLM_SERVER}" == "1" ]]; then
    log "INSTALL_VLLM_SERVER=1 -> installing vLLM server package"
    pip_retry vllm
fi

log "✓ LLM tooling extras installed"

# ── Final verification ────────────────────────────────────────────────────────
log "Verifying installation..."

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric installed')" 2>/dev/null || warn "PyTorch Geometric not fully available (may be OK)"
python -c "import ase; print(f'ASE: {ase.__version__}')"
python -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
if command -v hf >/dev/null 2>&1; then
    log "hf CLI: $(hf --version)"
else
    warn "hf CLI not found in PATH after install"
fi

if python -c "import matsim_agents" 2>/dev/null; then
    log "✓ matsim-agents Python module imported successfully"
else
    warn "matsim-agents module not importable; check installation"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Installation complete!"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Environment details:"
log "  Conda environment: ${VENV_PATH}"
log "  Python:           $(which python) ($(python --version 2>&1))"
log "  CUDA:             $(which nvcc 2>/dev/null || echo 'not found')"
log ""
log "To activate this environment in future sessions:"
log "  source ${SCRIPT_DIR}/perlmutter-module-stack.sh"
log "  load_perlmutter_modules$(if [[ "$USE_GPU" -eq 1 ]]; then echo "_gpu"; fi)"
log "  conda activate ${VENV_PATH}"
log ""
log "Or use the quick setup script:"
log "  source ${SCRIPT_DIR}/setup_matsim_perlmutter.sh$(if [[ "$USE_GPU" -eq 1 ]]; then echo " --gpu"; fi)"
log ""
