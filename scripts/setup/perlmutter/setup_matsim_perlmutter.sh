#!/bin/bash
# setup_matsim_perlmutter.sh
# Setup environment for matsim-agents on NERSC Perlmutter
# This script loads necessary Perlmutter modules and activates the HydraGNN virtual environment
#
# Usage:
#   source setup_matsim_perlmutter.sh [--gpu]
#
# Options:
#   --gpu    Load GPU-specific modules (for A100 nodes)

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MATSIM_AGENTS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Check for local environment first (from fresh installation)
LOCAL_VENV="${MATSIM_AGENTS_ROOT}/perlmutter_venv"

# Fall back to shared HydraGNN environment
SHARED_VENV="/global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv"

# Choose which environment to use (prefer local, fall back to shared)
if [[ -d "${LOCAL_VENV}" ]]; then
    HYDRAGNN_VENV="${LOCAL_VENV}"
    USING_LOCAL_ENV=1
else
    HYDRAGNN_VENV="${SHARED_VENV}"
    USING_LOCAL_ENV=0
fi

# Check if we should use GPU modules
USE_GPU=false
if [[ "${1:-}" == "--gpu" ]]; then
    USE_GPU=true
fi

echo "================================"
echo "matsim-agents Perlmutter Setup"
echo "================================"

# Source the module stack
echo "Loading Perlmutter modules..."
source "${SCRIPT_DIR}/perlmutter-module-stack.sh"

if [[ "${USE_GPU}" == "true" ]]; then
    load_perlmutter_modules_gpu
else
    load_perlmutter_modules
fi

# Activate HydraGNN conda environment
if [[ ! -d "${HYDRAGNN_VENV}" ]]; then
    echo "❌ Error: HydraGNN environment not found at:"
    echo "   ${HYDRAGNN_VENV}"
    echo ""
    if [[ "$USING_LOCAL_ENV" -eq 1 ]]; then
        echo "Local environment not found. Install it first:"
        echo "   bash ${SCRIPT_DIR}/install_matsim_perlmutter.sh [--gpu]"
    else
        echo "Shared environment not found. Please install it:"
        echo "   bash ${SCRIPT_DIR}/install_matsim_perlmutter.sh [--gpu]"
    fi
    return 1 2>/dev/null || exit 1
fi

echo "Activating HydraGNN environment..."
if [[ "$USING_LOCAL_ENV" -eq 1 ]]; then
    echo "Using local environment: ${HYDRAGNN_VENV}"
else
    echo "Using shared environment: ${HYDRAGNN_VENV}"
fi

# This is a conda environment, activate it by updating PATH
export PATH="${HYDRAGNN_VENV}/bin:${PATH}"
export CONDA_PREFIX="${HYDRAGNN_VENV}"
export CONDA_DEFAULT_ENV="matsim-agents"

# Set PYTHONPATH to include matsim-agents
MATSIM_AGENTS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${MATSIM_AGENTS_DIR}/src:${PYTHONPATH:-}"

echo ""
echo "✓ Environment setup complete!"
echo "  - Python: $(which python)"
echo "  - CUDA: $(which nvcc || echo 'not found')"
echo "  - matsim-agents path: ${MATSIM_AGENTS_DIR}"
echo ""
