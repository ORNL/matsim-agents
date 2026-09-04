#!/bin/bash
# setup_matsim_perlmutter.sh
# Setup environment for matsim-agents on NERSC Perlmutter
# This script loads Perlmutter modules and activates the matsim-owned environment.
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
DEFAULT_VENV="${MATSIM_AGENTS_ROOT}/.venv"
MATSIM_VENV="${MATSIM_PERLMUTTER_VENV:-${DEFAULT_VENV}}"

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

# Activate the matsim-owned Python environment.
if [[ ! -d "${MATSIM_VENV}" ]]; then
    echo "Error: matsim environment not found at:"
    echo "   ${MATSIM_VENV}"
    echo ""
    echo "Install it with:"
    echo "   bash ${SCRIPT_DIR}/install.sh"
    echo ""
    echo "Tip: override env path with MATSIM_PERLMUTTER_VENV=/path/to/env"
    return 1 2>/dev/null || exit 1
fi

echo "Activating matsim environment: ${MATSIM_VENV}"

# This is a conda environment, activate it by updating PATH
export PATH="${MATSIM_VENV}/bin:${PATH}"
export CONDA_PREFIX="${MATSIM_VENV}"
export CONDA_DEFAULT_ENV="matsim-agents"
export VIRTUAL_ENV="${MATSIM_VENV}"

# Ensure NVIDIA runtime libs used by VASP are resolvable at runtime.
NVIDIA_SDK_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5"
for libdir in \
    "${NVIDIA_SDK_ROOT}/compilers/lib" \
    "${NVIDIA_SDK_ROOT}/compilers/extras/qd/lib"; do
    if [[ -d "${libdir}" ]] && [[ ":${LD_LIBRARY_PATH:-}:" != *":${libdir}:"* ]]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
    fi
done

# Set PYTHONPATH to include matsim-agents
MATSIM_AGENTS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${MATSIM_AGENTS_DIR}/src:${PYTHONPATH:-}"

# Add the HydraGNN sc26 example dir to PYTHONPATH so `inference_fused`
# (used by src/matsim_agents/tools/relaxation.py) can be imported.
HYDRAGNN_SC26_DIR="${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26"
if [[ -f "${HYDRAGNN_SC26_DIR}/inference_fused.py" ]]; then
    export PYTHONPATH="${HYDRAGNN_SC26_DIR}:${PYTHONPATH}"
fi

echo ""
echo "✓ Environment setup complete!"
echo "  - Python: $(which python)"
echo "  - CUDA: $(which nvcc || echo 'not found')"
echo "  - matsim-agents path: ${MATSIM_AGENTS_DIR}"
echo ""
