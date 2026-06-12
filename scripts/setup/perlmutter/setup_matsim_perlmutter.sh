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
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "${MATSIM_AGENTS_ROOT}/.." && pwd)/HydraGNN}"

# Canonical install/runtime env path (matches install_matsim_perlmutter.sh:
# the venv lives inside HydraGNN's install root alongside its build deps).
SHARED_VENV="${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv"

# Legacy local env path (deprecated; only used as a last-resort fallback).
LOCAL_VENV="${MATSIM_AGENTS_ROOT}/perlmutter_venv"

# Choose which environment to use:
# 1) explicit override via MATSIM_PERLMUTTER_VENV
# 2) canonical shared env inside HydraGNN (matches the installer)
# 3) legacy local env in this repo (backward compatibility only)
if [[ -n "${MATSIM_PERLMUTTER_VENV:-}" ]]; then
    HYDRAGNN_VENV="${MATSIM_PERLMUTTER_VENV}"
    ENV_SOURCE="override"
elif [[ -d "${SHARED_VENV}" ]]; then
    HYDRAGNN_VENV="${SHARED_VENV}"
    ENV_SOURCE="shared"
elif [[ -d "${LOCAL_VENV}" || -L "${LOCAL_VENV}" ]]; then
    HYDRAGNN_VENV="${LOCAL_VENV}"
    ENV_SOURCE="local"
else
    HYDRAGNN_VENV="${SHARED_VENV}"
    ENV_SOURCE="missing"
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
    echo "Install it with:"
    echo "   bash ${SCRIPT_DIR}/install_matsim_perlmutter.sh [--gpu]"
    echo ""
    echo "Tip: override env path with MATSIM_PERLMUTTER_VENV=/path/to/env"
    return 1 2>/dev/null || exit 1
fi

echo "Activating HydraGNN environment..."
echo "Using ${ENV_SOURCE} environment: ${HYDRAGNN_VENV}"

# This is a conda environment, activate it by updating PATH
export PATH="${HYDRAGNN_VENV}/bin:${PATH}"
export CONDA_PREFIX="${HYDRAGNN_VENV}"
export CONDA_DEFAULT_ENV="matsim-agents"
export VIRTUAL_ENV="${HYDRAGNN_VENV}"

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
