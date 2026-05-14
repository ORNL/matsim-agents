#!/usr/bin/env bash
# setup_matsim_frontier.sh
# Quick setup for matsim-agents on Frontier.
#
# Usage:
#   source setup_matsim_frontier.sh [--rocm72]
#
# Optional overrides:
#   HYDRAGNN_ROOT=/path/to/HydraGNN
#   MATSIM_FRONTIER_VENV=/path/to/conda_env
#   MATSIM_FRONTIER_ROCM_VERSION=7.1|7.2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_AGENTS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "${MATSIM_AGENTS_ROOT}/.." && pwd)/HydraGNN}"

ROCM_VERSION="${MATSIM_FRONTIER_ROCM_VERSION:-7.1}"
if [[ "${1:-}" == "--rocm72" ]]; then
    ROCM_VERSION="7.2"
fi

VENV_71="${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv"
VENV_72="${HYDRAGNN_ROOT}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72"

if [[ "${ROCM_VERSION}" == "7.2" ]]; then
    DEFAULT_VENV="${VENV_72}"
else
    DEFAULT_VENV="${VENV_71}"
fi
FRONTIER_VENV="${MATSIM_FRONTIER_VENV:-${DEFAULT_VENV}}"

echo "================================"
echo "matsim-agents Frontier Setup"
echo "================================"

echo "Loading Frontier modules (ROCm ${ROCM_VERSION})..."
source "${SCRIPT_DIR}/frontier-module-stack.sh"
if [[ "${ROCM_VERSION}" == "7.2" ]]; then
    load_frontier_rocm72_modules
else
    load_frontier_rocm711_modules
fi

if [[ ! -d "${FRONTIER_VENV}" ]]; then
    echo "Error: Frontier environment not found at ${FRONTIER_VENV}" >&2
    echo "Install it with:" >&2
    echo "  bash ${SCRIPT_DIR}/install_matsim_frontier.sh$( [[ "${ROCM_VERSION}" == "7.2" ]] && echo " --rocm72" )" >&2
    return 1 2>/dev/null || exit 1
fi

# Conda environment activation by PATH export (same style as Perlmutter setup)
export PATH="${FRONTIER_VENV}/bin:${PATH}"
export CONDA_PREFIX="${FRONTIER_VENV}"
export CONDA_DEFAULT_ENV="matsim-frontier"
export VIRTUAL_ENV="${FRONTIER_VENV}"
export PYTHONPATH="${MATSIM_AGENTS_ROOT}/src:${PYTHONPATH:-}"

echo "Using environment: ${FRONTIER_VENV}"
echo "Python: $(which python)"
echo "matsim-agents root: ${MATSIM_AGENTS_ROOT}"
echo ""
echo "Environment setup complete."
