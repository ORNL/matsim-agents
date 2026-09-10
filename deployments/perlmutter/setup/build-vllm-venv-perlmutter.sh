#!/usr/bin/env bash
# Compatibility entry point. Prefer INSTALL_VLLM=1 with this facility's install.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-${MATSIM_DIR}/venv_vllm}"
MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${BASE_VENV}" VLLM_VENV_PATH="${VLLM_VENV_PATH}" \
    bash "${MATSIM_DIR}/deployments/common/setup/install-vllm-compat.sh"
