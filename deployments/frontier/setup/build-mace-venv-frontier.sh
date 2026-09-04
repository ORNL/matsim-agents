#!/usr/bin/env bash
# Compatibility entry point. Prefer INSTALL_MACE=1 with this facility's install.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"
MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${BASE_VENV}" MACE_VENV_PATH="${MACE_VENV_PATH}" \
    FACILITY=frontier bash "${MATSIM_DIR}/deployments/common/setup/install-mace-compat.sh"
