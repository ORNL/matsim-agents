#!/usr/bin/env bash
# Compatibility entry point for older benchmark instructions.
# Prefer: INSTALL_MACE=1 INSTALL_UMA=1 bash deployments/aurora/setup/install.sh
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"
MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${BASE_VENV}" MACE_VENV_PATH="${MACE_VENV_PATH}" \
    FACILITY=aurora bash "${MATSIM_DIR}/deployments/common/setup/install-mace-compat.sh"
