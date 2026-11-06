#!/usr/bin/env bash
# setup_matsim_aurora.sh
# Quick setup for matsim-agents on Aurora-style hosts.
#
# Usage:
#   source setup_matsim_aurora.sh
#
# Optional overrides:
#   MATSIM_AURORA_VENV=/path/to/venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_AGENTS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_VENV="${MATSIM_AGENTS_ROOT}/.venv"
AURORA_VENV="${MATSIM_AURORA_VENV:-${DEFAULT_VENV}}"

if [[ ! -d "${AURORA_VENV}" ]]; then
    echo "Error: Aurora virtual environment not found at ${AURORA_VENV}" >&2
    echo "Create it with:" >&2
    echo "  bash ${SCRIPT_DIR}/install.sh" >&2
    return 1 2>/dev/null || exit 1
fi

source "${SCRIPT_DIR}/aurora-python-env.sh"
activate_matsim_aurora_python "${MATSIM_AGENTS_ROOT}" "${AURORA_VENV}"

echo "================================"
echo "matsim-agents Aurora Setup"
echo "================================"
echo "Using venv: ${AURORA_VENV}"
echo "Python: $(which python)"
echo "matsim-agents root: ${MATSIM_AGENTS_ROOT}"
echo "Runtime defaults:"
source "${MATSIM_AGENTS_ROOT}/deployments/common/setup/matsim-runtime-env.sh" aurora "${MATSIM_AGENTS_ROOT}"
echo ""
echo "Environment setup complete."
