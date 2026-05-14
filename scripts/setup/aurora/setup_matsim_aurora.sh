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
DEFAULT_VENV="${MATSIM_AGENTS_ROOT}/aurora_venv"
AURORA_VENV="${MATSIM_AURORA_VENV:-${DEFAULT_VENV}}"

if [[ ! -d "${AURORA_VENV}" ]]; then
    echo "Error: Aurora virtual environment not found at ${AURORA_VENV}" >&2
    echo "Create it with:" >&2
    echo "  bash ${SCRIPT_DIR}/install_matsim_aurora.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${AURORA_VENV}/bin/activate"

export PYTHONPATH="${MATSIM_AGENTS_ROOT}/src:${PYTHONPATH:-}"

echo "================================"
echo "matsim-agents Aurora Setup"
echo "================================"
echo "Using venv: ${AURORA_VENV}"
echo "Python: $(which python)"
echo "matsim-agents root: ${MATSIM_AGENTS_ROOT}"
echo ""
echo "Environment setup complete."
