#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the persistent installation log directory}"
mkdir -p "${RUN_DIR}"

finish() {
    status=$?
    trap - EXIT
    if [[ "${status}" -eq 0 ]]; then
        printf 'SUCCEEDED exit_code=0 finished=%s\n' "$(date -Is)" > "${RUN_DIR}/status.txt"
    else
        printf 'FAILED exit_code=%s finished=%s\n' "${status}" "$(date -Is)" > "${RUN_DIR}/status.txt"
    fi
    exit "${status}"
}
trap finish EXIT

printf 'RUNNING started=%s pid=%s\n' "$(date -Is)" "$$" > "${RUN_DIR}/status.txt"
printf 'Host: %s\nBranch: %s\nCommit: %s\n' "$(hostname)" \
    "$(git -C "${MATSIM_DIR}" branch --show-current)" "$(git -C "${MATSIM_DIR}" rev-parse HEAD)"
export MATSIM_DIR
export RUN_MODEL_TESTS=1
export PYTHONUNBUFFERED=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --junitxml=${RUN_DIR}/tests.xml"
bash "${SCRIPT_DIR}/install.sh"