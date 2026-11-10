#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ "${NBTAO5_REPO_ROOT:-}" == "${REPO}" ]] || {
  echo "Set NBTAO5_REPO_ROOT to this absolute checkout path." >&2
  exit 2
}
[[ -x "${REPO}/.venv/bin/python" ]] || {
  echo "Missing checkout Python: ${REPO}/.venv/bin/python" >&2
  exit 2
}
exec "${REPO}/.venv/bin/python" "${REPO}/scripts/nbtao5_q1_gate.py" submit-preflight "$@"