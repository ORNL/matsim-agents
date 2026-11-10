#!/bin/bash
# Source before facility placement. A setup script is trusted executable configuration.
set -euo pipefail
if [[ -n "${VASP_ENV_SETUP:-}" ]]; then
  [[ -f "${VASP_ENV_SETUP}" ]] || {
    echo "VASP environment setup not found: ${VASP_ENV_SETUP}" >&2; exit 2;
  }
  # shellcheck disable=SC1090
  source "${VASP_ENV_SETUP}"
fi
[[ -x "${VASP_BIN:?VASP_BIN must identify the executable}" ]] || {
  echo "VASP binary not executable: ${VASP_BIN}" >&2; exit 2;
}
if [[ -n "${VASP_EXPECTED_SHA256:-}" ]]; then
  [[ "${VASP_EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "Invalid VASP_EXPECTED_SHA256" >&2; exit 2;
  }
  printf '%s  %s\n' "${VASP_EXPECTED_SHA256}" "${VASP_BIN}" | sha256sum -c -
fi
echo "[vasp-runtime] executable=$(realpath "${VASP_BIN}") setup=${VASP_ENV_SETUP:-facility-default}"
sha256sum "${VASP_BIN}"
if command -v module >/dev/null 2>&1; then module -t list 2>&1; fi
