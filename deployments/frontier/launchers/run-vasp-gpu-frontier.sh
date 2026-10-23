#!/bin/bash
# matsim-agents MATSIM_VASP_LAUNCHER for Frontier: runs VASP in the current
# directory (INCAR/POSCAR/KPOINTS/POTCAR), no argv, via _vasp-step-frontier.sh.
# Tunables: VASP_BIN, VASP_ROOT, VASP_VARIANT=std, NNODES=1, NRANKS=8, OMP_NUM_THREADS=7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.1}"
VASP_BIN="${VASP_BIN:-${VASP_ROOT}/bin/vasp_${VASP_VARIANT:-std}}"

exec bash "${SCRIPT_DIR}/_vasp-step-frontier.sh" "${PWD}" "${VASP_BIN}" \
  "${NNODES:-1}" "${NRANKS:-8}" "${OMP_NUM_THREADS:-7}"
