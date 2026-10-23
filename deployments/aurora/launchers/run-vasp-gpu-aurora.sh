#!/usr/bin/env bash
# matsim-agents MATSIM_VASP_LAUNCHER for Aurora: runs VASP in the current
# directory (INCAR/POSCAR/KPOINTS/POTCAR), no argv, via _vasp-step-aurora.sh.
# Tunables: VASP_BIN, VASP_ROOT, VASP_VARIANT=std, NRANKS=12, OMP_NUM_THREADS=8,
#   ONEAPI_MODULE, GPU_UMD_MODULE (must match build-vasp-gpu-aurora.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.1}"
VASP_BIN="${VASP_BIN:-${VASP_ROOT}/bin/vasp_${VASP_VARIANT:-std}}"

if ! command -v module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [[ -f "${init}" ]] && { source "${init}"; break; }
  done
fi
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_oneapi_modules

# Single-node run on the first node of the PBS allocation unless the caller assigned one.
if [[ -z "${MATSIM_ASSIGNED_NODES:-}" ]]; then
  MATSIM_ASSIGNED_NODES="$(head -n1 "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || true)"
  export MATSIM_ASSIGNED_NODES="${MATSIM_ASSIGNED_NODES:-$(hostname)}"
fi

exec bash "${SCRIPT_DIR}/_vasp-step-aurora.sh" "${PWD}" "${VASP_BIN}" \
  1 "${NRANKS:-12}" "${OMP_NUM_THREADS:-8}"
