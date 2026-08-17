#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# matsim-agents: Quantum ESPRESSO ``pw.x`` GPU launcher for Aurora.
#
# Invoked as:
#   deployments/aurora/launchers/run-pw-gpu-aurora.sh <pw-input-file> [extra args]
#
# Environment overrides:
#   QE_PREFIX        = <repo>/external/quantum-espresso
#   PW_BIN           = ${QE_PREFIX}/install-gpu/bin/pw.x
#   NRANKS           = 6
#   OMP_NUM_THREADS  = 4
#   GPU_BIND         = closest
#
# Notes:
# - Uses Aurora frameworks module stack.
# - Runs with srun when available, otherwise falls back to mpiexec.
# ---------------------------------------------------------------------------

set -euo pipefail
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
if [[ ! -f "${REPO}/pyproject.toml" ]]; then
  echo "ERROR: Could not detect matsim-agents repo root from ${SCRIPT_DIR}" >&2
  exit 2
fi

QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
PW_BIN="${PW_BIN:-${QE_PREFIX}/install-gpu/bin/pw.x}"
NRANKS="${NRANKS:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPU_BIND="${GPU_BIND:-closest}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pw-input-file> [extra pw.x args...]" >&2
  exit 2
fi
INPUT="$1"; shift
[[ ! -f "${INPUT}" ]] && { echo "Input file not found: ${INPUT}" >&2; exit 2; }
[[ ! -x "${PW_BIN}" ]] && { echo "pw.x not found or not executable: ${PW_BIN}" >&2; exit 2; }

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  fi
fi

if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi

# MPICH + Intel GPU run-time defaults used on Aurora stacks.
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

echo "=========================================="
echo "QE pw.x GPU run on Aurora"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "pw.x:        ${PW_BIN}"
echo "Input:       ${INPUT}"
echo "MPI ranks:   ${NRANKS}    OMP threads/rank: ${OMP_NUM_THREADS}"
echo "=========================================="

if command -v srun >/dev/null 2>&1; then
  # Keep launch semantics aligned with Frontier/Perlmutter launchers.
  srun -n "${NRANKS}" -c "${OMP_NUM_THREADS}" --gpu-bind="${GPU_BIND}" \
       "${PW_BIN}" -in "${INPUT}" "$@"
elif command -v mpiexec >/dev/null 2>&1; then
  mpiexec -n "${NRANKS}" "${PW_BIN}" -in "${INPUT}" "$@"
else
  echo "ERROR: neither srun nor mpiexec found for launching MPI job" >&2
  exit 2
fi
