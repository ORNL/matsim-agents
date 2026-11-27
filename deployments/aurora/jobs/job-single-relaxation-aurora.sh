#!/bin/bash
#PBS -N matsim-single-relax
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: typed, provenance-tracked MLIP relaxation on Aurora.
#
# Submit:
#   qsub deployments/aurora/jobs/job-single-relaxation-aurora.sh
#
# Override at submit time:
#   qsub -v MATSIM_STRUCTURE=tests/integration/data/Si.vasp \
#        deployments/aurora/jobs/job-single-relaxation-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── repo / paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
source "${REPO}/deployments/common/hydragnn-env.sh"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_BEST6_LOGDIR}}"
STRUCTURE="${MATSIM_STRUCTURE:-${REPO}/tests/integration/data/Si.vasp}"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/single-relaxation-aurora-${JOBID}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"

# ── modules + venv ──────────────────────────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

hydragnn_export_pythonpath
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# executor_node reads these env vars as fallback when config injection is
# unavailable (e.g. across LangGraph checkpoint boundaries).
export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora] matsim-agents single_relaxation smoke test"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Job ID:      ${JOBID}"
echo "Repo:        ${REPO}"
echo "Venv:        ${VENV}"
echo "Structure:   ${STRUCTURE}"
echo "Logdir:      ${LOGDIR}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

# ── run the typed relaxation contract ───────────────────────────────────────
source "${REPO}/deployments/common/run-mlip-relaxation.sh"

echo "[$(date)] Single-relaxation smoke test complete. Artifacts in ${OUTPUT_DIR}"
