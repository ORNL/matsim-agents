#!/bin/bash
#PBS -N matsim-active-learning-uq
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: active-learning loop on ALCF Aurora.
#
# Production AL contract using the shared scheduler-neutral runner.
#
# Submit:
#   qsub deployments/aurora/jobs/job-active-learning-uq-aurora.sh
#
# Override:
#   qsub -v MATSIM_STRUCTURES="a.vasp b.vasp",MATSIM_N_SELECT=4 \
#        deployments/aurora/jobs/job-active-learning-uq-aurora.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── repo / paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"

# Default candidate set: neutral fixtures that ship with the repo. Override
# with MATSIM_STRUCTURES="path1 path2 ..." for a real sweep.
DEFAULT_STRUCTURES=(
  "${REPO}/tests/integration/data/Si.vasp"
  "${REPO}/tests/integration/data/MgO.vasp"
  "${REPO}/tests/integration/data/NaCl.vasp"
)
STRUCTURE_LIST="${MATSIM_STRUCTURES:-${MATSIM_AL_STRUCTURES:-}}"
if [[ -n "${STRUCTURE_LIST}" ]]; then
  # shellcheck disable=SC2206
  STRUCTURES=( ${STRUCTURE_LIST} )
else
  STRUCTURES=( "${DEFAULT_STRUCTURES[@]}" )
fi

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/active-learning-uq-aurora-${JOBID}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"

# ── modules + venv ──────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH="${HYDRAGNN_EXAMPLE}:${PROJ}/HydraGNN:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# ── DFT launchers (the example will skip cleanly if these are unset) ────────
# Scheduler-step wrappers use the same backend contract as Frontier and
# Perlmutter and consume dispatcher-assigned, disjoint PBS node groups.
export MATSIM_DFT_BACKEND="${MATSIM_DFT_BACKEND:-qe}"
export MATSIM_PW_BIN="${MATSIM_PW_BIN:-${REPO}/external/quantum-espresso/install-gpu/bin/pw.x}"
export MATSIM_PSEUDO_DIR="${MATSIM_PSEUDO_DIR:-${REPO}/external/quantum-espresso/src/pseudo}"
export MATSIM_DFT_WRAPPER="${MATSIM_DFT_WRAPPER:-${REPO}/deployments/aurora/launchers/_qe-step-aurora.sh}"
export MATSIM_DFT_RANKS_PER_NODE="${MATSIM_DFT_RANKS_PER_NODE:-12}"
export MATSIM_DFT_THREADS_PER_RANK="${MATSIM_DFT_THREADS_PER_RANK:-1}"
export MATSIM_SEED_STRUCTURES="$(IFS=:; echo "${STRUCTURES[*]}")"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora] matsim-agents active-learning UQ loop"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Job ID:      ${JOBID}"
echo "Repo:        ${REPO}"
echo "Venv:        ${VENV}"
echo "Logdir:      ${LOGDIR}"
echo "MLP ckpt:    ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "Structures:  ${#STRUCTURES[@]}"
for s in "${STRUCTURES[@]}"; do echo "             - ${s}"; done
echo "DFT backend: ${MATSIM_DFT_BACKEND}"
echo "DFT wrapper: ${MATSIM_DFT_WRAPPER}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

# ── run the active-learning driver ──────────────────────────────────────────
source "${REPO}/deployments/common/run-active-learning.sh"

echo "[$(date)] Active-learning loop complete. Artifacts in ${OUTPUT_DIR}"
