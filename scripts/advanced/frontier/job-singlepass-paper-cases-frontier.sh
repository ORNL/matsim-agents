#!/bin/bash
#SBATCH -J singlepass-paper
#SBATCH -A lrn070
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# ---------------------------------------------------------------------------
# Single-pass (non-AL) feasibility sweep over the manuscript paper cases on
# OLCF Frontier (AMD MI250X). Runs the planner -> executor -> uq_gate ->
# analyst graph ONCE per case with the shared HydraGNN multidataset BEST6
# surrogate (MLIP only, no DFT).
#
# Frontier analogue of job-singlepass-paper-cases-perlmutter.sh. Only the
# module stack, venv, and default paths differ.
#
# Cases (5 ready; cu_bht skipped — needs a user-supplied CIF):
#   lifepo4 hea_bcc hea_fcc phosphorene zn_formate
#
# Allocation-portable configuration (override via env vars):
#   PROJECT_ROOT      matsim-agents working copy (default: lrn070 checkout)
#   RUNS_ROOT         parent dir for job logs + run outputs (default: <PROJ>/runs)
#   HYDRAGNN_EXAMPLE  surrogate model tree (default: <PROJ>/HydraGNN example tree)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/orion/lrn070/proj-shared/mlupopa/matstim-folder/matsim-agents}"
PROJ="$(dirname "${PROJECT_ROOT}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"
HYDRAGNN_EXAMPLE="${HYDRAGNN_EXAMPLE:-${PROJ}/HydraGNN/examples/multidataset_hpo_sc26}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this file with sbatch, not bash." >&2
  exit 2
fi

# Executor reads these env vars (LangGraph filters unknown config keys).
export MATSIM_HYDRAGNN_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}"

CASES="${CASES:-lifepo4 hea_bcc hea_fcc phosphorene zn_formate}"
RUN_DIR="${RUNS_ROOT}/singlepass-paper-${SLURM_JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source scripts/setup/frontier/setup_matsim_frontier.sh --rocm72

export PROJ_ROOT="${PROJECT_ROOT}"
export OUT_DIR="${RUN_DIR}/out_singlepass"
export RUN_ID="singlepass-${SLURM_JOB_ID:-local-$$}"
export MLIP_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"

echo "=========================================="
echo "Single-pass paper-case sweep (Frontier)"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Account:     ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Project:     ${PROJECT_ROOT}"
echo "Runs root:   ${RUNS_ROOT}"
echo "Logdir:      ${MATSIM_HYDRAGNN_LOGDIR}"
echo "MLP ckpt:    ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "Cases:       ${CASES}"
echo "Run dir:     ${RUN_DIR}"
echo "Out dir:     ${OUT_DIR}"
echo "Run ID:      ${RUN_ID}"
echo "=========================================="

rc=0
for c in ${CASES}; do
  echo ""
  echo "########## CASE: ${c} ##########"
  if python examples/paper_cases/singlepass.py --case "${c}"; then
    echo "[OK] ${c}"
  else
    echo "[FAILED] ${c}"
    rc=1
  fi
done

echo ""
echo "ALL-SINGLEPASS-DONE (rc=${rc})"
exit "${rc}"
