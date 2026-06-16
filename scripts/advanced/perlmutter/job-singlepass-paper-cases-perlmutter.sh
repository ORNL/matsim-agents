#!/bin/bash
#SBATCH -J singlepass-paper
#SBATCH -A amsc001_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 02:00:00
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/singlepass-paper-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/singlepass-paper-%j/job-%j.err

# ---------------------------------------------------------------------------
# Single-pass (non-AL) feasibility sweep over the manuscript paper cases.
# Runs the planner -> executor -> uq_gate -> analyst graph ONCE per case with
# the shared HydraGNN multidataset BEST6 surrogate (MLP only, no DFT).
#
# Cases (5 ready; cu_bht skipped — needs a user-supplied CIF):
#   lifepo4 hea_bcc hea_fcc phosphorene zn_formate
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents}"
PROJ="$(dirname "${PROJECT_ROOT}")"
HYDRAGNN_EXAMPLE="${HYDRAGNN_EXAMPLE:-$PROJ/HydraGNN/examples/multidataset_hpo_sc26}"

# Executor reads these env vars (LangGraph filters unknown config keys).
export MATSIM_HYDRAGNN_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}"
export MATSIM_HYDRAGNN_MLP_CKPT="${MATSIM_HYDRAGNN_MLP_CKPT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}"

CASES="${CASES:-lifepo4 hea_bcc hea_fcc phosphorene zn_formate}"
RUN_DIR="${PROJ}/runs/singlepass-paper-${SLURM_JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_ROOT}"
source scripts/setup/perlmutter/setup_matsim_perlmutter.sh --gpu

export PROJ_ROOT="${PROJECT_ROOT}"
export OUT_DIR="${RUN_DIR}/out_singlepass"
# singlepass.py also passes mlp_checkpoint via config; keep it consistent.
export MLP_LOGDIR="${MATSIM_HYDRAGNN_LOGDIR}"
export MLP_CKPT="${MATSIM_HYDRAGNN_MLP_CKPT}"

echo "=========================================="
echo "Single-pass paper-case sweep"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Logdir:      ${MATSIM_HYDRAGNN_LOGDIR}"
echo "MLP ckpt:    ${MATSIM_HYDRAGNN_MLP_CKPT}"
echo "Cases:       ${CASES}"
echo "Out dir:     ${OUT_DIR}"
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
