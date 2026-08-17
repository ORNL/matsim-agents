#!/bin/bash
#PBS -A CM2US
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
# For each input structure:
#   1. Relax with HydraGNN (multi-branch MLFF).
#   2. Score uncertainty from the per-step branch-weight CSV
#      (mean top-branch weight + mean normalized entropy).
#   3. If the prediction is flagged unreliable, trigger TWO reference
#      DFT calculations on the optimized structure:
#        - Quantum ESPRESSO pw.x   (Aurora launcher: run-pw-gpu-aurora.sh)
#        - VASP vasp_std           (Aurora launcher: provided via env var)
#   4. Append the flagged structures to training_candidates.csv for the
#      next HydraGNN training round.
#
# Submit:
#   qsub deployments/aurora/jobs/job-active-learning-uq-aurora.sh
#
# Override:
#   qsub -v MATSIM_STRUCTURES="a.vasp b.vasp",MATSIM_TOP_W_THR=0.5 \
#        deployments/aurora/jobs/job-active-learning-uq-aurora.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── repo / paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
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
# QE: ships with the repo.
export MATSIM_QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-${REPO}/deployments/aurora/launchers/run-pw-gpu-aurora.sh}"
# VASP: no in-tree launcher yet; user must point this at their Aurora wrapper.
export MATSIM_VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-}"

TOP_W_THR="${MATSIM_TOP_W_THR:-0.6}"
ENT_THR="${MATSIM_ENT_THR:-0.5}"

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
echo "QE launch:   ${MATSIM_QE_LAUNCHER:-<unset>}"
echo "VASP launch: ${MATSIM_VASP_LAUNCHER:-<unset>}"
echo "Thresholds:  top_weight<${TOP_W_THR}  entropy>${ENT_THR}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

# ── run the active-learning driver ──────────────────────────────────────────
python "${REPO}/examples/active_learning_uq.py" \
    "${STRUCTURES[@]}" \
    --logdir          "${LOGDIR}" \
    --mlp-checkpoint  "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --mlp-device      cuda \
    --optimizer       FIRE \
    --maxiter         200 \
    --fmax            0.02 \
    --top-weight-threshold "${TOP_W_THR}" \
    --entropy-threshold    "${ENT_THR}" \
    2>&1 | tee "${RUN_DIR}/active-learning-uq.log"

echo "[$(date)] Active-learning loop complete. Artifacts in ${OUTPUT_DIR}"
