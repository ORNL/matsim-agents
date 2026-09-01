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
# matsim-agents: smoke-test of `examples/single_relaxation.py` on ALCF Aurora.
#
# Runs the planner -> executor -> analyst LangGraph on a single structure
# using the HydraGNN MLFF backend. Mirrors the example but lets the
# structure path, logdir and MLP checkpoint be overridden via env vars.
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

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"
STRUCTURE="${MATSIM_STRUCTURE:-${REPO}/tests/integration/data/Si.vasp}"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/single-relaxation-aurora-${JOBID}"
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

# executor_node reads these env vars as fallback when config injection is
# unavailable (e.g. across LangGraph checkpoint boundaries).
export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"
export HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"

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
echo "MLP ckpt:    ${HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

# ── run the agent graph on a single structure ───────────────────────────────
# We call the CLI rather than the example file directly so the structure path
# can be overridden at submit time.
matsim-agents run \
    "Relax the structure at ${STRUCTURE} using HydraGNN and report the final energy." \
    --logdir          "${LOGDIR}" \
    --hydragnn-branch-mlp-checkpoint "${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --mlp-device      cuda \
    --max-iterations  3 \
    2>&1 | tee "${RUN_DIR}/single-relaxation.log"

echo "[$(date)] Single-relaxation smoke test complete. Artifacts in ${OUTPUT_DIR}"
