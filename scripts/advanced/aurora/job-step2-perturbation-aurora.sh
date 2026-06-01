#!/bin/bash
#PBS -A CM2US
#PBS -N step2-perturbation
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Step 2 of the BEST6 MLFF diagnostic:
#   Perturb two already-relaxed structures (BaTiO3 GS, MoNbTaW GS) and run
#   a tight relaxation. Tests whether FIRE-step-1 convergence reflects a
#   genuine local minimum or a flat numerical-noise plateau.
#
# Submit:
#   qsub scripts/advanced/aurora/job-step2-perturbation-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
HYDRAGNN_EXAMPLE="${PROJ}/HydraGNN/examples/multidataset_hpo_sc26"
LOGDIR="${MATSIM_HYDRAGNN_LOGDIR:-${HYDRAGNN_EXAMPLE}/multidataset_hpo-BEST6-fp64}"
MLP_CHECKPOINT="${MATSIM_HYDRAGNN_MLP_CKPT:-${HYDRAGNN_EXAMPLE}/mlp_branch_weights.pt}"

# Inputs identified from the perovskite + RHEA discovery runs (Step 1 analysis).
BTO_PROV="${PROJ}/runs/discovery-chat-perovskites-aurora-8510287.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov/outputs/discovery/BaO3Ti/relaxed"
BTO_STRUCT="${MATSIM_BTO_STRUCT:-${BTO_PROV}/BaO3Ti__ABC3_hR10_167_a_b_e_o0_optimized_structure.vasp}"
RHEA_PROV="${PROJ}/runs/discovery-chat-rhea-aurora-8510299.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov/outputs/discovery/MoNbTaW/relaxed"
RHEA_STRUCT="${MATSIM_RHEA_STRUCT:-${RHEA_PROV}/MoNbTaW__ABCD_oP16_57_d_c_d_d_o10_optimized_structure.vasp}"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/step2-perturbation-aurora-${JOBID}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"

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
export MATSIM_HYDRAGNN_LOGDIR="${LOGDIR}"
export MATSIM_HYDRAGNN_MLP_CKPT="${MLP_CHECKPOINT}"

echo "=========================================="
echo "[Aurora] step 2 perturbation diagnostic"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "Job ID:      ${JOBID}"
echo "Logdir:      ${LOGDIR}"
echo "BaTiO3:      ${BTO_STRUCT}"
echo "MoNbTaW:     ${RHEA_STRUCT}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

python "${REPO}/scripts/diagnostics/step2_perturbation_diagnostic.py" \
    --logdir          "${LOGDIR}" \
    --mlp-checkpoint  "${MLP_CHECKPOINT}" \
    --output-dir      "${OUTPUT_DIR}" \
    --device          cuda \
    --seed            42 \
    --displacement-scale 0.10 \
    --cases \
        "BaTiO3:${BTO_STRUCT}" \
        "MoNbTaW:${RHEA_STRUCT}" \
    2>&1 | tee "${RUN_DIR}/step2.log"

echo "[$(date)] Step 2 diagnostic complete. Summary: ${OUTPUT_DIR}/step2_summary.json"
