#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J single-relaxation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/single-relaxation-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/single-relaxation-%j/job-%j.out
# ---------------------------------------------------------------------------
# matsim-agents: smoke-test of the planner -> executor -> analyst LangGraph
# on a single structure using the HydraGNN MLFF backend (NERSC Perlmutter).
#
# Mirrors scripts/advanced/{aurora,frontier}/job-single-relaxation-*.sh.
#
# Submit:
#   sbatch scripts/advanced/perlmutter/job-single-relaxation-perlmutter.sh
#
# Override at submit time:
#   MATSIM_STRUCTURE=/path/to/POSCAR \
#     sbatch scripts/advanced/perlmutter/job-single-relaxation-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
MLP_CHECKPOINT=${MATSIM_HYDRAGNN_MLP_CKPT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
STRUCTURE=${MATSIM_STRUCTURE:-$REPO/tests/integration/data/Si.vasp}

RUN_DIR=$PROJ/runs/single-relaxation-${SLURM_JOB_ID:-$$}
OUTPUT_DIR=$RUN_DIR/outputs
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & venv (HydraGNN-aligned stack) ──────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter single-relaxation]"
echo "Date:        $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Host:        $(hostname)"
echo "Repo:        $REPO"
echo "Venv:        $VENV"
echo "Structure:   $STRUCTURE"
echo "Logdir:      $LOGDIR"
echo "MLP ckpt:    $MLP_CHECKPOINT"
echo "Run dir:     $RUN_DIR"
echo "=========================================="

# ── run the agent graph on a single structure ───────────────────────────────
matsim-agents run \
    "Relax the structure at ${STRUCTURE} using HydraGNN and report the final energy." \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --mlp-device      cuda \
    --max-iterations  3 \
    2>&1 | tee "$RUN_DIR/single-relaxation.log"

echo "[$(date)] Single-relaxation finished. Artifacts in $OUTPUT_DIR"
