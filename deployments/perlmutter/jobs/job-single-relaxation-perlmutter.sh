#!/bin/bash
#SBATCH -J single-relaxation
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# matsim-agents: typed, provenance-tracked MLIP relaxation on Perlmutter.
#
# Mirrors scripts/advanced/{aurora,frontier}/job-single-relaxation-*.sh.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-single-relaxation-perlmutter.sh
#
# Override at submit time:
#   MATSIM_STRUCTURE=/path/to/POSCAR \
#     sbatch deployments/perlmutter/jobs/job-single-relaxation-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "ERROR: set PROJECT_ROOT to the matsim-agents checkout" >&2
  exit 2
}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV=$REPO/.venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
STRUCTURE=${MATSIM_STRUCTURE:-$REPO/tests/integration/data/Si.vasp}

RUN_DIR=$RUNS_ROOT/single-relaxation-${SLURM_JOB_ID:-$$}
OUTPUT_DIR=$RUN_DIR/outputs
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & venv (HydraGNN-aligned stack) ──────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
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
echo "MLP ckpt:    $HYDRAGNN_BRANCH_MLP_CHECKPOINT"
echo "Run dir:     $RUN_DIR"
echo "=========================================="

# ── run the typed relaxation contract ───────────────────────────────────────
source "$REPO/deployments/common/run-mlip-relaxation.sh"

echo "[$(date)] Single-relaxation finished. Artifacts in $OUTPUT_DIR"
