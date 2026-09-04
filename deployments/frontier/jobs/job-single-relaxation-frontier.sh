#!/bin/bash
#SBATCH -J single-relaxation
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: typed, provenance-tracked MLIP relaxation on Frontier.
#
# Mirrors scripts/advanced/{aurora,perlmutter}/job-single-relaxation-*.sh.
#
# Submit:
#   sbatch deployments/frontier/jobs/job-single-relaxation-frontier.sh
#
# Override at submit time:
#   MATSIM_STRUCTURE=/path/to/POSCAR \
#     sbatch deployments/frontier/jobs/job-single-relaxation-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
STRUCTURE=${MATSIM_STRUCTURE:-$REPO/tests/integration/data/Si.vasp}

RUN_DIR=$PROJ/runs/single-relaxation-$SLURM_JOB_ID
OUTPUT_DIR=$RUN_DIR/outputs
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
TORCH_LIB=$VENV/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Frontier single-relaxation]"
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
