#!/bin/bash
#SBATCH -A mat746
#SBATCH -J single-relaxation
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: smoke-test of the planner -> executor -> analyst LangGraph
# on a single structure using the HydraGNN MLFF backend (Frontier).
#
# Mirrors scripts/advanced/{aurora,perlmutter}/job-single-relaxation-*.sh.
#
# Submit:
#   sbatch scripts/advanced/frontier/job-single-relaxation-frontier.sh
#
# Override at submit time:
#   MATSIM_STRUCTURE=/path/to/POSCAR \
#     sbatch scripts/advanced/frontier/job-single-relaxation-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/lustre/orion/mat746/proj-shared/matsim-agents
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
STRUCTURE=${MATSIM_STRUCTURE:-$REPO/tests/integration/data/Si.vasp}

RUN_DIR=$PROJ/runs/single-relaxation-$SLURM_JOB_ID
OUTPUT_DIR=$RUN_DIR/outputs
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/scripts/setup/frontier/frontier-module-stack.sh"
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

# ── run the agent graph on a single structure ───────────────────────────────
matsim-agents run \
    "Relax the structure at ${STRUCTURE} using HydraGNN and report the final energy." \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$HYDRAGNN_BRANCH_MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --mlp-device      cuda \
    --max-iterations  3 \
    2>&1 | tee "$RUN_DIR/single-relaxation.log"

echo "[$(date)] Single-relaxation finished. Artifacts in $OUTPUT_DIR"
