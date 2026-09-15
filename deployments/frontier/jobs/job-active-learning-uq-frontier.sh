#!/bin/bash
#SBATCH -J active-learning-uq
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: active-learning loop on Frontier.
#
# Production AL contract using the shared scheduler-neutral runner.
#
# Submit:
#   sbatch deployments/frontier/jobs/job-active-learning-uq-frontier.sh
#
# Override:
#   MATSIM_STRUCTURES="a.vasp b.vasp" MATSIM_N_SELECT=4 \
#     sbatch deployments/frontier/jobs/job-active-learning-uq-frontier.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
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

DEFAULT_STRUCTURES=(
  "$REPO/tests/integration/data/Si.vasp"
  "$REPO/tests/integration/data/MgO.vasp"
  "$REPO/tests/integration/data/NaCl.vasp"
)
STRUCTURE_LIST="${MATSIM_STRUCTURES:-${MATSIM_AL_STRUCTURES:-}}"
if [[ -n "${STRUCTURE_LIST}" ]]; then
  # shellcheck disable=SC2206
  STRUCTURES=( ${STRUCTURE_LIST} )
else
  STRUCTURES=( "${DEFAULT_STRUCTURES[@]}" )
fi

RUN_DIR=$PROJ/runs/active-learning-uq-$SLURM_JOB_ID
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

# ── DFT launchers (the example skips cleanly if either is unset) ────────────
export MATSIM_DFT_BACKEND="${MATSIM_DFT_BACKEND:-qe}"
export MATSIM_PW_BIN="${MATSIM_PW_BIN:-$REPO/external/quantum-espresso/install-gpu/bin/pw.x}"
export MATSIM_PSEUDO_DIR="${MATSIM_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"
export MATSIM_DFT_WRAPPER="${MATSIM_DFT_WRAPPER:-$REPO/deployments/frontier/launchers/_qe-step-frontier.sh}"
export MATSIM_DFT_RANKS_PER_NODE="${MATSIM_DFT_RANKS_PER_NODE:-8}"
export MATSIM_DFT_THREADS_PER_RANK="${MATSIM_DFT_THREADS_PER_RANK:-7}"
export MATSIM_SEED_STRUCTURES="$(IFS=:; echo "${STRUCTURES[*]}")"

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Frontier active-learning UQ loop]"
echo "Date:        $(date)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Host:        $(hostname)"
echo "Repo:        $REPO"
echo "Venv:        $VENV"
echo "Logdir:      $LOGDIR"
echo "MLP ckpt:    $HYDRAGNN_BRANCH_MLP_CHECKPOINT"
echo "Structures:  ${#STRUCTURES[@]}"
for s in "${STRUCTURES[@]}"; do echo "             - $s"; done
echo "DFT backend: ${MATSIM_DFT_BACKEND}"
echo "DFT wrapper: ${MATSIM_DFT_WRAPPER}"
echo "Run dir:     $RUN_DIR"
echo "=========================================="

# ── run the active-learning driver ──────────────────────────────────────────
source "$REPO/deployments/common/run-active-learning.sh"

echo "[$(date)] Active-learning loop complete. Artifacts in $OUTPUT_DIR"
