#!/bin/bash
#SBATCH -J active-learning-uq
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# matsim-agents: active-learning loop on NERSC Perlmutter.
#
# Production AL contract: MD candidates, acquisition, one selected DFT
# labeller, immutable labelled dataset, optional explicit retraining, resume,
# and provenance. Scientific policy is constructed by the shared runner.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-active-learning-uq-perlmutter.sh
#
# Override:
#   MATSIM_STRUCTURES="a.vasp b.vasp" MATSIM_N_SELECT=4 \
#     sbatch deployments/perlmutter/jobs/job-active-learning-uq-perlmutter.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
REPO="${PROJECT_ROOT:?export PROJECT_ROOT}"
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

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

RUN_DIR=$RUNS_ROOT/active-learning-uq-${SLURM_JOB_ID:-$$}
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

# ── DFT launchers (the example skips cleanly if either is unset) ────────────
export MATSIM_DFT_BACKEND="${MATSIM_DFT_BACKEND:-qe}"
export MATSIM_PW_BIN="${MATSIM_PW_BIN:-$REPO/external/quantum-espresso/install-gpu/bin/pw.x}"
export MATSIM_PSEUDO_DIR="${MATSIM_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"
export MATSIM_DFT_WRAPPER="${MATSIM_DFT_WRAPPER:-$REPO/deployments/perlmutter/launchers/_qe-step-perlmutter.sh}"
export MATSIM_DFT_RANKS_PER_NODE="${MATSIM_DFT_RANKS_PER_NODE:-4}"
export MATSIM_DFT_THREADS_PER_RANK="${MATSIM_DFT_THREADS_PER_RANK:-16}"
export MATSIM_SEED_STRUCTURES="$(IFS=:; echo "${STRUCTURES[*]}")"

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter active-learning UQ loop]"
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
