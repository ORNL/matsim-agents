#!/bin/bash
#SBATCH -A m5216
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
# For each input structure:
#   1. Relax with HydraGNN (multi-branch MLFF).
#   2. Score uncertainty from the per-step branch-weight CSV
#      (mean top-branch weight + mean normalized entropy).
#   3. If the prediction is flagged unreliable, trigger TWO reference
#      DFT calculations on the optimized structure:
#        - Quantum ESPRESSO pw.x (Perlmutter launcher: run-pw-gpu-perlmutter.sh)
#        - VASP vasp_std (skipped if MATSIM_VASP_LAUNCHER is unset)
#   4. Append flagged structures to training_candidates.csv.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-active-learning-uq-perlmutter.sh
#
# Override:
#   MATSIM_STRUCTURES="a.vasp b.vasp" MATSIM_TOP_W_THR=0.5 \
#     sbatch deployments/perlmutter/jobs/job-active-learning-uq-perlmutter.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
REPO="${PROJECT_ROOT:-/global/cfs/projectdirs/m5216/mlupopa/matsim-agents}"
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
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
export MATSIM_QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh}"
export MATSIM_VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-}"
export MATSIM_QE_PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"

TOP_W_THR=${MATSIM_TOP_W_THR:-0.6}
ENT_THR=${MATSIM_ENT_THR:-0.5}

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
echo "QE launch:   ${MATSIM_QE_LAUNCHER:-<unset>}"
echo "VASP launch: ${MATSIM_VASP_LAUNCHER:-<unset>}"
echo "Thresholds:  top_weight<${TOP_W_THR}  entropy>${ENT_THR}"
echo "Run dir:     $RUN_DIR"
echo "=========================================="

# ── run the active-learning driver ──────────────────────────────────────────
python "$REPO/examples/active_learning_uq.py" \
    "${STRUCTURES[@]}" \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$HYDRAGNN_BRANCH_MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --mlp-device      cuda \
    --optimizer       FIRE \
    --maxiter         200 \
    --fmax            0.02 \
    --top-weight-threshold "$TOP_W_THR" \
    --entropy-threshold    "$ENT_THR" \
    2>&1 | tee "$RUN_DIR/active-learning-uq.log"

echo "[$(date)] Active-learning loop complete. Artifacts in $OUTPUT_DIR"
