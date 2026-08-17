#!/bin/bash
#SBATCH -A mat746
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
# For each input structure:
#   1. Relax with HydraGNN (multi-branch MLFF).
#   2. Score uncertainty from the per-step branch-weight CSV
#      (mean top-branch weight + mean normalized entropy).
#   3. If the prediction is flagged unreliable, trigger TWO reference
#      DFT calculations on the optimized structure:
#        - Quantum ESPRESSO pw.x (Frontier launcher: run-pw-gpu-frontier.sh)
#        - VASP vasp_std (skipped if MATSIM_VASP_LAUNCHER is unset)
#   4. Append flagged structures to training_candidates.csv.
#
# Submit:
#   sbatch deployments/frontier/jobs/job-active-learning-uq-frontier.sh
#
# Override:
#   MATSIM_STRUCTURES="a.vasp b.vasp" MATSIM_TOP_W_THR=0.5 \
#     sbatch deployments/frontier/jobs/job-active-learning-uq-frontier.sh
# Backward-compatible alias: MATSIM_AL_STRUCTURES
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
export MATSIM_QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-$REPO/deployments/frontier/launchers/run-pw-gpu-frontier.sh}"
export MATSIM_VASP_LAUNCHER="${MATSIM_VASP_LAUNCHER:-}"

TOP_W_THR=${MATSIM_TOP_W_THR:-0.6}
ENT_THR=${MATSIM_ENT_THR:-0.5}

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
