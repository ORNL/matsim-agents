#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J qe-warmstart
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/qe-warmstart-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/qe-warmstart-%j/job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: HydraGNN warm-start vs Quantum ESPRESSO cold-start
# benchmark on NERSC Perlmutter.
#
# Runs tests/integration/test_qe_warmstart.py, which:
#   1. Relaxes each fixture with HydraGNN MLFF (warm start).
#   2. Runs pw.x cold-start and pw.x warm-start (initial coords from MLFF).
#   3. Reports SCF iterations / wall-time speed-up.
#
# Submit:
#   sbatch scripts/advanced/perlmutter/job-qe-warmstart-perlmutter.sh
#
# Override fixtures (comma-separated, see test file for available names):
#   MATSIM_WARMSTART_FIXTURES=Si_diamond \
#     sbatch scripts/advanced/perlmutter/job-qe-warmstart-perlmutter.sh
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

QE_LAUNCHER=${MATSIM_QE_LAUNCHER:-$REPO/scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh}
QE_PSEUDO_DIR=${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}

RUN_DIR=$PROJ/runs/qe-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/qe-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── QE / warmstart env ──────────────────────────────────────────────────────
export MATSIM_QE_LAUNCHER="$QE_LAUNCHER"
export MATSIM_QE_PSEUDO_DIR="$QE_PSEUDO_DIR"
export MATSIM_HYDRAGNN_LOGDIR="$LOGDIR"
export MATSIM_HYDRAGNN_MLP_CKPT="$MLP_CHECKPOINT"
export MATSIM_QE_MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"
export MATSIM_QE_TIMEOUT_SEC="${MATSIM_QE_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-Si_diamond}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter QE warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "HydraGNN log: $LOGDIR"
echo "MLP ckpt:     $MLP_CHECKPOINT"
echo "QE launcher:  $QE_LAUNCHER"
echo "QE pseudos:   $QE_PSEUDO_DIR"
echo "Fixtures:     $MATSIM_WARMSTART_FIXTURES"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "$REPO" >/dev/null
python -m pytest -xvs tests/integration/test_qe_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/qe-warmstart.log"
popd >/dev/null

echo "[$(date)] QE warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
