#!/bin/bash
#SBATCH -J mace-warmstart
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: MACE-MP (mace-torch) warm-start vs Quantum ESPRESSO cold-start
# benchmark on NERSC Perlmutter.
#
# Runs tests/integration/test_mace_warmstart.py, which:
#   1. Relaxes each fixture with a MACE-MP foundation model (warm start).
#   2. Runs pw.x cold-start and pw.x warm-start (initial coords from MACE).
#   3. Reports SCF iterations / wall-time speed-up.
#
# This script activates .venv-mace because upstream MACE 0.3.16 declares
# e3nn==0.4.4 while the primary HydraGNN environment declares e3nn==0.5.1.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-mace-warmstart-perlmutter.sh
#
# Override fixture (comma-separated, see fixtures.yaml for available names):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch deployments/perlmutter/jobs/job-mace-warmstart-perlmutter.sh
#
# PREREQUISITE — the MACE foundation weights must already be present in the
# shared MACE cache ($PROJ/models/mace_cache/mace). Compute nodes have no
# internet; mace_mp() reads the cached checkpoint and will NOT download it on
# first use. The cache is populated by any prior MACE run (e.g. the fine-tune
# study). Override the model with MATSIM_MACE_MODEL (small|medium|large).
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV="${MATSIM_MACE_VENV:-${REPO}/.venv-mace}"

QE_LAUNCHER=${MATSIM_QE_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh}
QE_PSEUDO_DIR=${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}

RUN_DIR=$RUNS_ROOT/mace-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/mace-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules ──────────────────────────────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Activate the matsim-owned MACE compatibility environment.
[[ ! -d "${VENV}" ]] && { echo "ERROR: MACE environment not found: ${VENV}" >&2; exit 2; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── MACE foundation-model cache (offline) ────────────────────────────────────
# mace_mp(model=...) resolves the checkpoint under $XDG_CACHE_HOME/mace. The
# shared project cache is prepopulated so compute nodes never hit the network.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJ}/models/mace_cache}"
export MACE_CACHE="${MACE_CACHE:-${XDG_CACHE_HOME}/mace}"
mkdir -p "${MACE_CACHE}"

# ── MACE / warmstart env ─────────────────────────────────────────────────────
export MATSIM_MACE_FAMILY="${MATSIM_MACE_FAMILY:-mace_mp}"
export MATSIM_MACE_MODEL="${MATSIM_MACE_MODEL:-medium}"
export MATSIM_MACE_PRECISION="${MATSIM_MACE_PRECISION:-fp64}"
export MATSIM_QE_LAUNCHER="$QE_LAUNCHER"
export MATSIM_QE_PSEUDO_DIR="$QE_PSEUDO_DIR"
export MATSIM_QE_MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"
export MATSIM_QE_TIMEOUT_SEC="${MATSIM_QE_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-MoNbTaW_HEA}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter MACE warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "MACE family:  $MATSIM_MACE_FAMILY"
echo "MACE model:   $MATSIM_MACE_MODEL"
echo "MACE prec:    $MATSIM_MACE_PRECISION"
echo "QE launcher:  $QE_LAUNCHER"
echo "QE pseudos:   $QE_PSEUDO_DIR"
echo "Fixtures:     $MATSIM_WARMSTART_FIXTURES"
echo "MACE cache:   $MACE_CACHE"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "$REPO" >/dev/null
# NOTE: no -x. Each fixture is an independent benchmark; a warm>cold
# assertion failure on one fixture (a legitimate "warm-start did not help"
# result for this model) must not prevent the remaining fixtures from
# running and writing their comparison.json for the harvester.
python -m pytest -vs tests/integration/test_mace_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/mace-warmstart.log"
popd >/dev/null

echo "[$(date)] MACE warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
