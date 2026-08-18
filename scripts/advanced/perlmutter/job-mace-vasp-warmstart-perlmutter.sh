#!/bin/bash
#SBATCH -A m5216
#SBATCH -J mace-vasp-warmstart
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: MACE-MP (mace-torch) warm-start vs VASP cold-start benchmark
# on NERSC Perlmutter.
#
# Runs tests/integration/test_mace_vasp_warmstart.py, which:
#   1. Relaxes each fixture with a MACE-MP foundation model (warm start).
#   2. Runs vasp_std cold-start and vasp_std warm-start (coords from MACE).
#   3. Reports ionic steps / SCF iterations / wall-time speed-up.
#
# This script activates the separate mace_venv (not hydragnn_venv/fairchem_venv)
# because mace-torch pins e3nn==0.4.4 (the version the foundation checkpoints
# were serialised with).  See scripts/setup/perlmutter/build-mace-venv-perlmutter.sh.
#
# Submit:
#   sbatch scripts/advanced/perlmutter/job-mace-vasp-warmstart-perlmutter.sh
#
# Override fixture (comma-separated, see fixtures.yaml for available names):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch scripts/advanced/perlmutter/job-mace-vasp-warmstart-perlmutter.sh
#
# PREREQUISITE — the MACE foundation weights must already be present in the
# shared MACE cache ($PROJ/models/mace_cache/mace). Compute nodes have no
# internet; mace_mp() reads the cached checkpoint and will NOT download it on
# first use. Override the model with MATSIM_MACE_MODEL (small|medium|large).
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
RUNS_ROOT="${RUNS_ROOT:-${PROJ}/runs}"

VENV_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV="${MATSIM_MACE_VENV:-${VENV_ROOT}/mace_venv}"

VASP_LAUNCHER=${MATSIM_VASP_LAUNCHER:-$REPO/scripts/launchers/perlmutter/run-vasp-gpu-perlmutter.sh}
VASP_POTCAR_DIR=${MATSIM_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}

RUN_DIR=$RUNS_ROOT/mace-vasp-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/mace-vasp-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules ──────────────────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Activate mace_venv (plain venv, not conda).
[[ ! -d "${VENV}" ]] && { echo "ERROR: mace_venv not found: ${VENV}" >&2; exit 2; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── MACE foundation-model cache (offline) ────────────────────────────────────
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJ}/models/mace_cache}"
export MACE_CACHE="${MACE_CACHE:-${XDG_CACHE_HOME}/mace}"
mkdir -p "${MACE_CACHE}"

# ── MACE+VASP / warmstart env ────────────────────────────────────────────────
export MATSIM_MACE_FAMILY="${MATSIM_MACE_FAMILY:-mace_mp}"
export MATSIM_MACE_MODEL="${MATSIM_MACE_MODEL:-medium}"
export MATSIM_MACE_PRECISION="${MATSIM_MACE_PRECISION:-fp64}"
export MATSIM_VASP_LAUNCHER="$VASP_LAUNCHER"
export MATSIM_VASP_POTCAR_DIR="$VASP_POTCAR_DIR"
export MATSIM_VASP_MLP_DEVICE="${MATSIM_VASP_MLP_DEVICE:-cuda}"
export MATSIM_VASP_TIMEOUT_SEC="${MATSIM_VASP_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-MoNbTaW_HEA}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter MACE+VASP warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "MACE family:  $MATSIM_MACE_FAMILY"
echo "MACE model:   $MATSIM_MACE_MODEL"
echo "MACE prec:    $MATSIM_MACE_PRECISION"
echo "VASP launcher:$VASP_LAUNCHER"
echo "POTCAR dir:   $VASP_POTCAR_DIR"
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
python -m pytest -vs tests/integration/test_mace_vasp_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/mace-vasp-warmstart.log"
popd >/dev/null

echo "[$(date)] MACE+VASP warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
