#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J uma-warmstart
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: UMA (fairchem-core) warm-start vs Quantum ESPRESSO cold-start
# benchmark on NERSC Perlmutter.
#
# Runs tests/integration/test_uma_warmstart.py, which:
#   1. Relaxes each fixture with UMA MLFF (warm start).
#   2. Runs pw.x cold-start and pw.x warm-start (initial coords from UMA).
#   3. Reports SCF iterations / wall-time speed-up.
#
# This script activates the separate fairchem_venv (not hydragnn_venv) because
# fairchem-core requires numpy>=2.0 which conflicts with HydraGNN's pin.
# See scripts/setup/perlmutter/README.md for details.
#
# Submit:
#   sbatch scripts/advanced/perlmutter/job-uma-warmstart-perlmutter.sh
#
# Override fixture (comma-separated, see fixtures.yaml for available names):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch scripts/advanced/perlmutter/job-uma-warmstart-perlmutter.sh
#
# The UMA model (uma-s-1p1) is downloaded automatically from Hugging Face on
# first use.  Set HF_HOME to a shared writable directory so the download is
# cached across jobs:
#   export HF_HOME=/global/cfs/projectdirs/m5216/mlupopa/models/hf_cache
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

# fairchem_venv lives alongside hydragnn_venv under the HydraGNN install root.
VENV_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

QE_LAUNCHER=${MATSIM_QE_LAUNCHER:-$REPO/scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh}
QE_PSEUDO_DIR=${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}

RUN_DIR=$RUNS_ROOT/uma-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/uma-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules ──────────────────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Activate the plain Python venv (fairchem_venv), NOT conda.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Cache UMA model downloads in a shared project directory so multiple jobs
# share one copy.  The directory is created on first use.
export HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache}"
mkdir -p "${HF_HOME}"

# Read the pre-fetched UMA cache in OFFLINE mode: compute nodes mount CFS via
# DVS, which does not support fcntl.flock (OSError [Errno 524]); offline reads
# skip the lock. Requires a prior run of
# scripts/download/perlmutter/download-uma-perlmutter.sh.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# facebook/UMA is a gated Hugging Face repo — a token is required.
# Prefer an explicit HF_TOKEN env var; fall back to the cached login token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  _TOKEN_FILE="${HOME}/.cache/huggingface/token"
  if [[ -f "${_TOKEN_FILE}" ]]; then
    export HF_TOKEN="$(< "${_TOKEN_FILE}")"
  else
    echo "WARNING: HF_TOKEN is unset and no cached token found at ${_TOKEN_FILE}." >&2
    echo "         The model download will fail for gated repos (e.g. facebook/UMA)." >&2
    echo "         Set HF_TOKEN or run: huggingface-cli login" >&2
  fi
fi

# ── UMA / warmstart env ──────────────────────────────────────────────────────
export MATSIM_UMA_MODEL_NAME="${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}"
export MATSIM_UMA_TASK="${MATSIM_UMA_TASK:-omat}"
export MATSIM_QE_LAUNCHER="$QE_LAUNCHER"
export MATSIM_QE_PSEUDO_DIR="$QE_PSEUDO_DIR"
export MATSIM_QE_MLP_DEVICE="${MATSIM_QE_MLP_DEVICE:-cuda}"
export MATSIM_QE_TIMEOUT_SEC="${MATSIM_QE_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-MoNbTaW_HEA}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter UMA warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "UMA model:    $MATSIM_UMA_MODEL_NAME"
echo "UMA task:     $MATSIM_UMA_TASK"
echo "QE launcher:  $QE_LAUNCHER"
echo "QE pseudos:   $QE_PSEUDO_DIR"
echo "Fixtures:     $MATSIM_WARMSTART_FIXTURES"
echo "HF_HOME:      $HF_HOME"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "$REPO" >/dev/null
python -m pytest -xvs tests/integration/test_uma_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/uma-warmstart.log"
popd >/dev/null

echo "[$(date)] UMA warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
