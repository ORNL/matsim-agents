#!/bin/bash
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
# This script activates the matsim-owned .venv-uma created with INSTALL_UMA=1.
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-uma-warmstart-perlmutter.sh
#
# Override fixture (comma-separated, see fixtures.yaml for available names):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch deployments/perlmutter/jobs/job-uma-warmstart-perlmutter.sh
#
# PREREQUISITE — install the UMA bundle on non-purgeable project storage first.
# Compute nodes run offline and load the checkpoint and references directly.
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
# Bundles live under $PROJ/models/artifacts/uma by default.
# See docs/model-download.md ("UMA MLIP weights on Perlmutter").
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

# Isolated FairChem/UMA compatibility environment.
VENV_ROOT=$REPO/.hpc-build/perlmutter
VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}"

QE_LAUNCHER=${MATSIM_QE_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh}
QE_PSEUDO_DIR=${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}

RUN_DIR=$RUNS_ROOT/uma-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/uma-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules ──────────────────────────────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Activate the UMA compatibility environment.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_uma_model_artifacts "${REPO}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# facebook/UMA is a gated Hugging Face repo — a token is required.
# Prefer an explicit HF_TOKEN env var; fall back to the cached login token.
_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "${_TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(< "${_TOKEN_FILE}")"
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is unset and no cached token found at ${_TOKEN_FILE}." >&2
  echo "         The model download will fail for gated repos (e.g. facebook/UMA)." >&2
  echo "         Set HF_TOKEN or run: huggingface-cli login" >&2
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
