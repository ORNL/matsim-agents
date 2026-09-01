#!/bin/bash
#SBATCH -J uma-vasp-warmstart
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: UMA (fairchem-core) warm-start vs VASP cold-start benchmark
# on NERSC Perlmutter.
#
# Mirror of job-uma-warmstart-perlmutter.sh for the VASP backend.
# Runs tests/integration/test_uma_vasp_warmstart.py, which:
#   1. Relaxes each fixture with UMA MLFF (warm start).
#   2. Runs vasp_std cold-start and vasp_std warm-start.
#   3. Reports ionic steps / wall-time speed-up.
#
# Uses the fairchem_venv (NOT hydragnn_venv) — fairchem-core requires numpy>=2.
#
# Artifacts:
#   $RUNS_ROOT/uma-vasp-warmstart-$SLURM_JOB_ID/
#       uma-vasp-warmstart/      ← pytest --basetemp
#       uma-vasp-warmstart.log   ← combined stdout/stderr
#
# Submit:
#   sbatch deployments/perlmutter/jobs/job-uma-vasp-warmstart-perlmutter.sh
#
# Override fixture (comma-separated):
#   MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA \
#     sbatch deployments/perlmutter/jobs/job-uma-vasp-warmstart-perlmutter.sh
#
# PREREQUISITE — prefetch the UMA weights first. This job reads the shared HF
# cache in OFFLINE mode (HF_HUB_OFFLINE=1): compute nodes have no internet and
# CFS does not support fcntl.flock over DVS (OSError [Errno 524]), so UMA is NOT
# downloaded on first use. Run once before submitting:
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
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

VENV_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"

VASP_LAUNCHER=${MATSIM_VASP_LAUNCHER:-$REPO/deployments/perlmutter/launchers/run-vasp-gpu-perlmutter.sh}
VASP_POTCAR_DIR=${MATSIM_VASP_POTCAR_DIR:-$REPO/external/vasp6/potcar/potpaw_PBE.64}

RUN_DIR=$RUNS_ROOT/uma-vasp-warmstart-$SLURM_JOB_ID
WARMSTART_DIR=$RUN_DIR/uma-vasp-warmstart
mkdir -p "$RUN_DIR" "$WARMSTART_DIR"

# ── modules ──────────────────────────────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Activate fairchem_venv (plain venv, not conda).
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Cache UMA model in shared project directory.
export HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache}"
mkdir -p "${HF_HOME}"

# fairchem's pretrained_mlip.get_predict_unit() ignores HF_HOME entirely -- it
# always calls hf_hub_download(..., cache_dir=FAIRCHEM_CACHE_DIR), which
# defaults to ~/.cache/fairchem on $HOME (CFS/DVS, no fcntl.flock support ->
# OSError [Errno 524], regardless of offline mode). Point it at $SCRATCH
# (flock-capable, persistent across jobs). Requires a prior run of
# deployments/perlmutter/download/download-uma-perlmutter.sh.
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
mkdir -p "${FAIRCHEM_CACHE_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# facebook/UMA is gated — read token from cached login at runtime.
_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "${_TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(< "${_TOKEN_FILE}")"
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is unset and no cached token found at ${_TOKEN_FILE}." >&2
  echo "         The model download will fail for gated repos (e.g. facebook/UMA)." >&2
  echo "         Set HF_TOKEN or run: huggingface-cli login" >&2
fi

# ── UMA+VASP / warmstart env ─────────────────────────────────────────────────
export MATSIM_UMA_MODEL_NAME="${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}"
export MATSIM_UMA_TASK="${MATSIM_UMA_TASK:-omat}"
export MATSIM_VASP_LAUNCHER="$VASP_LAUNCHER"
export MATSIM_VASP_POTCAR_DIR="$VASP_POTCAR_DIR"
export MATSIM_VASP_MLP_DEVICE="${MATSIM_VASP_MLP_DEVICE:-cuda}"
export MATSIM_VASP_TIMEOUT_SEC="${MATSIM_VASP_TIMEOUT_SEC:-3600}"
export MATSIM_WARMSTART_FIXTURES="${MATSIM_WARMSTART_FIXTURES:-MoNbTaW_HEA}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter UMA+VASP warm-start benchmark]"
echo "Date:         $(date)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Repo:         $REPO"
echo "Venv:         $VENV"
echo "UMA model:    $MATSIM_UMA_MODEL_NAME"
echo "UMA task:     $MATSIM_UMA_TASK"
echo "VASP launcher:$VASP_LAUNCHER"
echo "POTCAR dir:   $VASP_POTCAR_DIR"
echo "Fixtures:     $MATSIM_WARMSTART_FIXTURES"
echo "HF_HOME:      $HF_HOME"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"

# ── benchmark ───────────────────────────────────────────────────────────────
pushd "$REPO" >/dev/null
python -m pytest -xvs tests/integration/test_uma_vasp_warmstart.py \
  --basetemp="$WARMSTART_DIR" \
  2>&1 | tee "$RUN_DIR/uma-vasp-warmstart.log"
popd >/dev/null

echo "[$(date)] UMA+VASP warm-start benchmark complete. Artifacts in $WARMSTART_DIR"
