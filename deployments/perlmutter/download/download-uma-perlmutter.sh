#!/bin/bash
#SBATCH -J dl-uma
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
# ---------------------------------------------------------------------------
# Pre-fetch the UMA (fairchem-core) universal MLIP checkpoint(s) into the shared
# Hugging Face cache, so downstream AL / warm-start jobs do not spend GPU
# wall-time downloading on first use.
#
# UMA weights are NOT fetched by the `download-models-*.sh` scripts (those cover
# the LLM chat models). fairchem normally downloads UMA lazily on the first call
# to `pretrained_mlip.get_predict_unit(...)`; this script does that once, on the
# CPU partition, and verifies the model loads.
#
# Submit (default model uma-s-1p1):
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Multiple / alternate models:
#   UMA_MODELS="uma-s-1p1 uma-m-1p1" \
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Alternate cache location:
#   HF_HOME=/path/to/project/models/hf_cache \
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Notes:
# - facebook/UMA is a GATED repo: you must accept the license on Hugging Face
#   and provide a token. This script reads ~/.cache/huggingface/token if HF_TOKEN
#   is not already set (run `hf auth login` once beforehand).
# - Downloads are resumable; rerunning skips files already in the cache.
# - Runs from the matsim-owned .venv built with INSTALL_UMA=1.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# Unified HydraGNN/FairChem environment.
VENV_ROOT="$REPO/.hpc-build/perlmutter"
VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv}"
RUN_DIR="$PROJ/runs/download-uma-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: matsim .venv not found: $VENV" >&2
  echo "Build it first with:" >&2
  echo "  INSTALL_UMA=1 bash deployments/perlmutter/setup/install.sh" >&2
  exit 1
fi

# ── modules and matsim-owned environment ─────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Persistent (shared) destination cache on CFS.
DEST_HF="${HF_HOME:-${PROJ}/models/hf_cache}"
mkdir -p "${DEST_HF}"

# CFS/GPFS does NOT support fcntl.flock (OSError [Errno 524]), which
# huggingface_hub requires while downloading. Stage the download on a
# flock-capable filesystem ($SCRATCH Lustre, else node-local /tmp), then copy
# the result into the persistent CFS cache.
STAGE_BASE="${SCRATCH:-/tmp}"
STAGE_HF="${STAGE_BASE}/hf-stage.${USER}.${SLURM_JOB_ID:-manual}"
mkdir -p "${STAGE_HF}"
trap 'rm -rf "${STAGE_HF}" 2>/dev/null || true' EXIT
# Seed the stage with whatever is already cached so downloads stay incremental.
rsync -a "${DEST_HF}/" "${STAGE_HF}/" 2>/dev/null || true
export HF_HOME="${STAGE_HF}"
mkdir -p "${HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set and ~/.cache/huggingface/token not found." >&2
  echo "         The download will fail for the gated facebook/UMA repo." >&2
  echo "         Run 'hf auth login' once, or export HF_TOKEN=hf_..." >&2
fi

echo "[$(date)] Staging cache (flock-capable): $HF_HOME"
echo "[$(date)] Persistent destination (CFS):  $DEST_HF"

# fairchem's pretrained_mlip.get_predict_unit()/pretrained_checkpoint_path_from_name()
# call hf_hub_download(..., cache_dir=CACHE_DIR) with a HARDCODED cache_dir that
# ignores HF_HOME entirely. CACHE_DIR comes from FAIRCHEM_CACHE_DIR (env var),
# defaulting to ~/.cache/fairchem on $HOME (CFS -> no flock support, same
# OSError [Errno 524]). Point it at the flock-capable $SCRATCH directly; unlike
# a job-scoped stage, $SCRATCH persists across jobs, so no CFS copy-back needed.
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:-/tmp}/matsim-agents/fairchem_cache}"
mkdir -p "${FAIRCHEM_CACHE_DIR}"
echo "[$(date)] FAIRCHEM_CACHE_DIR (flock-capable, persistent): $FAIRCHEM_CACHE_DIR"

# ── model list ───────────────────────────────────────────────────────────────
UMA_MODELS="${UMA_MODELS:-uma-s-1p1}"

echo "[$(date)] Cache destination (HF_HOME): $HF_HOME"
echo "[$(date)] UMA models to pre-fetch: $UMA_MODELS"
echo "[$(date)] Using venv: $VENV"

rc=0
for model_name in $UMA_MODELS; do
  log="$RUN_DIR/${model_name}.download.log"
  echo
  echo "[$(date)] Fetching + validating UMA model: $model_name"
  if python - "$model_name" >"$log" 2>&1 <<'PY'
import sys

model_name = sys.argv[1]
from fairchem.core import pretrained_mlip

print("available_models:", getattr(pretrained_mlip, "available_models", "?"))
# device="cpu" avoids needing a GPU just to download + load the checkpoint.
predictor = pretrained_mlip.get_predict_unit(model_name, device="cpu")
print(f"OK: {model_name} downloaded and loaded on CPU -> {type(predictor).__name__}")
PY
  then
    echo "[$(date)] DONE: $model_name"
  else
    echo "[$(date)] FAILED: $model_name (see $log)"
    rc=1
  fi
done

echo
echo "[$(date)] Persisting staged cache to CFS: $DEST_HF"
rsync -a "${STAGE_HF}/" "${DEST_HF}/"

echo
echo "[$(date)] Completed UMA download job. Logs in $RUN_DIR"
exit "$rc"
