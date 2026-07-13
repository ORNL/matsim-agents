#!/bin/bash
#SBATCH -A amsc001
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
#   sbatch scripts/download/perlmutter/download-uma-perlmutter.sh
#
# Multiple / alternate models:
#   UMA_MODELS="uma-s-1p1 uma-m-1p1" \
#   sbatch scripts/download/perlmutter/download-uma-perlmutter.sh
#
# Alternate cache location:
#   HF_HOME=/global/cfs/projectdirs/m5216/mlupopa/models/hf_cache \
#   sbatch scripts/download/perlmutter/download-uma-perlmutter.sh
#
# Notes:
# - facebook/UMA is a GATED repo: you must accept the license on Hugging Face
#   and provide a token. This script reads ~/.cache/huggingface/token if HF_TOKEN
#   is not already set (run `hf auth login` once beforehand).
# - Downloads are resumable; rerunning skips files already in the cache.
# - Runs from fairchem_venv (UMA requires numpy>=2), NOT hydragnn_venv.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

# fairchem_venv lives alongside hydragnn_venv under the HydraGNN install root.
VENV_ROOT="$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"
VENV="${MATSIM_FAIRCHEM_VENV:-${VENV_ROOT}/fairchem_venv}"
RUN_DIR="$PROJ/runs/download-uma-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: fairchem_venv not found: $VENV" >&2
  echo "Build it first with:" >&2
  echo "  INSTALL_UMA=1 bash scripts/setup/perlmutter/install_matsim_perlmutter.sh --gpu" >&2
  exit 1
fi

# ── modules & venv (fairchem_venv for UMA) ───────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
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
