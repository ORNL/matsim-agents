#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J dl-open-models
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
# ---------------------------------------------------------------------------
# Download open/local-serving model set on Perlmutter.
#
# Submit (default open model set):
#   sbatch scripts/download/perlmutter/download-open-models-perlmutter.sh
#
# Submit a subset:
#   MODEL_IDS="Qwen/Qwen2.5-14B-Instruct mistralai/Mixtral-8x22B-Instruct-v0.1" \
#   sbatch scripts/download/perlmutter/download-open-models-perlmutter.sh
#
# Alternate destination:
#   MODEL_ROOT=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/models \
#   sbatch scripts/download/perlmutter/download-open-models-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV_LOCAL_DEFAULT="$REPO/perlmutter_venv"
VENV_SHARED_DEFAULT="$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv"
VENV="${VENV:-${MATSIM_PERLMUTTER_VENV:-$VENV_LOCAL_DEFAULT}}"
if [[ ! -d "$VENV" && -d "$VENV_SHARED_DEFAULT" ]]; then
  VENV="$VENV_SHARED_DEFAULT"
fi
MODEL_ROOT="${MODEL_ROOT:-$PROJ/models}"
RUN_DIR="$PROJ/runs/download-open-models-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR" "$MODEL_ROOT"

source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Perlmutter HydraGNN env not found: $VENV" >&2
  echo "Build it first with: bash scripts/setup/perlmutter/install_matsim_perlmutter.sh --gpu" >&2
  exit 1
fi

export PATH="$VENV/bin:$PATH"
export CONDA_PREFIX="$VENV"
export CONDA_DEFAULT_ENV="hydragnn_venv"

DEFAULT_MODELS=(
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "google/gemma-4-31B-it"
  "google/gemma-4-26B-A4B-it"
  "HuggingFaceTB/SmolLM3-3B"
)

if [[ -n "${MODEL_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=( ${MODEL_IDS} )
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "[$(date)] Download destination: $MODEL_ROOT"
echo "[$(date)] Models to download:"
for m in "${MODELS[@]}"; do echo "  - $m"; done

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found in active environment." >&2
  echo "Try: pip install -U huggingface_hub" >&2
  exit 1
fi

for model_id in "${MODELS[@]}"; do
  leaf="${model_id##*/}"
  dest="$MODEL_ROOT/$leaf"
  log="$RUN_DIR/${leaf}.download.log"

  mkdir -p "$dest"
  echo
  echo "[$(date)] Downloading $model_id -> $dest"

  if hf download "$model_id" --local-dir "$dest" >"$log" 2>&1; then
    shards=$(ls "$dest"/*.safetensors 2>/dev/null | wc -l || true)
    echo "[$(date)] DONE: $model_id (safetensors shards: $shards)"
  else
    echo "[$(date)] FAILED: $model_id (see $log)"
  fi
done

echo
echo "[$(date)] Completed download job. Logs in $RUN_DIR"
