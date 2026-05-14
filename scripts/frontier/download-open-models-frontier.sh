#!/bin/bash
#SBATCH -A mat746
#SBATCH -J dl-open-models
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/download-models-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/download-models-%j/job-%j.out
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# Download the six benchmark open-source/open-weight models to local storage.
#
# Submit:
#   sbatch scripts/frontier/download-open-models-frontier.sh
#
# Optional: choose a subset at submit time
#   MODEL_IDS="Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct" \
#   sbatch scripts/frontier/download-open-models-frontier.sh
#
# Optional: alternate destination root
#   MODEL_ROOT=/lustre/orion/mat746/proj-shared/models \
#   sbatch scripts/frontier/download-open-models-frontier.sh
#
# Notes:
# - Llama models are gated by Meta terms. Run "hf auth login" first.
# - Downloads are resumable; rerunning skips completed files.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
REPO=$PROJ/matsim-agents
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_ROOT=${MODEL_ROOT:-$PROJ/models}
RUN_DIR=$PROJ/runs/download-models-$SLURM_JOB_ID
mkdir -p "$RUN_DIR" "$MODEL_ROOT"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

DEFAULT_MODELS=(
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

if [[ -n "${MODEL_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=( ${MODEL_IDS} )
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "[$(date)] Download destination: $MODEL_ROOT"
echo "[$(date)] Models to download:"
for m in "${MODELS[@]}"; do
  echo "  - $m"
done

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

  # Resume-safe download. If gated access is missing, hf exits with non-zero.
  if hf download "$model_id" --local-dir "$dest" >"$log" 2>&1; then
    shards=$(ls "$dest"/*.safetensors 2>/dev/null | wc -l || true)
    echo "[$(date)] DONE: $model_id (safetensors shards: $shards)"
  else
    echo "[$(date)] FAILED: $model_id (see $log)"
  fi
done

echo
echo "[$(date)] Completed download job. Logs in $RUN_DIR"
