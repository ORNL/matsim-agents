#!/bin/bash
#SBATCH -A mat746
#SBATCH -J dl-open-models
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/download-open-models-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/download-open-models-%j/job-%j.out
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# Download open/local-serving model set on Frontier.
#
# Submit (all default models):
#   sbatch deployments/frontier/download/download-open-models-frontier.sh
#
# Submit a subset:
#   MODEL_IDS="Qwen/Qwen2.5-14B-Instruct mistralai/Mixtral-8x22B-Instruct-v0.1" \
#   sbatch deployments/frontier/download/download-open-models-frontier.sh
#
# Alternate destination:
#   MODEL_ROOT=/lustre/orion/mat746/proj-shared/models \
#   sbatch deployments/frontier/download/download-open-models-frontier.sh
#
# ---------------------------------------------------------------------------

set -euo pipefail

# Safety policy:
# - This downloader must run from the HydraGNN Frontier environment.
# - This script never runs pip installs or mutates Python package versions.
# - If tooling is missing, refresh the environment via setup scripts instead of
#   installing ad-hoc packages here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/lustre/orion/mat746/proj-shared/matsim-agents
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_ROOT=${MODEL_ROOT:-$PROJ/models}
RUN_DIR=$PROJ/runs/download-open-models-${SLURM_JOB_ID:-manual}
mkdir -p "$RUN_DIR" "$MODEL_ROOT"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

# Guard against unsupported Hugging Face CLI stacks that can overwrite
# sensitive HydraGNN dependencies (for example click/typer constraints).
if "$VENV/bin/python" - <<'PY'
import importlib.metadata as m
try:
    v = m.version("huggingface_hub")
except m.PackageNotFoundError:
    raise SystemExit(0)
major = int(v.split(".", 1)[0])
raise SystemExit(42 if major >= 1 else 0)
PY
then
  :
else
  rc=$?
  if [[ $rc -eq 42 ]]; then
    echo "ERROR: Unsupported huggingface_hub>=1.0 detected in HydraGNN venv: $VENV" >&2
    echo "Do not upgrade packages in-place in this environment." >&2
    echo "Recreate/fix the Frontier environment with:" >&2
    echo "  INSTALL_LLM_EXTRAS=1 bash deployments/frontier/setup/install_matsim_frontier.sh --rocm72" >&2
    exit 1
  fi
  exit $rc
fi

# ---------------------------------------------------------------------------
# Model catalogue
# Open-access model set only.
# ---------------------------------------------------------------------------
DEFAULT_MODELS=(
  # --- Qwen2.5 ---
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  # --- Qwen3 (MoE / dense) ---
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  # --- Mistral ---
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  # --- DeepSeek ---
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  # --- Google Gemma ---
  "google/gemma-4-31B-it"
  "google/gemma-4-26B-A4B-it"
  # --- SmolLM ---
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
  echo "Do not run ad-hoc pip upgrades in HydraGNN venv." >&2
  echo "Refresh the environment with pinned extras instead:" >&2
  echo "  INSTALL_LLM_EXTRAS=1 bash deployments/frontier/setup/install_matsim_frontier.sh --rocm72" >&2
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
