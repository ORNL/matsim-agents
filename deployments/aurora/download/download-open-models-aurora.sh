#!/bin/bash
#PBS -N dl-open-models
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Download open/local-serving model set on Aurora.
#
# Submit (default open model set):
#   qsub deployments/aurora/download/download-open-models-aurora.sh
#
# Submit a subset:
#   qsub -v MODEL_IDS="Qwen/Qwen2.5-14B-Instruct mistralai/Mixtral-8x22B-Instruct-v0.1" \
#        deployments/aurora/download/download-open-models-aurora.sh
#
# Alternate destination:
#   qsub -v MODEL_ROOT=/path/to/project/models \
#        deployments/aurora/download/download-open-models-aurora.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Safety policy:
# - This downloader must run from the HydraGNN Aurora environment.
# - This script never runs pip installs or mutates Python package versions.
# - If tooling is missing, refresh the environment via setup scripts instead of
#   installing ad-hoc packages here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

VENV_DEFAULT="${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv"
VENV="${MATSIM_AURORA_VENV:-$VENV_DEFAULT}"
MODEL_ROOT="${MODEL_ROOT:-$PROJ/models}"
JOB_TAG="${PBS_JOBID:-manual}"
RUN_DIR="$PROJ/runs/download-open-models-aurora-${JOB_TAG}"
mkdir -p "$RUN_DIR" "$MODEL_ROOT"

if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Aurora HydraGNN env not found: $VENV" >&2
  echo "Build it first with: bash deployments/aurora/setup/install_matsim_aurora.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Guard against unsupported Hugging Face CLI stacks that can overwrite
# sensitive HydraGNN dependencies (for example click/typer constraints).
if "${VENV}/bin/python" - <<'PY'
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
    echo "Recreate/fix the Aurora environment with:" >&2
    echo "  INSTALL_LLM_EXTRAS=1 bash deployments/aurora/setup/install_matsim_aurora.sh" >&2
    exit 1
  fi
  exit $rc
fi

DEFAULT_MODELS=(
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "Qwen/Qwen3-235B-A22B-Instruct-2507"
  "Qwen/Qwen3-235B-A22B-Thinking-2507"
  # --- Kimi / GLM ---
  "moonshotai/Kimi-K2.5"
  "zai-org/GLM-4.7"
  "zai-org/GLM-4.7-Flash"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "mistralai/Mistral-Large-3-675B-Instruct-2512"
  "mistralai/Devstral-2-123B-Instruct-2512"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "deepseek-ai/DeepSeek-V3.2"
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
  echo "Do not run ad-hoc pip upgrades in HydraGNN venv." >&2
  echo "Refresh the environment with pinned extras instead:" >&2
  echo "  INSTALL_LLM_EXTRAS=1 bash deployments/aurora/setup/install_matsim_aurora.sh" >&2
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
