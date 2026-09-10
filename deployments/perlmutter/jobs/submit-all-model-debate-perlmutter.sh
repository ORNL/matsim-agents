#!/bin/bash
# ---------------------------------------------------------------------------
# submit-all-model-debate-perlmutter.sh
#
# Submits one job-serve-multinode-perlmutter.sh Slurm job per
# open-model-catalog.json entry, sized per the table in that job script's
# header. Prints the base_url_env export lines to run once every job reaches
# RUNNING and its head-node IP is known (see squeue/job logs), so
# all_model_scientific_debate.py can be pointed at the live endpoints.
#
# This submits real multi-node GPU allocations (25 nodes total if all run
# concurrently) -- review the per-model --nodes counts below before running.
#
# Usage:
#   export PROJECT_ROOT=/path/to/matsim-agents
#   deployments/perlmutter/jobs/submit-all-model-debate-perlmutter.sh [model_name ...]
#
# With no arguments, submits all 10 models. Pass one or more catalog `name`
# fields (see deployments/common/open-model-catalog.json) to submit a subset.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
MODELS_DIR="$PROJ/models"
JOB_SCRIPT="$SCRIPT_DIR/job-serve-multinode-perlmutter.sh"

# name : model_dir : nodes : base_url_env : extra_args
CATALOG=(
  "kimi-k2.5:Kimi-K2.5:4:MATSIM_VLLM_KIMI_K25_BASE_URL:"
  "glm-4.7:GLM-4.7:4:MATSIM_VLLM_GLM47_BASE_URL:"
  "glm-4.7-flash:GLM-4.7-Flash:1:MATSIM_VLLM_GLM47_FLASH_BASE_URL:"
  "deepseek-v3.2:DeepSeek-V3.2:4:MATSIM_VLLM_DEEPSEEK_V32_BASE_URL:"
  "mistral-large-3:Mistral-Large-3-675B-Instruct-2512:4:MATSIM_VLLM_MISTRAL_LARGE3_BASE_URL:--tokenizer-mode mistral --config-format mistral --load-format mistral"
  "qwen3-235b-a22b-instruct-2507:Qwen3-235B-A22B-Instruct-2507:2:MATSIM_VLLM_QWEN3_235B_INSTRUCT_BASE_URL:"
  "qwen3-235b-a22b-thinking-2507:Qwen3-235B-A22B-Thinking-2507:2:MATSIM_VLLM_QWEN3_235B_THINKING_BASE_URL:"
  "devstral-2:Devstral-2-123B-Instruct-2512:2:MATSIM_VLLM_DEVSTRAL2_BASE_URL:"
  "gemma-4-31b-it:gemma-4-31B-it:1:MATSIM_VLLM_GEMMA4_31B_BASE_URL:"
  "gemma-4-26b-a4b-it:gemma-4-26B-A4B-it:1:MATSIM_VLLM_GEMMA4_26B_BASE_URL:"
)

WANT=("$@")

echo "=========================================="
echo "Submitting multi-node vLLM serve jobs"
echo "=========================================="

for entry in "${CATALOG[@]}"; do
  IFS=: read -r name dir nodes base_url_env extra <<< "$entry"
  if (( ${#WANT[@]} > 0 )); then
    match=0
    for w in "${WANT[@]}"; do [[ "$w" == "$name" ]] && match=1; done
    (( match )) || continue
  fi
  model_path="$MODELS_DIR/$dir"
  if [[ ! -d "$model_path" ]]; then
    echo "SKIP $name: model dir not found: $model_path" >&2
    continue
  fi
  echo ""
  echo "--- $name  (nodes=$nodes)  -> $base_url_env ---"
  # premium/gpu_premium QOS caps at 5 submitted jobs per user (MaxSubmitPU=5);
  # fall back to regular/gpu_regular (MaxSubmitPU=5000) once that's exhausted.
  jobid=$(SERVE_MODEL_PATH="$model_path" SERVE_MODEL_NAME="$name" \
          SERVE_EXTRA_ARGS="$extra" \
          sbatch --parsable -A "${SLURM_ACCOUNT:-m5216_g}" --nodes="$nodes" -J "vllm-$name" "$JOB_SCRIPT" \
          2>/tmp/submit-all-model-debate.$$.err) || {
    if grep -q QOSMaxSubmitJobPerUserLimit /tmp/submit-all-model-debate.$$.err 2>/dev/null; then
      echo "  premium QOS full, retrying with -q regular ..."
      jobid=$(SERVE_MODEL_PATH="$model_path" SERVE_MODEL_NAME="$name" \
              SERVE_EXTRA_ARGS="$extra" \
              sbatch --parsable -A "${SLURM_ACCOUNT:-m5216_g}" -q regular --nodes="$nodes" -J "vllm-$name" "$JOB_SCRIPT")
    else
      cat /tmp/submit-all-model-debate.$$.err >&2
    fi
  }
  rm -f /tmp/submit-all-model-debate.$$.err
  echo "  submitted job $jobid"
  echo "  once RUNNING, get head-node IP from: %x-%j.out (job name vllm-$name-$jobid)"
  echo "  then: export ${base_url_env}=http://<head_node_ip>:8000/v1"
done

echo ""
echo "Track with: squeue -u \$USER"
