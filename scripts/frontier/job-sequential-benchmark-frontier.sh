#!/bin/bash
#SBATCH -A mat746
#SBATCH -J seq-model-bench
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/seq-model-bench-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/seq-model-bench-%j/job-%j.out
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: sequential single-node model benchmark.
#
# For each model in the spec file this job:
#   1. Starts a vLLM server on localhost:8000 pointing at the local model
#   2. Waits for the server to be ready (health-check loop)
#   3. Runs eval_six_models_search_prompt.py for that single model
#   4. Kills the vLLM server
#   5. Repeats for the next model
# After all models finish:
#   6. Merges all per-model JSON results into one combined report
#   7. Runs rank_model_eval.py  → leaderboard CSV
#   8. Runs plot_model_eval.py  → comparison PNG
#
# Required env var at submission:
#   BENCHMARK_PROMPT   – the shared prompt text
#
# Optional env vars:
#   BENCHMARK_MODELS       – space-separated list of "name:hf_id:local_path" triples.
#                            Defaults to the 8-model list below.
#   BENCHMARK_OUTPUT_PREFIX – path prefix for output artifacts
#   BENCHMARK_KEYWORDS      – comma-separated keyword list
#   BENCHMARK_TEMPERATURE   – float (default 0.0)
#   BENCHMARK_TITLE         – plot title
#   BENCHMARK_SKIP_PLOT=1   – skip PNG generation
#   BENCHMARK_TP_SIZE       – tensor-parallel size per model (default 4 = full node)
#   VLLM_PORT               – port for vLLM (default 8000)
#
# Example submission:
#   BENCHMARK_PROMPT="Search for a Pb-free halide double perovskite and justify stability." \
#   sbatch scripts/frontier/job-sequential-benchmark-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
REPO=$PROJ/matsim-agents
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_ROOT=$PROJ/models
RUN_DIR=$PROJ/runs/seq-model-bench-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${BENCHMARK_PROMPT:-}" ]]; then
  echo "ERROR: BENCHMARK_PROMPT is required." >&2
  echo "Example:" >&2
  echo "  BENCHMARK_PROMPT='Search for a Pb-free halide double perovskite.' sbatch $0" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ── No outbound connections (HPC: no internet access) ────────────────────────
# Prevent any model auto-download during inference (all models are local)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM usage telemetry
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray telemetry
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
# Triton: disable remote kernel downloads
export TRITON_DISABLE_AUTOTUNE_CACHE=1
# Block all HTTP/S proxies
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# Use prebuilt tvm_ffi torch-c-dlpack .so from proj-shared (avoids JIT rebuild hang at import)
export TVM_FFI_CACHE_DIR=$PROJ/cache/tvm-ffi
TVM_FFI_SO=$TVM_FFI_CACHE_DIR/libtorch_c_dlpack_addon_torch211-rocm.so
if [[ ! -s "$TVM_FFI_SO" ]]; then
  echo "[FAIL] Missing or empty tvm_ffi prebuilt: $TVM_FFI_SO" >&2
  echo "       Rebuild with: scripts/frontier/prebuild-tvm-ffi-frontier.sh" >&2
  exit 1
fi
rm -f ~/.cache/tvm-ffi/*.lock 2>/dev/null || true

cd "$REPO"

PYTHON="$VENV/bin/python3"
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
# Use full node (4 GCDs on MI250X = 8 logical GPUs) by default.
# Override with BENCHMARK_TP_SIZE for smaller models that fit on fewer GCDs.
TP_SIZE=${BENCHMARK_TP_SIZE:-8}

OUTPUT_PREFIX=${BENCHMARK_OUTPUT_PREFIX:-$RUN_DIR/seq-benchmark}
KEYWORDS=${BENCHMARK_KEYWORDS:-stability,band gap,formation energy,synthesis}
TEMPERATURE=${BENCHMARK_TEMPERATURE:-0.0}
TITLE=${BENCHMARK_TITLE:-Sequential Single-Node matsim-agents Benchmark}
SKIP_PLOT=${BENCHMARK_SKIP_PLOT:-0}
# Cap KV-cache allocation. Models like Gemma 4 and Qwen3.6 have 256K native
# context windows; without this vLLM will OOM trying to pre-allocate the KV cache.
# Set to 0 to let each model use its native context length.
MAX_MODEL_LEN=${BENCHMARK_MAX_MODEL_LEN:-8192}

# ---------------------------------------------------------------------------
# Model list: "display_name:hf_model_id:local_dir_name"
# Override BENCHMARK_MODELS to change the set, or set BENCHMARK_PART=heavy|light.
# ---------------------------------------------------------------------------
DEFAULT_MODELS=(
  "qwen3-32b:Qwen/Qwen3-32B:Qwen3-32B"
  "qwen2.5-72b-instruct:Qwen/Qwen2.5-72B-Instruct:Qwen2.5-72B-Instruct"
  "qwen2.5-14b-instruct:Qwen/Qwen2.5-14B-Instruct:Qwen2.5-14B-Instruct"
  "llama-3.3-70b-instruct:meta-llama/Llama-3.3-70B-Instruct:Llama-3.3-70B-Instruct"
  "llama-3.1-70b-instruct:meta-llama/Llama-3.1-70B-Instruct:Llama-3.1-70B-Instruct"
  "llama-3.1-8b-instruct:meta-llama/Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct"
  "mixtral-8x22b-instruct:mistralai/Mixtral-8x22B-Instruct-v0.1:Mixtral-8x22B-Instruct-v0.1"
  "deepseek-r1-distill-qwen-32b:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:DeepSeek-R1-Distill-Qwen-32B"
  "qwen3.6-27b:Qwen/Qwen3.6-27B:Qwen3.6-27B"
  "qwen3.6-35b-a3b:Qwen/Qwen3.6-35B-A3B:Qwen3.6-35B-A3B"
  "gemma-4-31b-it:google/gemma-4-31B-it:gemma-4-31B-it"
  "gemma-4-26b-a4b-it:google/gemma-4-26B-A4B-it:gemma-4-26B-A4B-it"
  "smollm3-3b:HuggingFaceTB/SmolLM3-3B:SmolLM3-3B"
)

# BENCHMARK_PART=heavy  → large dense/MoE models (70B+, 32B dense)  ~75-90 min
# BENCHMARK_PART=light  → medium/small models (≤35B MoE, ≤14B dense) ~40-55 min
# BENCHMARK_PART=all    → all 13 models (may exceed 2h debug limit)
HEAVY_MODELS=(
  "mixtral-8x22b-instruct:mistralai/Mixtral-8x22B-Instruct-v0.1:Mixtral-8x22B-Instruct-v0.1"
  "qwen2.5-72b-instruct:Qwen/Qwen2.5-72B-Instruct:Qwen2.5-72B-Instruct"
  "llama-3.3-70b-instruct:meta-llama/Llama-3.3-70B-Instruct:Llama-3.3-70B-Instruct"
  "llama-3.1-70b-instruct:meta-llama/Llama-3.1-70B-Instruct:Llama-3.1-70B-Instruct"
  "qwen3-32b:Qwen/Qwen3-32B:Qwen3-32B"
  "deepseek-r1-distill-qwen-32b:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:DeepSeek-R1-Distill-Qwen-32B"
)

LIGHT_MODELS=(
  "qwen3.6-27b:Qwen/Qwen3.6-27B:Qwen3.6-27B"
  "qwen3.6-35b-a3b:Qwen/Qwen3.6-35B-A3B:Qwen3.6-35B-A3B"
  "gemma-4-31b-it:google/gemma-4-31B-it:gemma-4-31B-it"
  "gemma-4-26b-a4b-it:google/gemma-4-26B-A4B-it:gemma-4-26B-A4B-it"
  "qwen2.5-14b-instruct:Qwen/Qwen2.5-14B-Instruct:Qwen2.5-14B-Instruct"
  "llama-3.1-8b-instruct:meta-llama/Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct"
  "smollm3-3b:HuggingFaceTB/SmolLM3-3B:SmolLM3-3B"
)

if [[ -n "${BENCHMARK_MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<< "$BENCHMARK_MODELS"
elif [[ "${BENCHMARK_PART:-all}" == "heavy" ]]; then
  MODEL_LIST=("${HEAVY_MODELS[@]}")
elif [[ "${BENCHMARK_PART:-all}" == "light" ]]; then
  MODEL_LIST=("${LIGHT_MODELS[@]}")
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

# ---------------------------------------------------------------------------
# Helper: wait for vLLM health endpoint
# ---------------------------------------------------------------------------
wait_for_vllm() {
  local port=$1
  local max_wait=300  # seconds
  local interval=5
  local elapsed=0
  echo "[vllm] Waiting for server on port $port ..."
  while true; do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "[vllm] Server ready after ${elapsed}s."
      return 0
    fi
    if (( elapsed >= max_wait )); then
      echo "[vllm] ERROR: server did not become ready within ${max_wait}s." >&2
      return 1
    fi
    sleep $interval
    (( elapsed += interval ))
  done
}

# ---------------------------------------------------------------------------
# Helper: build a single-model JSON spec and write to a temp file
# ---------------------------------------------------------------------------
make_single_spec() {
  local name=$1
  local hf_id=$2
  local out_file=$3
  "$PYTHON" -c "
import json
spec = [{'name': '$name', 'provider': 'vllm', 'model': '$hf_id',
         'base_url_env': 'MATSIM_VLLM_BASE_URL', 'api_key_env': 'MATSIM_VLLM_API_KEY'}]
open('$out_file', 'w').write(json.dumps(spec, indent=2))
"
}

# ---------------------------------------------------------------------------
# Main loop: start vLLM → eval → stop, one model at a time
# ---------------------------------------------------------------------------
PER_MODEL_JSONS=()
VLLM_PID=""

cleanup_vllm() {
  if [[ -n "$VLLM_PID" ]]; then
    echo "[vllm] Stopping server PID $VLLM_PID ..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}
trap cleanup_vllm EXIT

for entry in "${MODEL_LIST[@]}"; do
  IFS=: read -r MODEL_NAME HF_ID LOCAL_DIR <<< "$entry"
  LOCAL_PATH="$MODEL_ROOT/$LOCAL_DIR"

  if [[ ! -d "$LOCAL_PATH" ]]; then
    echo "[SKIP] $MODEL_NAME — local path not found: $LOCAL_PATH" | tee -a "$RUN_DIR/skipped.txt"
    continue
  fi

  echo ""
  echo "========================================================"
  echo " MODEL: $MODEL_NAME  ($HF_ID)"
  echo " PATH:  $LOCAL_PATH"
  echo "========================================================"

  # --- Start vLLM server ---
  export MATSIM_VLLM_BASE_URL="$VLLM_BASE_URL"
  export MATSIM_VLLM_API_KEY="EMPTY"

  "$VENV/bin/vllm" serve "$LOCAL_PATH" \
    --served-model-name "$HF_ID" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --disable-log-requests \
    ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
    > "$RUN_DIR/vllm-${MODEL_NAME}.log" 2>&1 &
  VLLM_PID=$!
  echo "[vllm] Server started PID=$VLLM_PID"

  # --- Wait for readiness ---
  if ! wait_for_vllm "$VLLM_PORT"; then
    echo "[ERROR] vLLM failed for $MODEL_NAME — see $RUN_DIR/vllm-${MODEL_NAME}.log" >&2
    cleanup_vllm
    continue
  fi

  # --- Build single-model spec ---
  SPEC_TMP="$RUN_DIR/spec-${MODEL_NAME}.json"
  make_single_spec "$MODEL_NAME" "$HF_ID" "$SPEC_TMP"

  # --- Run eval ---
  MODEL_JSON="$RUN_DIR/eval-${MODEL_NAME}.json"
  "$PYTHON" scripts/eval_six_models_search_prompt.py \
    --prompt "$BENCHMARK_PROMPT" \
    --spec-file "$SPEC_TMP" \
    --keywords "$KEYWORDS" \
    --temperature "$TEMPERATURE" \
    --out "$MODEL_JSON"

  PER_MODEL_JSONS+=("$MODEL_JSON")
  echo "[eval] Wrote $MODEL_JSON"

  # --- Stop vLLM ---
  cleanup_vllm
  # Brief pause to allow GPU memory to fully release before next model
  sleep 10
done

# ---------------------------------------------------------------------------
# Merge per-model JSON results into one combined report
# ---------------------------------------------------------------------------
if [[ ${#PER_MODEL_JSONS[@]} -eq 0 ]]; then
  echo "ERROR: No models evaluated successfully." >&2
  exit 1
fi

COMBINED_JSON="${OUTPUT_PREFIX}.json"
echo ""
echo "Merging ${#PER_MODEL_JSONS[@]} per-model results → $COMBINED_JSON"

JSONS_ARG=$(printf '"%s",' "${PER_MODEL_JSONS[@]}")
JSONS_ARG="[${JSONS_ARG%,}]"

"$PYTHON" - <<PYEOF
import json, sys
from pathlib import Path

files = $JSONS_ARG
all_results = []
all_models  = []
first = None
for f in files:
    data = json.loads(Path(f).read_text())
    if first is None:
        first = data
    all_results.extend(data.get("results", []))
    all_models.extend(data.get("models", []))

combined = {
    "summary": {
        **first["summary"],
        "num_models": len(all_results),
        "num_success": sum(1 for r in all_results if r["status"] == "ok"),
        "num_errors":  sum(1 for r in all_results if r["status"] != "ok"),
    },
    "models":  all_models,
    "results": all_results,
}
Path("$COMBINED_JSON").write_text(json.dumps(combined, indent=2))
print(f"Combined report: $COMBINED_JSON  ({len(all_results)} models)")
PYEOF

# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------
LEADERBOARD_CSV="${OUTPUT_PREFIX}-leaderboard.csv"
echo "Ranking results → $LEADERBOARD_CSV"
"$PYTHON" scripts/rank_model_eval.py \
  --input "$COMBINED_JSON" \
  --out-csv "$LEADERBOARD_CSV" \
  --out-json "${OUTPUT_PREFIX}-ranked.json"

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if [[ "$SKIP_PLOT" != "1" ]]; then
  CHART_PNG="${OUTPUT_PREFIX}-chart.png"
  echo "Plotting → $CHART_PNG"
  "$PYTHON" scripts/plot_model_eval.py \
    --input-csv "$LEADERBOARD_CSV" \
    --out "$CHART_PNG" \
    --title "$TITLE" || echo "[WARN] Plot failed (matplotlib may be unavailable). Skipping."
fi

echo ""
echo "=== Benchmark complete ==="
echo "  Combined JSON : $COMBINED_JSON"
echo "  Leaderboard   : $LEADERBOARD_CSV"
[[ "$SKIP_PLOT" != "1" ]] && echo "  Chart         : ${OUTPUT_PREFIX}-chart.png"
echo "  Per-model logs: $RUN_DIR/vllm-*.log"
