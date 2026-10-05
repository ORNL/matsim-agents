#!/bin/bash
#SBATCH -J campaign-formula-e2e
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
# ---------------------------------------------------------------------------
# matsim-agents: end-to-end integration test for the campaign formula-
# discovery stage (element-set-to-formula enumeration + LLM-formula merge,
# see deployments/perlmutter/jobs/campaign_formula_discovery.py).
#
# A panel of *different* models from the local model zoo is served, each on
# its own OpenAI-compatible vLLM endpoint (isolated venv_vllm). A thin client
# (matsim-agents/.venv) runs one real "equal" multi-LLM debate over the
# Nb-Ta-O element set (see the technical plan discussed with the user), then
# runs the new deterministic formula generator + LLM-formula merge against
# the real debate transcript -- no mocked LLM calls. This intentionally stops
# at the formula registry: AFLOW/pyXtal candidate generation, MLIP
# relaxation, and DFT labelling already exist for a single composition and
# are exercised by other jobs (job-al-debate-portability-perlmutter.sh).
#
# Layout (single node, 4x A100 80GB):
#   GPU 0 :8000  qwen      Qwen2.5-14B-Instruct
#   GPU 1 :8001  deepseek  DeepSeek-R1-Distill-Qwen-32B
#   GPU 2 :8002  gemma     gemma-4-31B-it
#   GPU 3        (spare)
#
# Smoke test (one small model, ~10 min):
#   MATSIM_CAMPAIGN_SMOKE=1 sbatch deployments/perlmutter/jobs/job-campaign-formula-discovery-perlmutter.sh
#
# Full run:
#   PROJECT_ROOT=$PWD sbatch -A m5216_g -q gpu_premium deployments/perlmutter/jobs/job-campaign-formula-discovery-perlmutter.sh
# ---------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv          # debate + campaign client (matsim-agents + openai)
VLLM_VENV=$REPO/venv_vllm                 # isolated vLLM servers
MODELS_DIR=$PROJ/models
init_run_dirs "$PROJ" "campaign-formula-e2e" "${SLURM_JOB_ID:-$$}"

RUNS_ROOT=${MATSIM_RUNS_ROOT:-$PROJ/runs/campaign-formula-e2e}

# ── panel definition: "GPU PORT MODEL_DIR" per server ────────────────────────
SMOKE=${MATSIM_CAMPAIGN_SMOKE:-0}
if [[ "$SMOKE" == "1" ]]; then
  SERVERS=( "0 8000 ${MODELS_DIR}/Qwen2.5-14B-Instruct" )
  MODEL_ARGS=( --model qwen vllm Qwen2.5-14B-Instruct "http://127.0.0.1:8000/v1" )
else
  SERVERS=(
    "0 8000 ${MODELS_DIR}/Qwen2.5-14B-Instruct"
    "1 8001 ${MODELS_DIR}/DeepSeek-R1-Distill-Qwen-32B"
    "2 8002 ${MODELS_DIR}/gemma-4-31B-it"
  )
  MODEL_ARGS=(
    --model qwen     vllm Qwen2.5-14B-Instruct           "http://127.0.0.1:8000/v1"
    --model deepseek vllm DeepSeek-R1-Distill-Qwen-32B    "http://127.0.0.1:8001/v1"
    --model gemma    vllm gemma-4-31B-it                  "http://127.0.0.1:8002/v1"
  )
fi

# ── fail closed if any panel model isn't actually checked out on CFS ─────────
# (same "local-only" policy as benchmarks/portability/all_model_scientific_debate.py)
MISSING=()
for spec in "${SERVERS[@]}"; do
  read -r _g _p model_dir <<<"$spec"
  [[ -d "$model_dir" ]] || MISSING+=("$model_dir")
done
if ((${#MISSING[@]} > 0)); then
  echo "[ERROR] not locally stored under $MODELS_DIR: ${MISSING[*]}" >&2
  exit 2
fi

# The 32B models share one A100 each with their KV cache. Keep the qualified
# default conservative; larger contexts remain an explicit runtime override.
VLLM_MAXLEN=${MATSIM_VLLM_MAXLEN:-8192}
VLLM_GPU_UTIL=${MATSIM_VLLM_GPU_UTIL:-0.90}
declare -a VLLM_PIDS=()

# ── launch each vLLM server (background, isolated env) ───────────────────────
launch_server() {
  local gpu="$1" port="$2" model_dir="$3"
  local name; name="$(basename "$model_dir")"
  local log="$RUN_DIR/vllm-${name}-${port}.log"
  echo "[$(date)] serving $name on GPU $gpu :$port"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_DO_NOT_TRACK=1 PYTHONNOUSERSITE=1
    export VLLM_USE_FLASHINFER_SAMPLER=0
    local jit="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}.${port}"
    mkdir -p "$jit"
    export FLASHINFER_WORKSPACE_BASE="$jit" TRITON_CACHE_DIR="$jit/triton"
    export TORCHINDUCTOR_CACHE_DIR="$jit/inductor" VLLM_CACHE_ROOT="$jit/vllm"
    local pyhdr="$REPO/.venv/include/python3.11"
    export CPATH="${pyhdr}:${CPATH:-}" C_INCLUDE_PATH="${pyhdr}:${C_INCLUDE_PATH:-}"
    exec "$VLLM_VENV/bin/vllm" serve "$model_dir" \
        --served-model-name "$name" \
        --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size 1 \
        --max-model-len "$VLLM_MAXLEN" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --enforce-eager
  ) >"$log" 2>&1 &
  VLLM_PIDS+=("$!")
}

cleanup() {
  echo "[$(date)] stopping ${#VLLM_PIDS[@]} vLLM server(s) ..."
  for pid in "${VLLM_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${VLLM_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

for spec in "${SERVERS[@]}"; do
  # shellcheck disable=SC2086
  launch_server $spec
done

# ── wait for every endpoint to report ready ──────────────────────────────────
wait_ready() {
  local port="$1" url="http://127.0.0.1:${port}/v1/models"
  for _ in $(seq 1 300); do
    local alive=0
    for pid in "${VLLM_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ "$alive" == "0" ]] && { echo "[ERROR] a vLLM server exited early" >&2; return 1; }
    if grep -qaE "Engine core initialization failed|OutOfMemoryError|Failed to load model" "$RUN_DIR"/vllm-*-"${port}".log 2>/dev/null; then
      echo "[ERROR] engine on :$port failed to initialize" >&2; return 1
    fi
    curl -fsS "$url" >/dev/null 2>&1 && { echo "[$(date)] :$port ready"; return 0; }
    sleep 10
  done
  echo "[ERROR] endpoint :$port not ready in time" >&2; return 1
}
for spec in "${SERVERS[@]}"; do
  read -r _g port _m <<<"$spec"
  wait_ready "$port" || { tail -n 40 "$RUN_DIR"/vllm-*.log >&2 || true; exit 1; }
done

# ── client venv (matsim-agents + openai HTTP client; no GPU work) ────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MATSIM_LLM_PROVIDER=vllm MATSIM_VLLM_API_KEY=EMPTY

echo "[$(date)] running campaign formula-discovery driver ..."
CUDA_VISIBLE_DEVICES="" python "$REPO/deployments/perlmutter/jobs/campaign_formula_discovery.py" \
    --elements Nb Ta O \
    --oxidation-state Nb:3,4,5 \
    --oxidation-state Ta:3,4,5 \
    --oxidation-state O:-2 \
    --max-coefficient 6 \
    --max-atoms 12 \
    --rounds "${MATSIM_CAMPAIGN_ROUNDS:-2}" \
    --campaign-id "nb-ta-o-e2e-${SLURM_JOB_ID:-$$}" \
    --output-dir "$OUTPUT_DIR" \
    --output-root "$RUNS_ROOT" \
    "${MODEL_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/campaign-formula-discovery.log"
STATUS=${PIPESTATUS[0]}

echo "[$(date)] done (exit=$STATUS). Artifacts in $OUTPUT_DIR"
exit "$STATUS"
