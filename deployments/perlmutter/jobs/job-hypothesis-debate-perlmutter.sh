#!/bin/bash
#SBATCH -J hyp-debate
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
# ---------------------------------------------------------------------------
# matsim-agents: LLM hypothesis + MULTI-LLM debate demonstration, showing how
# the atomistic MLIP influences the hypothesis BEFORE vs AFTER active-learning
# fine-tuning.
#
# A panel of *different* models from the local model zoo is served, each on its
# own OpenAI-compatible vLLM endpoint (isolated vllm_venv, torch 2.11). A thin
# client (hydragnn_venv) drives the real matsim-agents debate machinery
# (_debate_hypothesis_response) twice -- once with the pretrained-MLIP evidence,
# once with the fine-tuned-MLIP evidence -- and records the two transcripts.
#
# Layout (single node, 4x A100 80GB):
#   GPU 0 :8000  proposer  Qwen2.5-14B-Instruct
#   GPU 1 :8001  critic    DeepSeek-R1-Distill-Qwen-32B
#   GPU 2 :8002  critic    gemma-4-31B-it
#   GPU 3        (spare)
#
# Smoke test (one small model as proposer+critic, ~10 min):
#   MATSIM_DEBATE_SMOKE=1 sbatch deployments/perlmutter/jobs/job-hypothesis-debate-perlmutter.sh
#
# Full run:
#   sbatch deployments/perlmutter/jobs/job-hypothesis-debate-perlmutter.sh
# ---------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
INSTALL_ROOT=$REPO/.hpc-build/perlmutter
VENV=$REPO/.venv          # debate client (matsim-agents + openai)
VLLM_VENV=$INSTALL_ROOT/vllm_venv         # isolated vLLM servers (torch 2.11)
MODELS_DIR=$PROJ/models
init_run_dirs "$PROJ" "hyp-debate" "${SLURM_JOB_ID:-$$}"

# One or more materials-characterization cases to debate (space-separated).
CASES=${MATSIM_DEBATE_CASES:-"cantor-fcc-al-001 hea-bcc-al-001"}
CASE_PRETTY=${MATSIM_DEBATE_CASE_PRETTY:-}   # empty => driver's registry name
RUNS_ROOT=${MATSIM_RUNS_ROOT:-$PROJ/runs/finetune-eval}

# ── panel definition: "GPU PORT MODEL_DIR" per server ────────────────────────
SMOKE=${MATSIM_DEBATE_SMOKE:-0}
if [[ "$SMOKE" == "1" ]]; then
  SERVERS=( "0 8000 ${MODELS_DIR}/Qwen2.5-14B-Instruct" )
  PROPOSER_PORT=8000; PROPOSER_MODEL=Qwen2.5-14B-Instruct
  CRITIC_PORTS=(8000); CRITIC_MODELS=(Qwen2.5-14B-Instruct)
else
  SERVERS=(
    "0 8000 ${MODELS_DIR}/Qwen2.5-14B-Instruct"
    "1 8001 ${MODELS_DIR}/DeepSeek-R1-Distill-Qwen-32B"
    "2 8002 ${MODELS_DIR}/gemma-4-31B-it"
  )
  PROPOSER_PORT=8000; PROPOSER_MODEL=Qwen2.5-14B-Instruct
  CRITIC_PORTS=(8001 8002); CRITIC_MODELS=(DeepSeek-R1-Distill-Qwen-32B gemma-4-31B-it)
fi

VLLM_MAXLEN=${MATSIM_VLLM_MAXLEN:-8192}
VLLM_GPU_UTIL=${MATSIM_VLLM_GPU_UTIL:-0.90}
declare -a VLLM_PIDS=()

# ── launch each vLLM server (background, isolated env) ───────────────────────
# Started BEFORE the system module stack so they use vLLM's bundled CUDA wheels.
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
    # vLLM JIT-compiles a CUDA helper at startup and needs Python.h; the base
    # python3.11 has no dev headers, so point at the hydragnn_venv headers.
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
  for _ in $(seq 1 300); do          # up to ~50 min (import + weights on CFS)
    local alive=0
    for pid in "${VLLM_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ "$alive" == "0" ]] && { echo "[ERROR] a vLLM server exited early" >&2; return 1; }
    # Detect a crashed engine core even if the wrapper process lingers.
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

# ── run the demonstration (client is CPU-only: keep it off the GPUs) ─────────
CRITIC_ARGS=()
for i in "${!CRITIC_MODELS[@]}"; do
  CRITIC_ARGS+=( --critic-model "${CRITIC_MODELS[$i]}" \
                 --critic-base-url "http://127.0.0.1:${CRITIC_PORTS[$i]}/v1" )
done

for CASE in $CASES; do
  CASE_ARGS=( --case "$CASE" )
  [[ -n "$CASE_PRETTY" ]] && CASE_ARGS+=( --case-pretty "$CASE_PRETTY" )
  CASE_OUT="$OUTPUT_DIR/$CASE"
  mkdir -p "$CASE_OUT"
  echo "[$(date)] running hypothesis-debate driver (case=$CASE) ..."
  CUDA_VISIBLE_DEVICES="" python "$REPO/deployments/perlmutter/jobs/hypothesis_debate_beforeafter.py" \
      "${CASE_ARGS[@]}" \
      --runs-root "$RUNS_ROOT" \
      --output-dir "$CASE_OUT" \
      --provider vllm \
      --proposer-model "$PROPOSER_MODEL" \
      --proposer-base-url "http://127.0.0.1:${PROPOSER_PORT}/v1" \
      "${CRITIC_ARGS[@]}" \
      --debate-rounds "${MATSIM_DEBATE_ROUNDS:-2}" \
      2>&1 | tee "$RUN_DIR/hypothesis-debate-${CASE}.log"
done

echo "[$(date)] done. Artifacts in $OUTPUT_DIR"
