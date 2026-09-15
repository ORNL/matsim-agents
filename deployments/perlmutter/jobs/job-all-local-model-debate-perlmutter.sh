#!/bin/bash
#SBATCH -J local-llm-debate
#SBATCH -N 21
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PROJ="$(dirname "$REPO")"
MODELS_ROOT="${MODEL_ROOT:-$PROJ/models}"
SERVER="$REPO/deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh"
OUTPUT="${RUNS_ROOT:-$PROJ/runs}/portability/local-llm-debate-${SLURM_JOB_ID}"
PYTHON="${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3"

NAMES=(
  kimi-k2.5 glm-4.7 glm-4.7-flash deepseek-v3.2
  qwen3-235b-a22b-instruct-2507 qwen3-235b-a22b-thinking-2507
  devstral-2 gemma-4-31b-it gemma-4-26b-a4b-it
)
MODEL_IDS=(
  moonshotai/Kimi-K2.5 zai-org/GLM-4.7 zai-org/GLM-4.7-Flash
  deepseek-ai/DeepSeek-V3.2 Qwen/Qwen3-235B-A22B-Instruct-2507
  Qwen/Qwen3-235B-A22B-Thinking-2507 mistralai/Devstral-2-123B-Instruct-2512
  google/gemma-4-31B-it google/gemma-4-26B-A4B-it
)
MODEL_DIRS=(
  Kimi-K2.5 GLM-4.7 GLM-4.7-Flash DeepSeek-V3.2
  Qwen3-235B-A22B-Instruct-2507 Qwen3-235B-A22B-Thinking-2507
  Devstral-2-123B-Instruct-2512 gemma-4-31B-it gemma-4-26B-A4B-it
)
NODE_COUNTS=(4 4 1 4 2 2 2 1 1)
URL_VARS=(
  MATSIM_VLLM_KIMI_K25_BASE_URL MATSIM_VLLM_GLM47_BASE_URL
  MATSIM_VLLM_GLM47_FLASH_BASE_URL MATSIM_VLLM_DEEPSEEK_V32_BASE_URL
  MATSIM_VLLM_QWEN3_235B_INSTRUCT_BASE_URL
  MATSIM_VLLM_QWEN3_235B_THINKING_BASE_URL MATSIM_VLLM_DEVSTRAL2_BASE_URL
  MATSIM_VLLM_GEMMA4_31B_BASE_URL MATSIM_VLLM_GEMMA4_26B_BASE_URL
)

mkdir -p "$OUTPUT/servers"
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#ALL_NODES[@]} == 21 )) || { echo "ERROR: expected 21 nodes" >&2; exit 2; }

SERVER_PIDS=()
SERVER_IPS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_for_cluster_bootstrap() {
  local name=$1 nodes=$2 pid=$3 log=$4
  local expected=$(( nodes * 4 )) deadline=$(( SECONDS + 900 ))
  while (( SECONDS < deadline )); do
    if grep -Fq "[ray] $expected/$expected GPUs available" "$log" 2>/dev/null; then
      echo "[$(date)] $name Ray cluster has all $expected GPUs"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name launcher exited during Ray bootstrap" >&2
      tail -80 "$log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "ERROR: $name Ray cluster did not form within 900s" >&2
  tail -80 "$log" >&2 || true
  return 1
}

offset=0
for index in "${!NAMES[@]}"; do
  name=${NAMES[$index]}
  nodes=${NODE_COUNTS[$index]}
  group=("${ALL_NODES[@]:offset:nodes}")
  head=${group[0]}
  nodelist=$(IFS=,; echo "${group[*]}")
  model_path="$MODELS_ROOT/${MODEL_DIRS[$index]}"
  [[ -d "$model_path" ]] || { echo "ERROR: missing local model: $model_path" >&2; exit 2; }

  head_ip=$(srun --nodes=1 --ntasks=1 --nodelist="$head" hostname -I | awk '{print $1}')
  SERVER_IPS+=("$head_ip")
  export "${URL_VARS[$index]}=http://${head_ip}:8000/v1"

  echo "[$(date)] $name -> $nodelist ($head_ip)"
  server_log="$OUTPUT/servers/$name.log"
  srun --nodes=1 --ntasks=1 --nodelist="$head" --exclusive \
    --gpus-per-node=4 --cpus-per-task=64 \
    env PROJECT_ROOT="$REPO" SERVE_MODEL_PATH="$model_path" \
      SERVE_MODEL_NAME="${MODEL_IDS[$index]}" SERVE_N_NODES="$nodes" \
      SERVE_NODELIST="$nodelist" SERVE_RUN_PREFIX="vllm-coordinated-$name" \
      "$SERVER" >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
  if (( nodes > 1 )); then
    wait_for_cluster_bootstrap "$name" "$nodes" "${SERVER_PIDS[-1]}" "$server_log" || exit 1
  fi
  (( offset += nodes ))
done

echo "[$(date)] Waiting for all nine local model endpoints ..."
deadline=$(( SECONDS + 7200 ))
while true; do
  ready=0
  for index in "${!SERVER_IPS[@]}"; do
    if curl -sf "http://${SERVER_IPS[$index]}:8000/health" >/dev/null 2>&1; then
      (( ready += 1 ))
    elif ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
      echo "ERROR: ${NAMES[$index]} server launcher exited" >&2
      tail -80 "$OUTPUT/servers/${NAMES[$index]}.log" >&2 || true
      exit 1
    fi
  done
  echo "[$(date)] Ready endpoints: $ready/9"
  (( ready == 9 )) && break
  (( SECONDS < deadline )) || { echo "ERROR: endpoint readiness timed out" >&2; exit 1; }
  sleep 30
done

echo "[$(date)] Running local-model scientific debate ..."
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  "$REPO/benchmarks/portability/all_model_scientific_debate.py" \
  --rounds "${MATSIM_DEBATE_ROUNDS:-2}" \
  --models-root "$MODELS_ROOT" \
  --output "$OUTPUT/debate"

RESULT="$OUTPUT/debate/all_model_scientific_debate_result.json"
DIALOGUE=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["dialogue_path"])' "$RESULT")
echo "[$(date)] Debate complete"
echo "Result: $RESULT"
echo "Complete discussion with contribution IDs: $DIALOGUE"