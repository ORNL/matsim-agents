#!/bin/bash
#SBATCH -J al-debate-portability
#SBATCH -N 16
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# matsim-agents: real active-learning loop (UMA MD sampling + QE DFT labelling
# of the benchmark Si cell) followed by the full first-class local-model
# debate, grounded in that loop's labelled evidence instead of a synthetic
# scenario (benchmarks/portability/active_learning_scientific_debate.py).
#
# 15 nodes serve the same 7-model catalog as
# job-all-local-model-debate-perlmutter.sh (devstral-2 and deepseek-v3.2
# excluded: FP8/MLA kernels incompatible with Ampere A100). The 16th,
# dedicated node runs the active-learning loop (.venv-uma) once every
# endpoint is live, then drives the debate against them.
#
# Required at submission:
#   PROJECT_ROOT   matsim-agents checkout
#
# Submit with:
#   PROJECT_ROOT=$PWD sbatch -A <allocation> -q premium \
#     deployments/perlmutter/jobs/job-al-debate-portability-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
PROJ="$(dirname "$REPO")"
MODELS_ROOT="${MODEL_ROOT:-$PROJ/models}"
SERVER="$REPO/deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh"
OUTPUT="${RUNS_ROOT:-$PROJ/runs}/portability/al-debate-${SLURM_JOB_ID}"
AL_CONFIG="$REPO/benchmarks/portability/config/active_learning/al-si-uma-qe.yaml"

# Same 7-model catalog and exclusions as job-all-local-model-debate-perlmutter.sh
# (devstral-2: FP8 CUTLASS sm80 crash; deepseek-v3.2: sm80 MLA kernel gap).
NAMES=(
  kimi-k2.5 glm-4.7 glm-4.7-flash
  qwen3-235b-a22b-instruct-2507 qwen3-235b-a22b-thinking-2507
  gemma-4-31b-it gemma-4-26b-a4b-it
)
MODEL_IDS=(
  moonshotai/Kimi-K2.5 zai-org/GLM-4.7 zai-org/GLM-4.7-Flash
  Qwen/Qwen3-235B-A22B-Instruct-2507
  Qwen/Qwen3-235B-A22B-Thinking-2507
  google/gemma-4-31B-it google/gemma-4-26B-A4B-it
)
MODEL_DIRS=(
  Kimi-K2.5 GLM-4.7 GLM-4.7-Flash
  Qwen3-235B-A22B-Instruct-2507 Qwen3-235B-A22B-Thinking-2507
  gemma-4-31B-it gemma-4-26B-A4B-it
)
NODE_COUNTS=(4 4 1 2 2 1 1)
URL_VARS=(
  MATSIM_VLLM_KIMI_K25_BASE_URL MATSIM_VLLM_GLM47_BASE_URL
  MATSIM_VLLM_GLM47_FLASH_BASE_URL
  MATSIM_VLLM_QWEN3_235B_INSTRUCT_BASE_URL
  MATSIM_VLLM_QWEN3_235B_THINKING_BASE_URL
  MATSIM_VLLM_GEMMA4_31B_BASE_URL MATSIM_VLLM_GEMMA4_26B_BASE_URL
)

mkdir -p "$OUTPUT/servers"
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
EXPECTED_NODES=16
(( ${#ALL_NODES[@]} == EXPECTED_NODES )) || {
  echo "ERROR: expected $EXPECTED_NODES nodes (15 for the model catalog + 1 for the AL loop)" >&2
  exit 2
}
AL_NODE="${ALL_NODES[15]}"

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

EXPECTED_MODELS=${#NAMES[@]}
echo "[$(date)] Waiting for all $EXPECTED_MODELS compatible local model endpoints ..."
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
  echo "[$(date)] Ready endpoints: $ready/$EXPECTED_MODELS"
  (( ready == EXPECTED_MODELS )) && break
  (( SECONDS < deadline )) || { echo "ERROR: endpoint readiness timed out" >&2; exit 1; }
  sleep 30
done

echo "[$(date)] Running active-learning loop (UMA MD + QE DFT) + grounded debate on $AL_NODE ..."
mkdir -p "$OUTPUT/al-debate"
srun --nodes=1 --ntasks=1 --nodelist="$AL_NODE" --exclusive \
  --gpus-per-node=4 --cpus-per-task=64 \
  bash -c "
    set -euo pipefail
    source '$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh'
    load_perlmutter_modules_gpu
    source '$REPO/.venv-uma/bin/activate'
    source '$REPO/deployments/perlmutter/setup/model-artifacts-perlmutter.sh'
    configure_uma_model_artifacts '$REPO'
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-1}\" TRANSFORMERS_OFFLINE=\"\${TRANSFORMERS_OFFLINE:-1}\"
    if [[ -z \"\${HF_TOKEN:-}\" && -f \"\${HOME}/.cache/huggingface/token\" ]]; then
      export HF_TOKEN=\"\$(< \"\${HOME}/.cache/huggingface/token\")\"
    fi
    cd '$REPO'
    python3 benchmarks/portability/active_learning_scientific_debate.py \
      --al-config '$AL_CONFIG' \
      --rounds \"\${MATSIM_DEBATE_ROUNDS:-2}\" \
      --models-root '$MODELS_ROOT' \
      --exclude-model deepseek-v3.2 \
      --exclude-model devstral-2 \
      --output '$OUTPUT/al-debate'
  " 2>&1 | tee "$OUTPUT/al-debate.log"

RESULT="$OUTPUT/al-debate/active_learning_scientific_debate_result.json"
echo "[$(date)] Combined active-learning + debate qualification complete"
echo "Result: $RESULT"
