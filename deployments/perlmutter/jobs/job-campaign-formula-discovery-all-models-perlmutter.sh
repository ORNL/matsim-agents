#!/bin/bash
#SBATCH -J campaign-formula-e2e-all
#SBATCH -N 16
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ---------------------------------------------------------------------------
# Full local-model-panel variant of job-campaign-formula-discovery-perlmutter.sh:
# serves the same ~7-model local catalog panel as
# job-all-local-model-debate-perlmutter.sh (kimi-k2.5, glm-4.7, glm-4.7-flash,
# qwen3-235b-a22b-instruct-2507, qwen3-235b-a22b-thinking-2507, gemma-4-31b-it,
# gemma-4-26b-a4b-it -- deepseek-v3.2 and devstral-2 excluded, see NAMES below),
# then runs an initial formula-discovery debate followed by a bounded,
# resumable campaign on the 16th node: UMA phase exploration, UMA-driven MD,
# QE DFT labelling, and repeated seven-model evidence reviews.
#
# Required at submission (Slurm spools this script, so it cannot self-locate
# the checkout):
#   PROJECT_ROOT   matsim-agents checkout
# Optional retraining controls:
#   MATSIM_CAMPAIGN_RETRAIN=1              train a candidate UMA checkpoint
#   MATSIM_CAMPAIGN_TRAIN_EPOCHS=5         fine-tuning epochs per formula
#   MATSIM_CAMPAIGN_PROMOTE_MODEL=1        approve promotion and reevaluation
# Optional DFT hull controls:
#   MATSIM_CAMPAIGN_DFT_REFINE=0            disable DFT relaxation/hull ranking
#   MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES=1 refined phases per formula
#
# Submit with:
#   PROJECT_ROOT=$PWD sbatch -A m5216_g -q premium \
#     deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail
export MATSIM_CAMPAIGN_MAX_DFT="${MATSIM_CAMPAIGN_MAX_DFT:-12}"

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
PROJ="$(dirname "$REPO")"
MODELS_ROOT="${MODEL_ROOT:-$PROJ/models}"
SERVER="$REPO/deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh"
CAMPAIGN_RUN_TAG="${MATSIM_CAMPAIGN_RUN_TAG:-campaign-formula-e2e-all}"
OUTPUT="${RUNS_ROOT:-$PROJ/runs}/portability/${CAMPAIGN_RUN_TAG}-${SLURM_JOB_ID}"
PYTHON="${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python3"
AL_CONFIG="${MATSIM_CAMPAIGN_AL_CONFIG:-$REPO/deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-qe.yaml}"
REFERENCE_BACKEND="${MATSIM_CAMPAIGN_REFERENCE_BACKEND:-qe}"
DFT_METHOD_SIGNATURE="${MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE:-qe-pbe-pslibrary-k4-o2-triplet-gamma-v1}"
[[ -f "$AL_CONFIG" ]] || { echo "ERROR: campaign AL config not found: $AL_CONFIG" >&2; exit 2; }
[[ "$REFERENCE_BACKEND" == "qe" || "$REFERENCE_BACKEND" == "vasp" ]] || {
  echo "ERROR: MATSIM_CAMPAIGN_REFERENCE_BACKEND must be qe or vasp" >&2
  exit 2
}

# same exclusions as job-all-local-model-debate-perlmutter.sh: deepseek-v3.2 and
# devstral-2 both crash vLLM on A100 (jobs 58627251 and the sm80 DeepSeek case).
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

mkdir -p "$OUTPUT/servers"
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
EXPECTED_NODES=16
(( ${#ALL_NODES[@]} == EXPECTED_NODES )) || {
  echo "ERROR: expected $EXPECTED_NODES nodes (15 model-serving + 1 campaign compute)" >&2
  exit 2
}
CAMPAIGN_NODE="${ALL_NODES[15]}"

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

# ── build one repeatable --model NAME PROVIDER MODEL BASE_URL per endpoint ──
MODEL_ARGS=()
for index in "${!NAMES[@]}"; do
  MODEL_ARGS+=(--model "${NAMES[$index]}" vllm "${MODEL_IDS[$index]}" "http://${SERVER_IPS[$index]}:8000/v1")
done

echo "[$(date)] Running campaign formula-discovery driver against $EXPECTED_MODELS local models ..."
"$PYTHON" "$REPO/deployments/perlmutter/jobs/campaign_formula_discovery.py" \
  --elements Nb Ta O \
  --oxidation-state Nb:3,4,5 --oxidation-state Ta:3,4,5 --oxidation-state O:-2 \
  --max-coefficient 6 --max-atoms 12 \
  --rounds "${MATSIM_CAMPAIGN_ROUNDS:-2}" \
  --campaign-id "nb-ta-o-e2e-all-${SLURM_JOB_ID}" \
  --output-dir "$OUTPUT/campaign" --output-root "$OUTPUT" \
  "${MODEL_ARGS[@]}"

printf -v MODEL_ARGS_QUOTED '%q ' "${MODEL_ARGS[@]}"
TRAIN_ARGS=()
if [[ "${MATSIM_CAMPAIGN_RETRAIN:-0}" == "1" ]]; then
  TRAIN_ARGS=(
    --retrain
    --approve-retraining
    --train-script "$REPO/src/matsim_agents/active_learning/finetune_uma.py"
    --train-epochs "${MATSIM_CAMPAIGN_TRAIN_EPOCHS:-5}"
  )
  if [[ -n "${MATSIM_CAMPAIGN_TRAIN_LAUNCHER:-}" ]]; then
    TRAIN_ARGS+=(--train-launcher "$MATSIM_CAMPAIGN_TRAIN_LAUNCHER")
  fi
  if [[ "${MATSIM_CAMPAIGN_PROMOTE_MODEL:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--promote-model --approve-model-promotion)
  fi
fi
TRAIN_ARGS_QUOTED=""
if (( ${#TRAIN_ARGS[@]} > 0 )); then
  printf -v TRAIN_ARGS_QUOTED '%q ' "${TRAIN_ARGS[@]}"
fi
DFT_REFINEMENT_ARGS=()
if [[ "${MATSIM_CAMPAIGN_DFT_REFINE:-1}" == "1" ]]; then
  REFERENCE_DIR="$OUTPUT/campaign/dft-references"
  "$PYTHON" "$REPO/deployments/perlmutter/jobs/prepare_nb_ta_o_references.py" \
    --output-dir "$REFERENCE_DIR" --backend "$REFERENCE_BACKEND"
  DFT_REFINEMENT_ARGS=(
    --dft-reference-structures "$REFERENCE_DIR/reference_structures.json"
    --dft-method-signature "$DFT_METHOD_SIGNATURE"
    --dft-refine-candidates "${MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES:-1}"
    --dft-relax-max-steps "${MATSIM_CAMPAIGN_DFT_RELAX_STEPS:-100}"
    --dft-force-tolerance "${MATSIM_CAMPAIGN_DFT_FORCE_TOLERANCE:-0.02}"
  )
fi
DFT_REFINEMENT_ARGS_QUOTED=""
if (( ${#DFT_REFINEMENT_ARGS[@]} > 0 )); then
  printf -v DFT_REFINEMENT_ARGS_QUOTED '%q ' "${DFT_REFINEMENT_ARGS[@]}"
fi
TRAIN_ARGS_QUOTED="$DFT_REFINEMENT_ARGS_QUOTED$TRAIN_ARGS_QUOTED"
echo "[$(date)] Running bounded UMA/QE campaign on $CAMPAIGN_NODE ..."
srun --nodes=1 --ntasks=1 --nodelist="$CAMPAIGN_NODE" --overlap \
  --gpus-per-node=1 --cpus-per-task=16 \
  bash -c "
    set -euo pipefail
    source '$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh'
    load_perlmutter_modules_gpu
    source '$REPO/.venv-uma/bin/activate'
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-1}\" TRANSFORMERS_OFFLINE=\"\${TRANSFORMERS_OFFLINE:-1}\"
    export PROJECT_ROOT='$REPO' MATSIM_GPUS_PER_NODE=4
    export SLURM_JOB_NODELIST='$CAMPAIGN_NODE' SLURM_JOB_NUM_NODES=1
    cd '$REPO'
    python3 deployments/perlmutter/jobs/campaign_execute.py \
      --campaign-state '$OUTPUT/campaign/campaign_state.json' \
      --al-config '$AL_CONFIG' \
      --review-rounds \"\${MATSIM_CAMPAIGN_REVIEW_ROUNDS:-2}\" \
      --minimum-review-agreement \"\${MATSIM_CAMPAIGN_REVIEW_AGREEMENT:-1.0}\" \
      --formulas-per-iteration \"\${MATSIM_CAMPAIGN_FORMULAS_PER_ITERATION:-1}\" \
      --max-iterations \"\${MATSIM_CAMPAIGN_MAX_ITERATIONS:-3}\" \
      --max-candidates \"\${MATSIM_CAMPAIGN_MAX_CANDIDATES:-3}\" \
      --max-dft-calculations \"\${MATSIM_CAMPAIGN_MAX_DFT:-6}\" \
      --max-al-iterations \"\${MATSIM_CAMPAIGN_MAX_AL_ITERATIONS:-3}\" \
      --max-node-hours \"\${MATSIM_CAMPAIGN_MAX_NODE_HOURS:-8}\" \
      --n-random \"\${MATSIM_CAMPAIGN_RANDOM_SEEDS:-0}\" \
      --relax-maxiter \"\${MATSIM_CAMPAIGN_RELAX_MAXITER:-100}\" \
      --approve-dft \
      $TRAIN_ARGS_QUOTED \
      $MODEL_ARGS_QUOTED
  " 2>&1 | tee "$OUTPUT/campaign-execute.log"

echo "[$(date)] campaign complete. Artifacts in $OUTPUT/campaign"
