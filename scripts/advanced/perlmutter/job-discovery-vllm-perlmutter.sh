#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J discovery-vllm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: discovery run on NERSC Perlmutter using a *local vLLM server*
# as the LLM backend (OpenAI-compatible /v1 endpoint).
#
# Layout (single node, 4xA100 80GB):
#   • vLLM server (Qwen2.5-14B)  : GPU 0           [vllm_venv, torch 2.11]
#   • matsim-agents discovery    : GPUs 1,2,3      [hydragnn_venv, torch 2.8]
#       - HydraGNN MLFF relaxations run inline on the 3 reserved GPUs
#       - LLM proposer + critic both talk to localhost:8000/v1
#
# The two stacks live in separate venvs because vLLM hard-pins torch 2.11
# while HydraGNN/UMA require torch 2.8. The matsim vLLM client only needs the
# `openai` package plus the HTTP endpoint, so no torch conflict at the client.
#
# Usage:
#   sbatch scripts/advanced/perlmutter/job-discovery-vllm-perlmutter.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=$PROJ/models/Qwen2.5-72B-Instruct MATSIM_VLLM_TP=2 \
#     sbatch scripts/advanced/perlmutter/job-discovery-vllm-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
source "${SCRIPT_DIR}/../common/runtime-env.sh"
REPO="$(resolve_repo_root "${SCRIPT_DIR}" "/global/cfs/projectdirs/m5216/mlupopa/matsim-agents")"
PROJ="$(dirname "${REPO}")"
INSTALL_ROOT=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV=$INSTALL_ROOT/hydragnn_venv          # discovery client + HydraGNN MLFF
VLLM_VENV=$INSTALL_ROOT/vllm_venv         # isolated vLLM server (torch 2.11)
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-14B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
init_run_dirs "$PROJ" "discovery-vllm" "${SLURM_JOB_ID:-$$}"

# ── vLLM server knobs ────────────────────────────────────────────────────────
VLLM_HOST=${MATSIM_VLLM_HOST:-127.0.0.1}
VLLM_PORT=${MATSIM_VLLM_PORT:-8000}
VLLM_TP=${MATSIM_VLLM_TP:-1}                       # tensor-parallel size (GPUs for server)
VLLM_GPUS=${MATSIM_VLLM_GPUS:-0}                   # CUDA ids for the server
VLLM_MAXLEN=${MATSIM_VLLM_MAXLEN:-8192}
VLLM_GPU_UTIL=${MATSIM_VLLM_GPU_UTIL:-0.90}
CLIENT_GPUS=${MATSIM_CLIENT_GPUS:-1,2,3}           # CUDA ids for MLFF relaxations
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# ── 1) launch vLLM server (background, isolated env, GPU 0) ───────────────────
# Launched BEFORE loading the system module stack so it uses vLLM's bundled
# CUDA wheels (cu13) rather than a conflicting system cudatoolkit.
echo "[$(date)] Starting vLLM server for $MODEL_NAME on GPU(s) $VLLM_GPUS (tp=$VLLM_TP) ..."
VLLM_LOG="$RUN_DIR/vllm-server.log"
(
  export CUDA_VISIBLE_DEVICES="$VLLM_GPUS"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export VLLM_DO_NOT_TRACK=1
  export PYTHONNOUSERSITE=1
  # FlashInfer's sampler JIT-compiles + takes an fcntl.flock in its cache dir;
  # CFS/GPFS/$HOME do not support flock -> OSError [Errno 524]. Use the native
  # torch sampler (no JIT, no lock), and pin any remaining caches to tmpfs.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  _JIT_TMP="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-$$}"
  mkdir -p "$_JIT_TMP"
  export FLASHINFER_WORKSPACE_BASE="$_JIT_TMP"
  export TRITON_CACHE_DIR="$_JIT_TMP/triton"
  export TORCHINDUCTOR_CACHE_DIR="$_JIT_TMP/inductor"
  export VLLM_CACHE_ROOT="$_JIT_TMP/vllm"
  _PY_HDR="$INSTALL_ROOT/hydragnn_venv/include/python3.11"
  export CPATH="${_PY_HDR}:${CPATH:-}"
  export C_INCLUDE_PATH="${_PY_HDR}:${C_INCLUDE_PATH:-}"
  exec "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
      --served-model-name "$MODEL_NAME" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --tensor-parallel-size "$VLLM_TP" \
      --max-model-len "$VLLM_MAXLEN" \
      --gpu-memory-utilization "$VLLM_GPU_UTIL" \
      --enforce-eager
) >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Ensure the server is torn down whenever the job exits (success or failure).
cleanup() {
  echo "[$(date)] Stopping vLLM server (pid $VLLM_PID) ..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── 2) modules & client venv (HydraGNN-aligned stack) ────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Point the matsim vLLM client at the local server.
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_VLLM_BASE_URL="$BASE_URL"
export MATSIM_VLLM_API_KEY=EMPTY

# ── 3) wait for the server's /v1/models endpoint to come up ──────────────────
echo "[$(date)] Waiting for vLLM endpoint $BASE_URL ..."
READY=0
for i in $(seq 1 240); do          # up to ~40 min (vLLM import+weights load is slow on CFS)
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM server process exited early. Tail of $VLLM_LOG:" >&2
    tail -n 40 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    READY=1
    echo "[$(date)] vLLM endpoint is ready after ~$((i*10))s."
    break
  fi
  sleep 10
done
if [[ "$READY" -ne 1 ]]; then
  echo "[ERROR] vLLM endpoint did not become ready in time. Tail of $VLLM_LOG:" >&2
  tail -n 40 "$VLLM_LOG" >&2 || true
  exit 1
fi

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "[$(date)] Client python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] LLM endpoint:  $BASE_URL  (model=$MODEL_NAME)"
CUDA_VISIBLE_DEVICES="$CLIENT_GPUS" python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# ── 4) discovery query ───────────────────────────────────────────────────────
QUERY="${MATSIM_DISCOVERY_QUERY:-}"
if [[ -z "$QUERY" ]]; then
  QUERY="$(cat <<'EOF'
Propose 3 to 5 candidate inorganic materials for high-temperature structural applications.
For each candidate, provide the formula, likely crystal family, and a brief physics-based
justification. Then relax the proposed structures with the MLFF and summarize relative
stability from final energies and residual forces.
EOF
)"
fi

# ── active-learning handoff mode ─────────────────────────────────────────────
# MATSIM_AL_MODE = dry-run (default; plan/report only) | run (execute AL loop).
# When MATSIM_AL_MODE=run, supply MATSIM_AL_CONFIG = path to a base AL YAML
# (e.g. examples/paper_cases/al_hea_bcc.yaml). Discovery overrides its
# seed_source to the detected formula on handoff.
AL_MODE=${MATSIM_AL_MODE:-dry-run}
AL_ARGS=()
if [[ "$AL_MODE" == "run" ]]; then
  AL_ARGS+=( --al-run )
  if [[ -n "${MATSIM_AL_CONFIG:-}" ]]; then
    AL_CFG="${MATSIM_AL_CONFIG}"
    [[ "$AL_CFG" = /* ]] || AL_CFG="$REPO/$AL_CFG"
    [[ -f "$AL_CFG" ]] || { echo "[ERROR] MATSIM_AL_CONFIG not found: $AL_CFG" >&2; exit 1; }
    AL_ARGS+=( --al-config "$AL_CFG" )
  fi
else
  AL_ARGS+=( --al-dry-run )
fi

echo "[$(date)] Submitting discovery query to matsim-agents (vLLM provider, AL mode=$AL_MODE) ..."
echo "$QUERY" | CUDA_VISIBLE_DEVICES="$CLIENT_GPUS" matsim-agents chat \
    --logdir          "$LOGDIR" \
    --hydragnn-branch-mlp-checkpoint "$HYDRAGNN_BRANCH_MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --llm-provider    vllm \
    --llm-model       "$MODEL_NAME" \
    --llm-base-url    "$BASE_URL" \
    --llm-peer-review \
    --peer-review-rounds "${MATSIM_PEER_REVIEW_ROUNDS:-2}" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    "${AL_ARGS[@]}" \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
