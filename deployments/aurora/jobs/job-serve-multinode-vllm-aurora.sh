#!/bin/bash
#PBS -N matsim-vllm-multinode
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Multi-node vLLM serve on ALCF Aurora (Intel PVC, XPU backend).
#
# Aurora analog of `deployments/frontier/jobs/job-serve-multinode-frontier.sh`:
# bootstraps a Ray cluster across all allocated nodes, then starts a vLLM
# server with tensor parallelism spanning every PVC tile across all nodes.
#
# Aurora geometry: 6 PVC GPUs × 2 tiles = 12 ranks/node.
#
# Prerequisite:
#   - vLLM XPU venv built via:
#       bash deployments/aurora/setup/install-vllm-xpu-aurora.sh
#
# Required env at submission:
#   SERVE_MODEL_PATH   – absolute path to local model directory
#
# Optional env:
#   SERVE_MODEL_NAME    – default: dir basename of SERVE_MODEL_PATH
#   SERVE_PORT          – vLLM HTTP port (default: 8000)
#   SERVE_TP_SIZE       – default: NNODES * 12 (one per PVC tile)
#   SERVE_DTYPE         – bfloat16 | float16 (default: bfloat16)
#   SERVE_MAX_MODEL_LEN – default: 32768
#   SERVE_EXTRA_ARGS    – verbatim extra args for `vllm serve`
#   SERVE_SMOKE=1       – after readiness, send one chat completion and exit
#                         (used by launchers/launch-test-multinode-aurora.sh)
#   RAY_PORT            – Ray head port (default: 6379)
#
# Submit (2 nodes, Mixtral-8x22B):
#   SERVE_MODEL_PATH=$PROJ/models/Mixtral-8x22B-Instruct-v0.1 \
#   qsub deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh
#
# Submit (4 nodes, override default select):
#   SERVE_MODEL_PATH=$PROJ/models/Llama-3.3-70B-Instruct \
#   qsub -l select=4 \
#        -v SERVE_MODEL_PATH=$PROJ/models/Llama-3.3-70B-Instruct \
#        deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh
#
# Unless SERVE_SMOKE=1, the server stays alive until walltime or manual
# cancel. Connect clients to:
#   http://<head_node_hostname>:${SERVE_PORT}/v1
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# vLLM is provided by the `frameworks` module (vLLM 0.15 + PyTorch 2.10/XPU as
# of frameworks/2025.3.1). We then activate the matsim-owned .venv (built with
# --system-site-packages on top of that same Python 3.12) so HydraGNN +
# matsim-agents are importable alongside vLLM.
VENV_PATH="${VENV_PATH:-${REPO}/.venv}"
JOBID="${PBS_JOBID:-local-$$}"
RUN_DIR="${PROJ}/runs/vllm-multinode-aurora-${JOBID}"
mkdir -p "$RUN_DIR"

# ── Validate inputs ─────────────────────────────────────────────────────────
if [[ -z "${SERVE_MODEL_PATH:-}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH is required." >&2
  echo "  SERVE_MODEL_PATH=/path/to/model qsub $0" >&2
  exit 2
fi
if [[ ! -d "${SERVE_MODEL_PATH}" ]]; then
  echo "ERROR: SERVE_MODEL_PATH does not exist: ${SERVE_MODEL_PATH}" >&2
  exit 2
fi

# ── Configuration ───────────────────────────────────────────────────────────
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_DTYPE="${SERVE_DTYPE:-bfloat16}"
SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-32768}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_NAME="${SERVE_MODEL_NAME:-$(basename "$SERVE_MODEL_PATH")}"

NNODES=$(wc -l < "$PBS_NODEFILE")
TILES_PER_NODE=12   # 6 PVC GPUs × 2 tiles
SERVE_TP_SIZE="${SERVE_TP_SIZE:-$(( NNODES * TILES_PER_NODE ))}"

# ── Modules ─────────────────────────────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
else
  echo "WARN: $VENV_PATH/bin/activate not found; using bare frameworks Python" >&2
fi
RAY="$(command -v ray)"
VLLM_BIN="$(command -v vllm)"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Compute nodes have no outbound internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM telemetry off
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray telemetry off
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# Aurora oneCCL / fabric tunings.
# NOTE: do NOT set CCL_KVS_MODE=mpi or CCL_PROCESS_LAUNCHER=pmix here. Those
# are valid only when ranks are MPI-launched (HydraGNN training pattern).
# vLLM's TP workers are spawned by Ray (multinode) or multiproc_executor
# (singlenode) — they are NOT MPI ranks, so oneCCL must use its default
# internal-KVS over TCP. Setting MPI mode triggers:
#   |CCL_ERROR| internal_kvs.cpp:42 kvs_set_value: condition
#   can_use_internal_kvs() failed
# and "WorkerProc initialization failed" (first seen in job 8508267,
# single-node variant).
export CCL_KVS_CONNECTION_TIMEOUT=900
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1

# Let Ray see all 12 tiles per node
unset ZE_AFFINITY_MASK
# IMPORTANT: do NOT override ONEAPI_DEVICE_SELECTOR — keep module's
# "opencl:gpu;level_zero:gpu" so Triton-XPU + vLLM are functional.

# ── Discover nodes ──────────────────────────────────────────────────────────
mapfile -t ALL_NODES < "$PBS_NODEFILE"
HEAD_NODE="${ALL_NODES[0]}"
# Resolve head node IP from this rank-0 node (we are running on the head).
HEAD_NODE_IP="$(hostname -I | awk '{print $1}')"
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "=========================================="
echo "Multi-node vLLM-XPU serve (Aurora)"
echo "Date:           $(date)"
echo "Job ID:         $JOBID"
echo "Nodes:          $NNODES   (head=$HEAD_NODE  ip=$HEAD_NODE_IP)"
echo "Workers:        ${ALL_NODES[*]:1}"
echo "Model:          $MODEL_NAME"
echo "Path:           $SERVE_MODEL_PATH"
echo "TP size:        $SERVE_TP_SIZE  (= $NNODES nodes × $TILES_PER_NODE tiles)"
echo "dtype:          $SERVE_DTYPE"
echo "max_model_len:  $SERVE_MAX_MODEL_LEN"
echo "Port:           $SERVE_PORT"
echo "Run dir:        $RUN_DIR"
echo "=========================================="

# ── Cleanup trap ────────────────────────────────────────────────────────────
VLLM_PID=""
RAY_HEAD_PID=""
WORKER_PIDS=()
# Stop processes with SIGTERM, wait up to $1 seconds for all of them, then kill the rest.
stop_processes() {
  local deadline=$(( SECONDS + $1 )) pid
  shift
  (( $# )) || return 0
  kill -TERM "$@" 2>/dev/null || true
  for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
# Workers on other nodes stop when RAY_STOP appears (ray-node.sh): signalling
# their launcher would kill them before they save their logs.
stop_workers() {
  (( ${#WORKER_PIDS[@]} )) || return 0
  [[ -z "${RAY_STOP:-}" ]] || touch "$RAY_STOP"
  local deadline=$(( SECONDS + 40 )) pid
  for pid in "${WORKER_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  done
  stop_processes 5 "${WORKER_PIDS[@]}"
  [[ -z "${RAY_STOP:-}" ]] || rm -f "$RAY_STOP"
}
cleanup() {
  echo
  echo "[cleanup] Stopping vLLM and Ray cluster ..."
  [[ -n "$VLLM_PID" ]]      && kill "$VLLM_PID"      2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  # Workers leave while the head still runs, then the head stops; each node
  # saves its Ray logs as it stops (ray-node.sh).
  stop_workers
  stop_processes 20 $RAY_HEAD_PID
  echo "[cleanup] Done."
}
trap cleanup EXIT

# ── Start Ray head on this node ─────────────────────────────────────────────
# Every Ray node runs under ray-node.sh: its session directory is short,
# node-local scratch (Ray's Unix sockets have a path-length limit), and its
# logs are copied to RAY_LOGS/<host> when the node stops.
RAY_NODE="$REPO/deployments/common/ray-node.sh"
RAY_TMP="/tmp/r${JOBID%%.*}.${RAY_PORT}"
RAY_LOGS="$RUN_DIR/ray-logs"
RAY_STOP="$RAY_LOGS/.stop.$$"
mkdir -p "$RAY_LOGS"
rm -f "$RAY_STOP"

echo "[ray] Starting head at $RAY_ADDRESS ..."
"$RAY_NODE" "$RAY_LOGS" "$RAY" start --head \
    --node-ip-address="$HEAD_NODE_IP" \
    --port="$RAY_PORT" \
    --temp-dir="$RAY_TMP" \
    --num-cpus=104 \
    --num-gpus="$TILES_PER_NODE" \
    --block \
  > "$RUN_DIR/ray-head.log" 2>&1 &
RAY_HEAD_PID=$!
sleep 10

# ── Start Ray workers on remaining nodes ────────────────────────────────────
for node in "${ALL_NODES[@]:1}"; do
  echo "[ray] Starting worker on $node ..."
  mpiexec -n 1 --ppn 1 --hosts "$node" \
      bash -c "
        module reset >/dev/null 2>&1 || true
        module load frameworks
        [[ -f '$VENV_PATH/bin/activate' ]] && source '$VENV_PATH/bin/activate'
        export RAY_USAGE_STATS_ENABLED=0
        unset ZE_AFFINITY_MASK
        # Keep ONEAPI_DEVICE_SELECTOR as set by the frameworks module.
        RAY_NODE_STOP_FILE='$RAY_STOP' '$RAY_NODE' '$RAY_LOGS' ray start --address='$RAY_ADDRESS' \
             --temp-dir='$RAY_TMP' \
             --num-cpus=104 \
             --num-gpus=$TILES_PER_NODE \
             --block
      " > "$RUN_DIR/ray-worker-$node.log" 2>&1 &
  WORKER_PIDS+=($!)
done
sleep 20

echo "[ray] Cluster status:"
"$RAY" status --address="$RAY_ADDRESS" || true
echo

# ── Start vLLM ──────────────────────────────────────────────────────────────
echo "[vllm] Starting server TP=${SERVE_TP_SIZE} on port ${SERVE_PORT} ..."
"$VLLM_BIN" serve "$SERVE_MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$SERVE_TP_SIZE" \
    --distributed-executor-backend ray \
    --dtype "$SERVE_DTYPE" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --port "$SERVE_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --no-enable-log-requests \
    --enforce-eager \
    ${SERVE_EXTRA_ARGS:-} \
  > "$RUN_DIR/vllm-serve.log" 2>&1 &
VLLM_PID=$!

echo "[vllm] PID=$VLLM_PID, waiting for /health ..."

# ── Wait for vLLM /health ───────────────────────────────────────────────────
MAX_WAIT=900  # up to 15 min to load very large models
ELAPSED=0
INTERVAL=10
while true; do
  if curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; then
    echo "[vllm] Server ready after ${ELAPSED}s."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vllm] ERROR: server process died." >&2
    tail -80 "$RUN_DIR/vllm-serve.log" >&2
    exit 1
  fi
  if (( ELAPSED >= MAX_WAIT )); then
    echo "[vllm] ERROR: server did not become ready within ${MAX_WAIT}s." >&2
    tail -80 "$RUN_DIR/vllm-serve.log" >&2
    exit 1
  fi
  sleep $INTERVAL
  (( ELAPSED += INTERVAL ))
done

echo
echo "=========================================="
echo "vLLM server READY"
echo "  HEAD NODE:   $HEAD_NODE  ($HEAD_NODE_IP)"
echo "  BASE URL:    http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo "  MODEL NAME:  $MODEL_NAME"
echo "  TP SIZE:     $SERVE_TP_SIZE  ($NNODES nodes × $TILES_PER_NODE tiles)"
echo "  JOB ID:      $JOBID"
echo "=========================================="
echo
echo "Connect a client (e.g. matsim-agents) with:"
echo "  export MATSIM_LLM_PROVIDER=vllm"
echo "  export MATSIM_VLLM_BASE_URL=http://${HEAD_NODE_IP}:${SERVE_PORT}/v1"
echo "  export MATSIM_VLLM_API_KEY=EMPTY"
echo

if [[ "${SERVE_SMOKE:-0}" == "1" ]]; then
  echo "[smoke] Sending one chat-completion request ..."
  curl -sf -X POST "http://localhost:${SERVE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\":\"user\",\"content\":\"What is 2+2? Reply with only the number.\"}], \"max_tokens\": 16, \"temperature\": 0.0}" \
    | tee "$RUN_DIR/response.json"
  echo
  echo "[OK] vLLM-XPU multi-node smoke test PASSED."
  exit 0
fi

echo "[serve] Server running. Waiting for walltime or cancellation ..."
wait "$VLLM_PID"
echo "[serve] vLLM process exited."
