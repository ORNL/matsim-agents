#!/bin/bash
# Run `vllm serve` from the isolated vLLM environment on a Perlmutter GPU node.
#
# matsim-agents starts this for every managed-vllm server whose `launcher` names
# it (see deployments/perlmutter/resources/*.yaml) and appends the model and the
# server arguments derived from the placement; the device mask is already set.
# A server placed on several nodes is started on its first node, which runs the
# Ray head, starts Ray workers on the other nodes with `srun`, waits until every
# GPU has joined, and serves with --distributed-executor-backend ray.
#
#   bash vllm-server-perlmutter.sh <model> --served-model-name ... --port ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VLLM_VENV="${MATSIM_VLLM_VENV:-$REPO/venv_vllm}"
[[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "vllm not found in $VLLM_VENV" >&2; exit 2; }

# The driver runs from the client environment; keep its Python settings out of vLLM.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export PYTHONNOUSERSITE=1 VLLM_USE_FLASHINFER_SAMPLER=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
# JIT caches on node-local storage: CFS does not support the locking they need.
JIT_TMP="${TMPDIR:-/tmp}/vllm-jit.${USER}.${SLURM_JOB_ID:-0}.$$"
mkdir -p "$JIT_TMP"
export FLASHINFER_WORKSPACE_BASE="$JIT_TMP"
export TRITON_CACHE_DIR="$JIT_TMP/triton"
export TORCHINDUCTOR_CACHE_DIR="$JIT_TMP/inductor"
export VLLM_CACHE_ROOT="$JIT_TMP/vllm"
export XDG_CACHE_HOME="$JIT_TMP/xdg-cache" HF_HOME="$JIT_TMP/hf-home" TORCH_HOME="$JIT_TMP/torch-home"
export TMPDIR="$JIT_TMP/tmp"
mkdir -p "$TMPDIR"
# FlashInfer JIT builds need the Python headers of the client environment.
PY_HDR="$REPO/.venv/include/python3.11"
export CPATH="${PY_HDR}:${CPATH:-}"
export C_INCLUDE_PATH="${PY_HDR}:${C_INCLUDE_PATH:-}"

echo "[vllm-server] $(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

# MATSIM_SERVER_SLICES lists the server's nodes as "host=devices;host=devices".
# One node: serve directly. Several nodes: this is the head node; start a Ray
# cluster over all of them first, as job-serve-multinode-perlmutter.sh does.
IFS=';' read -r -a SLICES <<<"${MATSIM_SERVER_SLICES:-}"
if (( ${#SLICES[@]} <= 1 )); then
  exec "$VLLM_VENV/bin/vllm" serve "$@"
fi

export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_IMPORT_WARNING=1
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}" NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
RAY_PORT="${MATSIM_SERVER_RAY_PORT:?matsim-agents sets MATSIM_SERVER_RAY_PORT}"
NODE_GPUS="${MATSIM_NODE_DEVICES:-4}"
NODE_CPUS="${MATSIM_NODE_CORES:-64}"
# A worker step cannot take more CPUs than the job has on the node.
STEP_CPUS=$(( ${SLURM_CPUS_ON_NODE:-$NODE_CPUS} < NODE_CPUS ? ${SLURM_CPUS_ON_NODE:-$NODE_CPUS} : NODE_CPUS ))
HEAD_IP=$(hostname -I | awk '{print $1}')
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
# Every Ray node runs under ray-node.sh: its session directory is node-local
# scratch (a short path, for Ray's Unix sockets), and its logs are copied to
# RAY_LOGS/<host> on shared storage when it stops.
RAY_NODE="$REPO/deployments/common/ray-node.sh"
RAY_TMP="/tmp/r${SLURM_JOB_ID:-0}.${RAY_PORT}"
RAY_LOGS="${MATSIM_SERVER_LOG_DIR:?matsim-agents sets MATSIM_SERVER_LOG_DIR}"
# Workers stop when this file appears (ray-node.sh): signalling their srun would
# make Slurm kill the step before the worker saves its logs.
RAY_STOP="$RAY_LOGS/.stop.${SLURM_JOB_ID:-0}.$$"
mkdir -p "$RAY_LOGS"
rm -f "$RAY_STOP"

count() { local IFS=,; set -- $1; echo $#; }
EXPECTED=0
for slice in "${SLICES[@]}"; do EXPECTED=$(( EXPECTED + $(count "${slice#*=}") )); done
HEAD_GPUS=$(count "${SLICES[0]#*=}")

ray_gpus() {
  timeout 15s "$VLLM_VENV/bin/python" -c \
    'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0))); ray.shutdown()' \
    2>/dev/null | tail -1 || echo 0
}
wait_for_gpus() {
  local expected=$1 deadline=$(( SECONDS + $2 )) total=0
  while (( SECONDS < deadline )); do
    total=$(ray_gpus); [[ "$total" =~ ^[0-9]+$ ]] || total=0
    if (( total >= expected )); then
      echo "[ray] ${total}/${expected} GPUs available"
      return 0
    fi
    sleep 5
  done
  echo "[ray] ERROR: only ${total}/${expected} GPUs after $2 s" >&2
  return 1
}

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
# Wait up to $1 seconds for processes to exit on their own.
wait_for_exit() {
  local deadline=$(( SECONDS + $1 )) pid
  shift
  for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  done
}
VLLM_PID= HEAD_PID=
WORKER_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo "[vllm-server] stopping vLLM and the Ray cluster"
  # vLLM first; then the workers while the head still runs, so they leave the
  # cluster at once and save their logs; then the head.
  stop_processes 20 $VLLM_PID
  touch "$RAY_STOP"
  wait_for_exit 40 "${WORKER_PIDS[@]}"
  stop_processes 5 "${WORKER_PIDS[@]}"
  stop_processes 20 $HEAD_PID
  rm -f "$RAY_STOP"
}
# A stop request (signal or stop file) exits, and the EXIT trap cleans up once:
# returning from a signal trap would resume the bootstrap after cleanup.
trap cleanup EXIT
trap 'exit 143' INT TERM
# matsim-agents asks a multi-node server to stop by creating this file: when the
# launcher runs in an srun step, a signal would make Slurm kill the step before
# cleanup() has stopped the workers and saved the logs.
if [[ -n "${MATSIM_SERVER_STOP_FILE:-}" ]]; then
  ( while [[ ! -e "$MATSIM_SERVER_STOP_FILE" ]] && kill -0 $$ 2>/dev/null; do sleep 2; done
    kill -TERM $$ 2>/dev/null ) &
fi

echo "[ray] head ${HEAD_IP}:${RAY_PORT} with ${HEAD_GPUS} GPU(s); ${#SLICES[@]} node(s), ${EXPECTED} GPU(s)"
# The Ray nodes run in their own sessions: stopping the server reaches them only
# through cleanup(), which stops the workers before the head.
setsid "$RAY_NODE" "$RAY_LOGS" "$VLLM_VENV/bin/ray" start --head \
  --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
  --temp-dir="$RAY_TMP" --include-dashboard=false \
  --num-cpus="$NODE_CPUS" --num-gpus="$HEAD_GPUS" --block &
HEAD_PID=$!
wait_for_gpus "$HEAD_GPUS" 300

for slice in "${SLICES[@]:1}"; do
  host=${slice%%=*}
  devices=${slice#*=}
  echo "[ray] worker on ${host} (GPUs ${devices})"
  # Take the node's GPUs and expose only the placed ones (Slurm picks a step's
  # GPUs itself); --overlap so other pools' steps on the node are not blocked.
  # The worker creates the node-local directories exported above on its own node.
  setsid srun --nodes=1 --ntasks=1 --nodelist="$host" --overlap \
    --gpus-per-node="$NODE_GPUS" --gpu-bind=none --cpus-per-task="$STEP_CPUS" --export=ALL \
    env "CUDA_VISIBLE_DEVICES=${devices}" "RAY_NODE_STOP_FILE=${RAY_STOP}" \
    bash -c 'mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" && exec "$@"' worker \
    "$RAY_NODE" "$RAY_LOGS" "$VLLM_VENV/bin/ray" start --address="$RAY_ADDRESS" --temp-dir="$RAY_TMP" \
      --num-cpus="$NODE_CPUS" --num-gpus="$(count "$devices")" --block &
  WORKER_PIDS+=("$!")
done
wait_for_gpus "$EXPECTED" 600
"$VLLM_VENV/bin/ray" status --address="$RAY_ADDRESS" || true

"$VLLM_VENV/bin/vllm" serve "$@" --distributed-executor-backend ray &
VLLM_PID=$!
wait "$VLLM_PID"
