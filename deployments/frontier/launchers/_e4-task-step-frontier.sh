#!/bin/bash
# One E4 controlled task step on its assigned nodes inside a Frontier Slurm allocation.
# Contract: <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>
# Step creation mirrors _qe-step-frontier.sh (exclusive srun step per node group).
set -euo pipefail

[[ $# -ge 6 && "$5" == "--" ]] || {
  echo "Usage: $0 <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>" >&2
  exit 2
}
WORK_DIR=$1; NNODES=$2; RANKS_PER_NODE=$3; THREADS_PER_RANK=$4
shift 5
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
GPUS_PER_NODE=${MATSIM_DEVICES_PER_NODE:-0}

export OMP_NUM_THREADS=${THREADS_PER_RANK}
cd "${WORK_DIR}"
NODE_ARGS=()
[[ -n "${MATSIM_ASSIGNED_NODES:-}" ]] && NODE_ARGS=(--nodelist "${MATSIM_ASSIGNED_NODES}")
GPU_ARGS=()
STEP_ENV=()
STEP_SHARE=(--exclusive)
if [[ "${GPUS_PER_NODE}" -gt 0 ]]; then
  GPU_ARGS=(--gpus-per-node="${GPUS_PER_NODE}" --gpu-bind=closest)
  if [[ -n "${MATSIM_DEVICE_MASK:-}" && -n "${MATSIM_NODE_DEVICES:-}" \
        && "${GPUS_PER_NODE}" != "${MATSIM_NODE_DEVICES}" ]]; then
    GPU_ARGS=(--gpus-per-node="${MATSIM_NODE_DEVICES}" --gpu-bind=none)
    STEP_SHARE=(--overlap)
    STEP_ENV=(env -u HIP_VISIBLE_DEVICES "ROCR_VISIBLE_DEVICES=${MATSIM_DEVICE_MASK}")
  fi
fi
echo "[e4-step] assigned_nodes=${MATSIM_ASSIGNED_NODES:-scheduler-selected} ranks=${TOTAL_RANKS}"
exec srun "${STEP_SHARE[@]}" "${NODE_ARGS[@]}" -N "${NNODES}" -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" "${GPU_ARGS[@]}" "${STEP_ENV[@]}" "$@"
