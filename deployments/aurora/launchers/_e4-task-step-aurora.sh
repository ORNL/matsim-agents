#!/bin/bash
# One E4 controlled task step on its assigned nodes inside an Aurora PBS allocation.
# Contract: <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>
# Binding mirrors _qe-step-aurora.sh so the step exercises the same placement path.
set -euo pipefail

[[ $# -ge 6 && "$5" == "--" ]] || {
  echo "Usage: $0 <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>" >&2
  exit 2
}
WORK_DIR=$1; NNODES=$2; RANKS_PER_NODE=$3; THREADS_PER_RANK=$4
shift 5
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
HOSTS=${MATSIM_ASSIGNED_NODES:?dispatcher must assign disjoint Aurora nodes}

export OMP_NUM_THREADS=${THREADS_PER_RANK}
cd "${WORK_DIR}"
CPU_ARGS=(--depth="${THREADS_PER_RANK}" --cpu-bind=depth)
IFS=, read -r -a TILES <<<"${MATSIM_DEVICE_MASK:-0,1,2,3,4,5,6,7,8,9,10,11}"
RANK_TILES=()
for ((rank = 0; rank < RANKS_PER_NODE; rank++)); do
  RANK_TILES+=("${TILES[rank % ${#TILES[@]}]}")
done
GPU_ARGS=(--gpu-bind="list:$(IFS=:; echo "${RANK_TILES[*]}")")
unset ZE_AFFINITY_MASK
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
echo "[e4-step] assigned_nodes=${HOSTS} ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}"
exec mpiexec --hosts "${HOSTS}" -n "${TOTAL_RANKS}" --ppn "${RANKS_PER_NODE}" \
  "${CPU_ARGS[@]}" "${GPU_ARGS[@]}" "$@"
