#!/bin/bash
# One exclusive logical VASP labeling step inside an Aurora PBS allocation.
# Contract: <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>
set -euo pipefail

[[ $# -eq 5 ]] || { echo "Usage: $0 <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>" >&2; exit 2; }
WORK_DIR=$1; VASP_BIN=$2; NNODES=$3; RANKS_PER_NODE=$4; THREADS_PER_RANK=$5
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
HOSTS=${MATSIM_ASSIGNED_NODES:?dispatcher must assign disjoint Aurora nodes}

export OMP_NUM_THREADS=${THREADS_PER_RANK}
export MPICH_GPU_SUPPORT_ENABLED=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
cd "${WORK_DIR}"
echo "[vasp-step] assigned_nodes=${HOSTS} ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}"
# Each rank owns THREADS_PER_RANK cores (--cpu-bind=cores would put all of a
# rank's OpenMP threads on one core).
CPU_ARGS=(--depth="${THREADS_PER_RANK}" --cpu-bind=depth)
# Ranks go to the placed tiles: nearest tile when the step uses the whole node,
# otherwise an explicit rank-to-tile list from the placement's MATSIM_DEVICE_MASK.
GPU_ARGS=(--gpu-bind=closest)
if [[ -n "${MATSIM_DEVICE_MASK:-}" && -n "${MATSIM_NODE_DEVICES:-}" \
      && "${MATSIM_DEVICES_PER_NODE:-}" != "${MATSIM_NODE_DEVICES}" ]]; then
  IFS=, read -r -a TILES <<<"${MATSIM_DEVICE_MASK}"
  RANK_TILES=()
  for ((rank = 0; rank < RANKS_PER_NODE; rank++)); do
    RANK_TILES+=("${TILES[rank % ${#TILES[@]}]}")
  done
  GPU_ARGS=(--gpu-bind="list:$(IFS=:; echo "${RANK_TILES[*]}")")
fi
# mpiexec sets each rank's ZE_AFFINITY_MASK from --gpu-bind (FLAT tile indices).
unset ZE_AFFINITY_MASK
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
echo "[step] cpu_args=${CPU_ARGS[*]} gpu_args=${GPU_ARGS[*]}"
exec mpiexec --hosts "${HOSTS}" -n "${TOTAL_RANKS}" --ppn "${RANKS_PER_NODE}" \
  "${CPU_ARGS[@]}" "${GPU_ARGS[@]}" "${VASP_BIN}"
