#!/bin/bash
# One exclusive logical VASP labeling step inside an Aurora PBS allocation.
# Contract: <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>
set -euo pipefail

[[ $# -eq 5 ]] || { echo "Usage: $0 <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>" >&2; exit 2; }
WORK_DIR=$1; VASP_BIN=$2; NNODES=$3; RANKS_PER_NODE=$4; THREADS_PER_RANK=$5
TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
HOSTS=${MATSIM_DFT_ASSIGNED_NODES:?dispatcher must assign disjoint Aurora nodes}

export OMP_NUM_THREADS=${THREADS_PER_RANK}
export MPICH_GPU_SUPPORT_ENABLED=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
cd "${WORK_DIR}"
echo "[vasp-step] assigned_nodes=${HOSTS} ranks=${TOTAL_RANKS} ppn=${RANKS_PER_NODE}"
exec mpiexec --hosts "${HOSTS}" -n "${TOTAL_RANKS}" --ppn "${RANKS_PER_NODE}" \
  --cpu-bind=cores --gpu-bind=closest "${VASP_BIN}"
