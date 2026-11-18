#!/bin/bash
# One E4 controlled task step on this host (local smoke tests; no scheduler).
# Contract: <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>
set -euo pipefail

[[ $# -ge 6 && "$5" == "--" ]] || {
  echo "Usage: $0 <work_dir> <nodes> <ranks_per_node> <threads_per_rank> -- <command...>" >&2
  exit 2
}
WORK_DIR=$1; THREADS_PER_RANK=$4
shift 5
export OMP_NUM_THREADS=${THREADS_PER_RANK}
cd "${WORK_DIR}"
echo "[e4-step] local assigned_nodes=${MATSIM_ASSIGNED_NODES:-local}"
exec "$@"
