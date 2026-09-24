#!/bin/bash
# =============================================================================
# _vasp-step-frontier.sh
#
# Inner-step launcher for ONE VASP single-point on Frontier (AMD MI250X).
# Called by the active-learning Python driver via:
#
#   bash _vasp-step-frontier.sh <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>
#
# This script does the full module reset/swap so the driver shell (which runs
# under PrgEnv-gnu + the HydraGNN venv + rocm/7.2.0) never has to. We never
# `source activate` any Python venv here — the VASP step is pure binary.
#
# The module stack loaded depends on where <vasp_bin> resolves to: OLCF's
# facility module (/sw/frontier/vasp/..., default vasp/6.6.1-gpu, override via
# VASP_FACILITY_MODULE) if that's what build-vasp-gpu-frontier.sh resolved to,
# otherwise the from-source PrgEnv-cray/rocm-6.2.0 stack.
#
# It is safe to invoke this script concurrently from multiple Python threads
# inside one SLURM allocation: each call issues its own `srun` step which
# Slurm queues against the allocation's free resources.
#
# Exit code is propagated from `srun`. The Python driver captures both stdout
# and stderr into <work_dir>/vasp.out.
# =============================================================================

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>" >&2
  exit 2
fi

WORK_DIR="$1"
VASP_BIN="$2"
NNODES="$3"
RANKS_PER_NODE="$4"
THREADS_PER_RANK="$5"
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))

[[ -d "${WORK_DIR}" ]] || { echo "work_dir not a directory: ${WORK_DIR}" >&2; exit 2; }
[[ -x "${VASP_BIN}"   ]] || { echo "vasp binary not executable: ${VASP_BIN}" >&2; exit 2; }

# ── Module reset to match how this VASP binary was built ────────────────────
# Two possible provenances for VASP_BIN, each needing its own module stack:
#   1. OLCF facility module (/sw/frontier/vasp/...) — preferred by
#      build-vasp-gpu-frontier.sh when available; load it directly.
#   2. Our own from-source build (external/vasp6/src/vasp.6.6.1/...) — needs
#      the exact PrgEnv-cray + cce + rocm/6.2.0 stack it was compiled with
#      (see build-vasp-gpu-frontier.sh for why rocm/6.2.0 is pinned).
# IMPORTANT: do NOT inherit modules from the parent shell — the parent driver
# runs under PrgEnv-gnu + rocm/7.2.0, incompatible with either VASP stack.
if command -v module >/dev/null 2>&1; then
  :
elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/lmod/lmod/init/bash
fi

VASP_BIN_RESOLVED="$(readlink -f "${VASP_BIN}" 2>/dev/null || echo "${VASP_BIN}")"
module reset
if [[ "${VASP_BIN_RESOLVED}" == /sw/frontier/vasp/* ]]; then
  VASP_FACILITY_MODULE="${VASP_FACILITY_MODULE:-vasp/6.6.1-gpu}"
  echo "[vasp-step] using OLCF facility module: ${VASP_FACILITY_MODULE}"
  module load "${VASP_FACILITY_MODULE}"
else
  echo "[vasp-step] using from-source build toolchain (PrgEnv-cray + rocm/6.2.0)"
  module load cpe/24.07
  module load PrgEnv-cray
  module load cce
  module load craype-accel-amd-gfx90a
  module load rocm/6.2.0
  module load amd-mixed/6.2.0
  module load cray-fftw
fi
module unload darshan-runtime 2>/dev/null || true

# Strip env vars that would perturb the VASP launch (set by the Python venv)
unset LD_PRELOAD
unset VLLM_CUDART_SO_PATH
unset PYTHONPATH

# ── VASP runtime tuning ─────────────────────────────────────────────────────
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_TARGET_OFFLOAD="${OMP_TARGET_OFFLOAD:-MANDATORY}"
# Increase per-rank stack to avoid VASP-OpenMP-target stack overflows
export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"

cd "${WORK_DIR}"

echo "[vasp-step] $(date) host=$(hostname) work_dir=${WORK_DIR}"
NODE_ARGS=()
if [[ -n "${MATSIM_DFT_ASSIGNED_NODES:-}" ]]; then
  NODE_ARGS=(--nodelist "${MATSIM_DFT_ASSIGNED_NODES}")
fi
echo "[vasp-step] assigned_nodes=${MATSIM_DFT_ASSIGNED_NODES:-scheduler-selected}"
echo "[vasp-step] srun -N ${NNODES} -n ${TOTAL_RANKS} -c ${THREADS_PER_RANK} \\"
echo "             --gpus-per-node=${RANKS_PER_NODE} --gpu-bind=closest ${VASP_BIN}"

exec srun \
  --exclusive \
  "${NODE_ARGS[@]}" \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" \
  --gpus-per-node="${RANKS_PER_NODE}" \
  --gpu-bind=closest \
  "${VASP_BIN}"
