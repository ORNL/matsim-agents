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

# ── Module reset to the VASP build's toolchain ──────────────────────────────
# IMPORTANT: do NOT inherit modules from the parent shell — VASP needs
# PrgEnv-cray + cce + craype-accel-amd-gfx90a + rocm/6.2.0 + cray-fftw,
# while the parent driver runs under PrgEnv-gnu + rocm/7.2.0.
# rocm/6.2.0 (LLVM 18) MUST match what VASP was built with: the cce 18.x GPU
# device link (llvm-link + lld) can only read ROCm bitcode from LLVM <= 18 AND
# needs the non-overloaded readfirstlane intrinsic (ROCm 6.2.4+ broke this), so
# build and runtime are pinned to the same ROCm as build-vasp-gpu-frontier.sh.
# `module reset` is mandatory; otherwise Lmod will refuse to swap PrgEnv.
if command -v module >/dev/null 2>&1; then
  :
elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/lmod/lmod/init/bash
fi

module reset
module load cpe/24.07
module load PrgEnv-cray
module load cce
module load craype-accel-amd-gfx90a
module load rocm/6.2.0
module load amd-mixed/6.2.0
module load cray-fftw
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
echo "[vasp-step] srun -N ${NNODES} -n ${TOTAL_RANKS} -c ${THREADS_PER_RANK} \\"
echo "             --gpus-per-node=${RANKS_PER_NODE} --gpu-bind=closest ${VASP_BIN}"

exec srun \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" \
  --gpus-per-node="${RANKS_PER_NODE}" \
  --gpu-bind=closest \
  "${VASP_BIN}"
