#!/bin/bash
# =============================================================================
# _vasp-step-perlmutter.sh
#
# Inner-step launcher for ONE VASP single-point on NERSC Perlmutter
# (NVIDIA A100, sm_80).  Called by the active-learning Python driver
# (matsim_agents.active_learning.backends.vasp) via:
#
#   bash _vasp-step-perlmutter.sh <work_dir> <vasp_bin> <nodes> <ranks_per_node> <threads_per_rank>
#
# The driver shell runs under PrgEnv-gnu + the matsim/HydraGNN venv. This
# script does the full module reset/swap to the VASP build's toolchain so the
# driver never has to. We never `source activate` any Python venv here — the
# VASP step is a pure binary launch.
#
# Safe to invoke concurrently from multiple Python threads inside one SLURM
# allocation: each call issues its own `srun` step which Slurm queues against
# the allocation's free resources.
#
# Exit code is propagated from `srun`. The Python driver captures stdout and
# stderr into <work_dir>/vasp.out.
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
[[ -x "${VASP_BIN}"  ]] || { echo "vasp binary not executable: ${VASP_BIN}" >&2; exit 2; }

# ── Module reset to the VASP build's toolchain ──────────────────────────────
# Use the same NVIDIA stack used at build time (PrgEnv-gnu + cpe/24.07 +
# cray-mpich/8.1.30 + cudatoolkit/12.9 + nvfortran on PATH) so the ABI matches
# the linked NVHPC libs. `module reset` is mandatory to drop inherited modules.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/lmod/lmod/init/bash
fi

STACK="${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
if [[ -f "${STACK}" ]]; then
  # shellcheck disable=SC1090
  source "${STACK}"
  load_perlmutter_modules_nvidia
else
  echo "ERROR: module stack helper missing at ${STACK}" >&2
  exit 1
fi

# Strip env vars that would perturb the VASP launch (set by the Python venv)
unset LD_PRELOAD
unset VLLM_CUDART_SO_PATH
unset PYTHONPATH

# ── NVHPC runtime paths ─────────────────────────────────────────────────────
# Include compilers/extras/qd/lib so the linked-in libqdmod.so.0 / libqd.so.0
# (NVHPC quad-double precision lib) resolve at startup.
export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/extras/qd/lib:${LD_LIBRARY_PATH:-}"

# ── VASP runtime tuning (A100: 4 GPUs/node, 64 cores) ───────────────────────
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"
export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"
# VASP 6.6.0 was built with CUDA-aware MPI and refuses to run without it.
# Keep GPU-aware MPI enabled (=1). Disable both NCCL P2P and SHM transports,
# which otherwise fail with "Cuda failure 101 / invalid device ordinal" on
# single-node Perlmutter jobs (GPU cgroup isolation conflicts with NCCL's
# intra-node device enumeration). Disabling both forces NCCL onto socket
# (loopback TCP) transport, which works correctly on a single node. This
# mirrors run-vasp-gpu-perlmutter.sh, the known-good warm-start launcher.
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

cd "${WORK_DIR}"

echo "[vasp-step] $(date) host=$(hostname) work_dir=${WORK_DIR}"
echo "[vasp-step] srun --exclusive -N ${NNODES} -n ${TOTAL_RANKS} -c ${THREADS_PER_RANK} \\"
echo "             --gpus-per-node=${RANKS_PER_NODE} --gpu-bind=closest ${VASP_BIN}"

# --exclusive is REQUIRED for concurrent multi-node dispatch: the AL driver
# launches up to (SLURM_JOB_NUM_NODES / nodes_per_job) of these steps at once
# from a ThreadPool. Without --exclusive, each srun step defaults to requesting
# the allocation's full resource set, so the 2nd+ concurrent step blocks with
# "srun: Job step creation temporarily disabled, retrying". With --exclusive
# each -N1 step claims one whole, distinct node and they run in parallel.
# For a single-node (-N1) allocation this is a harmless no-op.
exec srun \
  --exclusive \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" \
  --gpus-per-node="${RANKS_PER_NODE}" \
  --gpu-bind=closest \
  "${VASP_BIN}"
