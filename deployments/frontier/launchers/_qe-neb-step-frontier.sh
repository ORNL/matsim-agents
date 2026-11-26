#!/bin/bash
# =============================================================================
# _qe-neb-step-frontier.sh
#
# Launcher prefix for ONE Quantum ESPRESSO neb.x run on Frontier (AMD MI250X).
# QENebConfig.launcher is a prefix; the Python side appends the executable and its
# arguments, so the command it runs is
#
#   bash _qe-neb-step-frontier.sh <ranks> <threads_per_rank> <neb_bin> [neb.x args...]
#
# The script rebuilds the QE toolchain (the driver shell runs under another module stack),
# then execs a single srun step. It must be called inside a Slurm allocation, from the
# directory neb.x should run in (the Python side sets the working directory).
#
# Module pin matches _qe-step-frontier.sh and deployments/frontier/setup/build-qe-gpu-frontier.sh.
# =============================================================================

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <ranks> <threads_per_rank> <neb_bin> <neb.x args...>" >&2
  exit 2
fi

RANKS="$1"
THREADS_PER_RANK="$2"
NEB_BIN="$3"
shift 3

[[ -x "${NEB_BIN}" ]] || { echo "neb.x not executable: ${NEB_BIN}" >&2; exit 2; }

if command -v module >/dev/null 2>&1; then
  :
elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/lmod/lmod/init/bash
fi

ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.0}"

# module reset does not clear an exported LD_LIBRARY_PATH; the Python venv's bundled libtinfo
# would shadow the system one (see run-pw-gpu-frontier.sh).
unset LD_LIBRARY_PATH

module reset
module load PrgEnv-cray
module load cce
module load craype-accel-amd-gfx90a
module load "${ROCM_MODULE}"
module load cray-fftw
module unload darshan-runtime 2>/dev/null || true

unset LD_PRELOAD
unset VLLM_CUDART_SO_PATH
unset PYTHONPATH


export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_TARGET_OFFLOAD="${OMP_TARGET_OFFLOAD:-DEFAULT}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"

echo "[qe-neb-step] $(date) host=$(hostname) cwd=$(pwd) ranks=${RANKS} threads=${THREADS_PER_RANK}"
echo "[qe-neb-step] srun -N 1 -n ${RANKS} ${NEB_BIN} $*"
exec srun \
  --exclusive \
  -N 1 \
  -n "${RANKS}" \
  -c "${THREADS_PER_RANK}" \
  --gpus-per-node="${MATSIM_GPUS_PER_NODE:-8}" \
  --gpu-bind=closest \
  "${NEB_BIN}" "$@"
