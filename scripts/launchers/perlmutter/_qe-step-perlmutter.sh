#!/bin/bash
# =============================================================================
# _qe-step-perlmutter.sh
#
# Inner-step launcher for ONE Quantum ESPRESSO pw.x single-point on NERSC
# Perlmutter (NVIDIA A100 / sm_80). Called by the active-learning Python driver
# (matsim_agents.active_learning.backends.qe.QEBackend) via:
#
#   bash _qe-step-perlmutter.sh <work_dir> <pw_bin> <input_file> \
#                               <nodes> <ranks_per_node> <threads_per_rank>
#
# This mirrors _vasp-step-perlmutter.sh / _qe-step-frontier.sh: it performs the
# full module swap to the QE build's toolchain so the driver shell (PrgEnv-gnu
# + fairchem/HydraGNN venv) never has to, and it is safe to invoke concurrently
# from multiple Python threads in one allocation (each call issues its own
# `srun --exclusive` step).
#
# Toolchain pin matches scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh and
# scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh:
#   PrgEnv-nvidia + NVHPC 25.5 (CUDA 12.9) + cray-mpich + cray-fftw.
# =============================================================================

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <work_dir> <pw_bin> <input_file> <nodes> <ranks_per_node> <threads_per_rank>" >&2
  exit 2
fi

WORK_DIR="$1"
PW_BIN="$2"
INPUT="$3"
NNODES="$4"
RANKS_PER_NODE="$5"
THREADS_PER_RANK="$6"
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))

[[ -d "${WORK_DIR}" ]] || { echo "work_dir not a directory: ${WORK_DIR}" >&2; exit 2; }
[[ -x "${PW_BIN}"   ]] || { echo "pw.x not executable: ${PW_BIN}" >&2; exit 2; }
[[ -f "${INPUT}"    ]] || { echo "input file missing: ${INPUT}" >&2; exit 2; }

# ── Module swap to the QE build's toolchain ──────────────────────────────────
# Resolve the repo so we can reuse the shared module-stack helpers. The
# active-learning driver passes absolute paths, so derive REPO from this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents

# shellcheck disable=SC1091
source "${REPO}/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_nvidia
module load cray-fftw || echo "WARNING: cray-fftw not loaded"

export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Strip env vars set by the Python venv that would perturb the QE launch.
unset LD_PRELOAD  || true
unset PYTHONPATH  || true

# ── QE runtime tuning ────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"
# GPU-aware MPI (GTL) is off by default: the GTL library is not linked in the
# benchmark environment and forcing it triggers init aborts. Opt in explicitly
# via MATSIM_QE_ENABLE_GPU_AWARE_MPI=1 only when GTL-linked MPICH is guaranteed.
if [[ "${MATSIM_QE_ENABLE_GPU_AWARE_MPI:-0}" == "1" ]]; then
  export MPICH_GPU_SUPPORT_ENABLED=1
else
  export MPICH_GPU_SUPPORT_ENABLED=0
fi

cd "${WORK_DIR}"

echo "[qe-step] $(date) host=$(hostname) work_dir=${WORK_DIR}"
echo "[qe-step] srun --exclusive -N ${NNODES} -n ${TOTAL_RANKS} -c ${THREADS_PER_RANK} \\"
echo "             --gpus-per-node=${RANKS_PER_NODE} --gpu-bind=closest ${PW_BIN} -in ${INPUT}"

# --exclusive lets the AL driver run several of these srun steps concurrently
# inside one allocation (as with _vasp-step-perlmutter.sh); without it Slurm
# serialises the steps ("Job step creation temporarily disabled").
exec srun \
  --exclusive \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" \
  --gpus-per-node="${RANKS_PER_NODE}" \
  --gpu-bind=closest \
  "${PW_BIN}" -in "${INPUT}"
