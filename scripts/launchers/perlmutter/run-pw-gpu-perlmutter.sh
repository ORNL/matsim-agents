#!/bin/bash
# ---------------------------------------------------------------------------
# matsim-agents: Quantum ESPRESSO ``pw.x`` GPU launcher for NERSC Perlmutter.
#
# Invoked by ``matsim_agents.tools.qe_relax`` (and the warm-start benchmark)
# as ``$MATSIM_QE_LAUNCHER <pw-input-file> [extra pw.x args...]``.
#
# Toolchain (must match build-qe-gpu-perlmutter.sh):
#   PrgEnv-nvidia + NVHPC 25.5 (bundles CUDA 12.9) + cray-mpich + cray-fftw.
#
# Perlmutter GPU node topology: 4× NVIDIA A100 (sm_80) per node, 64 CPU cores.
# Default rank/thread layout: 4 MPI ranks (one per GPU), 16 OMP threads/rank.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents

QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
PW_BIN="${PW_BIN:-${QE_PREFIX}/install-gpu/bin/pw.x}"
NRANKS="${NRANKS:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pw-input-file> [extra pw.x args...]" >&2
  exit 2
fi
INPUT="$1"; shift
[[ ! -f "${INPUT}" ]] && { echo "Input file not found: ${INPUT}" >&2; exit 2; }
[[ ! -x "${PW_BIN}" ]] && { echo "pw.x not found or not executable: ${PW_BIN}" >&2; exit 2; }

# Same module stack as the build (see build-qe-gpu-perlmutter.sh for the
# rationale on the NVHPC 25.5 / CUDA 12.9 pin: it must match the CUDA runtime
# used by the HydraGNN PyTorch wheel — torch 2.11.0+cu129).
source "${REPO}/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_nvidia
module load cray-fftw || echo "WARNING: cray-fftw not loaded"

export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# Default to non-GTL mode on Perlmutter launcher paths used by these
# benchmarks. We intentionally ignore inherited MPICH_GPU_SUPPORT_ENABLED from
# outer environments because those often force 1 and trigger GTL init aborts.
# Opt in explicitly via MATSIM_QE_ENABLE_GPU_AWARE_MPI=1 when GTL-linked MPICH
# is guaranteed in the runtime environment.
if [[ "${MATSIM_QE_ENABLE_GPU_AWARE_MPI:-0}" == "1" ]]; then
  export MPICH_GPU_SUPPORT_ENABLED=1
else
  export MPICH_GPU_SUPPORT_ENABLED=0
fi

echo "=========================================="
echo "QE pw.x GPU run on Perlmutter"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "pw.x:        ${PW_BIN}"
echo "Input:       ${INPUT}"
echo "MPI ranks:   ${NRANKS}    OMP threads/rank: ${OMP_NUM_THREADS}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "MPICH GPU-aware MPI: MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
echo "=========================================="

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun -n "${NRANKS}" -c "${OMP_NUM_THREADS}" \
       --gpus-per-node="${GPUS_PER_NODE}" --gpu-bind=closest \
       "${PW_BIN}" -in "${INPUT}" "$@"
else
  srun -N1 -n "${NRANKS}" -c "${OMP_NUM_THREADS}" \
       --gpus-per-node="${GPUS_PER_NODE}" --gpu-bind=closest \
       "${PW_BIN}" -in "${INPUT}" "$@"
fi
