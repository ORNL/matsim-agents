#!/bin/bash
# ---------------------------------------------------------------------------
# matsim-agents: VASP GPU launcher for NERSC Perlmutter (NVIDIA A100, sm_80).
#
# Invoked by ``matsim_agents.tools.vasp_relax`` (and the VASP warm-start
# benchmark) via the ``MATSIM_VASP_LAUNCHER`` env var. The launcher is
# expected to be run *inside the working directory* that already contains
# ``INCAR`` / ``POSCAR`` / ``KPOINTS`` / ``POTCAR``; no argv is passed in.
#
# Toolchain (must match build-vasp-gpu-perlmutter.sh):
#   PrgEnv-gnu/8.5.0 + cpe/24.07 + cray-mpich/8.1.30 + cudatoolkit/12.9
#   + cray-fftw + nvfortran/nvc/nvc++ from NVHPC 25.5 SDK.
# If VASP_BIN resolves under NERSC's facility module install
# (/global/common/software/nersc9/vasp/vasp/...), the NERSC facility module
# (default vasp/6.6.1-gpu, override via VASP_FACILITY_MODULE) is loaded
# instead of the from-source toolchain above.
#
# Perlmutter GPU node topology: 4× NVIDIA A100 (sm_80) per node, 64 cores.
# Default rank/thread layout: 4 MPI ranks (one per GPU), 16 OMP threads/rank.
#
# Tunables (env vars):
#   VASP_VARIANT=std|gam|ncl                (default: std)
#   VASP_BIN=/abs/path/to/vasp_std          (overrides VASP_VARIANT lookup)
#   VASP_ROOT=<repo>/external/vasp6/src/vasp.6.6.1
#   NRANKS=4 OMP_NUM_THREADS=16 GPUS_PER_NODE=4
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.1}"
VASP_VARIANT="${VASP_VARIANT:-std}"
VASP_BIN="${VASP_BIN:-${VASP_ROOT}/bin/vasp_${VASP_VARIANT}}"

NRANKS="${NRANKS:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

[[ -x "${VASP_BIN}" ]] || { echo "vasp binary not found / not executable: ${VASP_BIN}" >&2; exit 2; }
for f in INCAR POSCAR KPOINTS POTCAR; do
  [[ -f "${f}" ]] || echo "WARNING: ${f} not present in $(pwd) — VASP will likely fail" >&2
done

# Load the same module stack used at build time, unless VASP_BIN resolves to
# a NERSC facility module install (mirrors _vasp-step-perlmutter.sh /
# _vasp-step-frontier.sh): if build-vasp-gpu-perlmutter.sh preferred the
# facility module, load that directly instead of the from-source toolchain.
VASP_BIN_RESOLVED="$(readlink -f "${VASP_BIN}" 2>/dev/null || echo "${VASP_BIN}")"
module reset 2>/dev/null || true
if [[ "${VASP_BIN_RESOLVED}" == /global/common/software/nersc9/vasp/vasp/* ]]; then
  VASP_FACILITY_MODULE="${VASP_FACILITY_MODULE:-vasp/6.6.1-gpu}"
  echo "Using NERSC facility module: ${VASP_FACILITY_MODULE}"
  module load "${VASP_FACILITY_MODULE}"
else
  echo "Using from-source build toolchain (PrgEnv-gnu + cudatoolkit/12.9)"
  STACK="${REPO}/deployments/perlmutter/setup/perlmutter-module-stack.sh"
  if [[ -f "${STACK}" ]]; then
    # shellcheck disable=SC1090
    source "${STACK}"
    load_perlmutter_modules_nvidia
  else
    echo "ERROR: module stack helper missing at ${STACK}" >&2
    exit 1
  fi

  export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9}"
  # NVHPC runtime paths: include the `compilers/extras/qd/lib` directory so the
  # linked-in `libqdmod.so.0` / `libqd.so.0` (NVHPC quad-double precision lib)
  # resolve at startup. Without it, vasp exits with
  # "error while loading shared libraries: libqdmod.so.0".
  # Include math_libs/lib64 (version-agnostic dir) FIRST: NVHPC 25.5 does not
  # ship libcusparse under math_libs/12.9/, only under math_libs/lib64/.
  # Without it, the loader falls through to a stray CUDA-13.2 libcusparse.so.12
  # elsewhere on the system, which pulls in libcudart.so.13 (not present on
  # compute nodes) and fails with "libcudart.so.13: cannot open shared object".
  export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/lib64:${CUDA_HOME}/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/extras/qd/lib:${LD_LIBRARY_PATH:-}"

  # Cray's GPU-aware MPI transport library (libmpi_gtl_cuda.so.0) has a hard
  # dependency on libcudart.so.13, independent of the CUDA 12.9 toolchain used
  # to build VASP. NVHPC 25.5 only ships CUDA 12.9, so pull libcudart.so.13
  # from the NVHPC 26.5/CUDA 13.2 bundle. Coexists fine with the CUDA 12.9 libs
  # above since the SONAMEs differ (libcudart.so.12 vs libcudart.so.13).
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64"
fi
# VASP 6.6.1 was built with CUDA-aware MPI and refuses to run without it.
# Keep GPU-aware MPI enabled (=1). Disable both NCCL P2P and SHM transports,
# which both fail with "Cuda failure 101 / invalid device ordinal" on single-
# node Perlmutter jobs due to GPU cgroup isolation conflicting with NCCL's
# intra-node device enumeration. Disabling both forces NCCL onto socket
# (loopback TCP) transport, which works correctly on a single node.
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

echo "=========================================="
echo "VASP GPU run on Perlmutter"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "WorkDir:     $(pwd)"
echo "vasp bin:    ${VASP_BIN}"
echo "Variant:     ${VASP_VARIANT}"
echo "MPI ranks:   ${NRANKS}    OMP threads/rank: ${OMP_NUM_THREADS}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "=========================================="

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec srun -n "${NRANKS}" -c "${OMP_NUM_THREADS}" \
       --gpus-per-node="${GPUS_PER_NODE}" --gpu-bind=closest \
       "${VASP_BIN}"
else
  exec srun -N1 -n "${NRANKS}" -c "${OMP_NUM_THREADS}" \
       --gpus-per-node="${GPUS_PER_NODE}" --gpu-bind=closest \
       "${VASP_BIN}"
fi
