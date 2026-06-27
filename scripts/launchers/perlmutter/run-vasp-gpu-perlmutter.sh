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
#
# Perlmutter GPU node topology: 4× NVIDIA A100 (sm_80) per node, 64 cores.
# Default rank/thread layout: 4 MPI ranks (one per GPU), 16 OMP threads/rank.
#
# Tunables (env vars):
#   VASP_VARIANT=std|gam|ncl                (default: std)
#   VASP_BIN=/abs/path/to/vasp_std          (overrides VASP_VARIANT lookup)
#   VASP_ROOT=<repo>/external/vasp6/src/vasp.6.6.0
#   NRANKS=4 OMP_NUM_THREADS=16 GPUS_PER_NODE=4
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents

VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.0}"
VASP_VARIANT="${VASP_VARIANT:-std}"
VASP_BIN="${VASP_BIN:-${VASP_ROOT}/bin/vasp_${VASP_VARIANT}}"

NRANKS="${NRANKS:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

[[ -x "${VASP_BIN}" ]] || { echo "vasp binary not found / not executable: ${VASP_BIN}" >&2; exit 2; }
for f in INCAR POSCAR KPOINTS POTCAR; do
  [[ -f "${f}" ]] || echo "WARNING: ${f} not present in $(pwd) — VASP will likely fail" >&2
done

# Load the same module stack used at build time. We use the QE-aligned helper
# (PrgEnv-gnu + cpe/24.07 + cray-mpich/8.1.30 + cudatoolkit/12.9 + nvfortran on
# PATH) so that ABI matches both the matsim venv and the linked NVHPC libs.
STACK="${REPO}/scripts/setup/perlmutter/perlmutter-module-stack.sh"
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
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/extras/qd/lib:${LD_LIBRARY_PATH:-}"
# VASP 6.6.0 was built with CUDA-aware MPI and refuses to run without it.
# Keep GPU-aware MPI enabled (=1) but disable NCCL P2P transport, which
# causes "invalid device ordinal" / CUDA_ERROR_ILLEGAL_ADDRESS on single-node
# jobs where intra-node GPU-GPU transfers via NCCL P2P conflict with the
# cray-mpich GTL shim. Disabling P2P falls back to NCCL SHM/socket transport.
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_P2P_DISABLE=1

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
