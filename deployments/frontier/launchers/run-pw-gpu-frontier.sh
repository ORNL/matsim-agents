#!/bin/bash
#SBATCH -J qe-pw-gpu
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# =============================================================================
# Run pw.x on Frontier with AMD MI250X (gfx90a) OpenMP target offload.
#
# This launcher pairs with: deployments/frontier/setup/build-qe-gpu-frontier.sh
# It loads the same module stack used at build time and executes pw.x via
# srun with one MPI rank per GCD (8 GCDs/node on MI250X).
#
# Usage (sbatch):
#   sbatch deployments/frontier/launchers/run-pw-gpu-frontier.sh path/to/input.in
#
# Usage (interactive on a compute node):
#   bash deployments/frontier/launchers/run-pw-gpu-frontier.sh path/to/input.in
#
# Environment overrides (defaults shown):
#   QE_PREFIX        = <repo>/external/quantum-espresso
#   PW_BIN           = ${QE_PREFIX}/install-gpu/bin/pw.x
#   ROCM_MODULE      = rocm/6.2.4
#   NRANKS           = 8         (one MPI rank per GCD)
#   OMP_NUM_THREADS  = 7         (Frontier core-per-GCD count)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
PW_BIN="${PW_BIN:-${QE_PREFIX}/install-gpu/bin/pw.x}"
ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.4}"
NRANKS="${NRANKS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pw-input-file> [extra pw.x args...]" >&2
  exit 2
fi
INPUT="$1"; shift
[[ ! -f "${INPUT}" ]] && { echo "Input file not found: ${INPUT}" >&2; exit 2; }
[[ ! -x "${PW_BIN}" ]] && { echo "pw.x not found or not executable: ${PW_BIN}" >&2; exit 2; }

# A caller (e.g. job-portability-benchmark-frontier.sh) may have this launcher
# as a subprocess of a rocm-7.2.0 Python venv shell, which exports
# LD_LIBRARY_PATH pointing at torch's bundled libtinfo.so.6. module reset does
# not clear manually-exported env vars, so without this the venv's libtinfo
# gets picked up by pw.x instead of the system one, crashing it (SIGSEGV).
# Clearing it here lets the module loads below repopulate it from scratch.
unset LD_LIBRARY_PATH

# Same module stack as the build (see build-qe-gpu-frontier.sh for rationale
# on the rocm/6.2.4 pin).
#
# module commands here can spuriously return non-zero: Cray's Lmod init
# (sourced via BASH_ENV in every subshell) tries to auto-deactivate conda
# and fails with "CondaError: Run 'conda init' before 'conda deactivate'"
# when invoked from a script rather than a login shell, which otherwise
# aborts this script under set -e before pw.x ever runs.
set +e
module reset
module load PrgEnv-cray
module load cce
module load craype-accel-amd-gfx90a
module load "${ROCM_MODULE}"
module load cray-fftw
set -e

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_TARGET_OFFLOAD="${OMP_TARGET_OFFLOAD:-DEFAULT}"

echo "=========================================="
echo "QE pw.x GPU run on Frontier"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "pw.x:        ${PW_BIN}"
echo "Input:       ${INPUT}"
echo "MPI ranks:   ${NRANKS}    OMP threads/rank: ${OMP_NUM_THREADS}"
echo "=========================================="

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Inside an allocation: srun directly.
  srun -n "${NRANKS}" -c "${OMP_NUM_THREADS}" --threads-per-core=1 --cpu-bind=cores \
       --gpus-per-node=8 --gpu-bind=closest \
       "${PW_BIN}" -in "${INPUT}" "$@"
else
  # Stand-alone invocation outside an allocation: assume the caller already
  # acquired one (e.g. `salloc -N1 -p batch -A mat746 ...`).
  srun -N1 -n "${NRANKS}" -c "${OMP_NUM_THREADS}" --threads-per-core=1 --cpu-bind=cores \
       --gpus-per-node=8 --gpu-bind=closest \
       "${PW_BIN}" -in "${INPUT}" "$@"
fi
