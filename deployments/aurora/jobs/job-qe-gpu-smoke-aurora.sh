#!/usr/bin/env bash
#PBS -N qe-gpu-smoke
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe

set -eo pipefail

REPO="${PROJECT_ROOT:-${PBS_O_WORKDIR:-}}"
if [[ ! -f "${REPO}/pyproject.toml" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
fi
[[ -f "${REPO}/pyproject.toml" ]] || {
  echo "Could not locate matsim-agents; set PROJECT_ROOT or submit from the repo root" >&2
  exit 1
}

QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
BUILD_DIR="${QE_BUILD_DIR:-${QE_PREFIX}/build-gpu}"
QE_BIN_DIR="${QE_BIN_DIR:-${QE_PREFIX}/install-gpu/bin}"
TEST_SUITE="${QE_PREFIX}/src/test-suite"
RUN_DIR="${REPO}/runs/qe-gpu-smoke-${PBS_JOBID:-local}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/tmp"

for executable in pw.x ph.x epw.x; do
  [[ -x "${QE_BIN_DIR}/${executable}" ]] || {
    echo "QE executable missing or not executable: ${QE_BIN_DIR}/${executable}" >&2
    exit 1
  }
done
grep -Fqx 'QE_GPU:STRING=openmp;oneapi' "${BUILD_DIR}/CMakeCache.txt" \
  || { echo "QE build is not configured for oneAPI GPU offload: ${BUILD_DIR}/CMakeCache.txt" >&2; exit 1; }
grep -Fqx 'QE_GPU_ARCHS:STRING=intel_gpu_pvc' "${BUILD_DIR}/CMakeCache.txt" \
  || { echo "QE build does not target intel_gpu_pvc: ${BUILD_DIR}/CMakeCache.txt" >&2; exit 1; }
[[ -d "${TEST_SUITE}/epw_mob_polar" ]] || {
  echo "EPW GPU test fixture missing: ${TEST_SUITE}/epw_mob_polar" >&2
  exit 1
}

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
module load xpu-smi/1.3.5

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=16
export OMP_DISPLAY_ENV=VERBOSE
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=4
export KMP_SETTINGS=TRUE

cd "${TEST_SUITE}"
ESPRESSO_ROOT="${QE_PREFIX}" ESPRESSO_BUILD="${QE_BIN_DIR%/bin}" \
  ESPRESSO_TMPDIR="${RUN_DIR}/tmp" make prolog \
  2>&1 | tee "${RUN_DIR}/testcode-prolog.log"

"$(command -v xpu-smi)" dump -d -1 -m 0,9 -i 1 --date \
  > "${RUN_DIR}/gpu-telemetry.log" 2>&1 &
TELEMETRY_PID=$!
trap 'kill "${TELEMETRY_PID}" 2>/dev/null || true' EXIT

set +e
QE_USE_MPI=1 ESPRESSO_BUILD="${QE_BIN_DIR%/bin}" \
  python3 "${TEST_SUITE}/testcode/bin/testcode.py" --verbose \
  --category=epw_mob_polar --test-id=aurora-intel-gpu \
  2>&1 | tee "${RUN_DIR}/epw-test.log"
TEST_RC=${PIPESTATUS[0]}
set -e

kill "${TELEMETRY_PID}" 2>/dev/null || true
wait "${TELEMETRY_PID}" 2>/dev/null || true
trap - EXIT

grep -R -Fq 'JOB DONE.' "${TEST_SUITE}/epw_mob_polar" || {
  echo "EPW test did not complete successfully; see ${RUN_DIR}/epw-test.log" >&2
  exit 1
}
[[ "${TEST_RC}" -eq 0 ]] || {
  echo "QE EPW test runner failed with status ${TEST_RC}; see ${RUN_DIR}/epw-test.log" >&2
  exit "${TEST_RC}"
}
grep -Eq '([1-9][0-9]*\.?[0-9]*|0\.[0-9]*[1-9])[[:space:]]*%' \
  "${RUN_DIR}/gpu-telemetry.log" || {
  echo "EPW test passed but xpu-smi recorded no nonzero GPU utilization; see ${RUN_DIR}/gpu-telemetry.log" >&2
  exit 1
}

echo "PASS: QE EPW test passed and xpu-smi recorded GPU utilization."