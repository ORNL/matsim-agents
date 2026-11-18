#!/usr/bin/env bash
#PBS -N qe-epw2-diag
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
SOURCE_TEST_SUITE="${QE_PREFIX}/src/test-suite"
RUN_DIR="${REPO}/runs/qe-epw2-diagnostic-${PBS_JOBID:-local}"
OFFLOAD_MODE="${QE_OFFLOAD_MODE:-MANDATORY}"

[[ "${OFFLOAD_MODE}" == "MANDATORY" || "${OFFLOAD_MODE}" == "DISABLED" || "${OFFLOAD_MODE}" == "BOTH" ]] || {
  echo "QE_OFFLOAD_MODE must be MANDATORY, DISABLED, or BOTH" >&2
  exit 1
}

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

source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
module load xpu-smi/1.3.5

export LIBOMPTARGET_INFO=16
export OMP_DISPLAY_ENV=VERBOSE
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=4
export KMP_SETTINGS=TRUE
export QE_USE_MPI=1
export ESPRESSO_ROOT="${QE_PREFIX}"
export ESPRESSO_BUILD="${QE_BIN_DIR%/bin}"

run_diagnostic() {
  local mode="$1"
  local mode_dir="${RUN_DIR}/${mode}"
  local test_suite="${mode_dir}/test-suite"
  local diagnostic_id="${PBS_JOBID//[^A-Za-z0-9_-]/_}"
  local test_rc
  local telemetry_pid

  mkdir -p "${test_suite}/epw_mob_polar" "${mode_dir}/tmp" || return 1
  cp "${SOURCE_TEST_SUITE}/jobconfig" "${test_suite}/jobconfig" || return 1
  cp "${SOURCE_TEST_SUITE}/userconfig" "${test_suite}/userconfig" || return 1
  tar -C "${SOURCE_TEST_SUITE}/epw_mob_polar" --exclude='core.*' -cf - . \
    | tar -C "${test_suite}/epw_mob_polar" -xf - || return 1

  export OMP_TARGET_OFFLOAD="${mode}"
  export ESPRESSO_TMPDIR="${mode_dir}/tmp"
  "$(command -v xpu-smi)" dump -d -1 -m 0,9 -i 1 --date \
    > "${mode_dir}/gpu-telemetry.log" 2>&1 &
  telemetry_pid=$!
  cd "${test_suite}"
  if python3 "${SOURCE_TEST_SUITE}/testcode/bin/testcode.py" --verbose \
    --jobconfig="${test_suite}/jobconfig" \
    --category=epw_mob_polar \
    --test-id="epw2-${mode}-${diagnostic_id}" \
    --job-option 'epw_mob_polar/' inputs_args "('epw2.in','3')" \
    2>&1 | tee "${mode_dir}/epw-test.log"; then
    test_rc=0
  else
    test_rc=${PIPESTATUS[0]}
  fi
  kill "${telemetry_pid}" 2>/dev/null || true
  wait "${telemetry_pid}" 2>/dev/null || true
  echo "QE epw2 diagnostic mode=${mode} testcode_status=${test_rc}"
  return "${test_rc}"
}

if [[ "${OFFLOAD_MODE}" == "BOTH" ]]; then
  if run_diagnostic MANDATORY; then GPU_RC=0; else GPU_RC=$?; fi
  if run_diagnostic DISABLED; then HOST_RC=0; else HOST_RC=$?; fi
  if [[ "${GPU_RC}" -ne 0 || "${HOST_RC}" -ne 0 ]]; then
    exit 1
  fi
else
  if run_diagnostic "${OFFLOAD_MODE}"; then
    exit 0
  else
    exit $?
  fi
fi