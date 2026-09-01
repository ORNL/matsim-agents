#!/bin/bash
#SBATCH -J build-qe-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o runs/build-qe-cpu-%j.out
#SBATCH -e runs/build-qe-cpu-%j.err

# =============================================================================
# Build Quantum ESPRESSO CPU-only on Perlmutter
#
# Compiler: PrgEnv-gnu (gfortran 13 / gcc 13)
# MPI:      cray-mpich
# FFTW:     cray-fftw
# BLAS/LAPACK: cray-libsci (implicit via ftn/cc wrappers)
#
# Useful for:
#   - Testing QE builds before GPU version
#   - Running on CPU nodes without GPU
#   - Debugging or development (faster edit-compile cycles)
#
# GPU notes:
#   - For GPU builds on Perlmutter, use build-qe-gpu-perlmutter.sh (PrgEnv-nvidia)
#
# Usage:
#   sbatch deployments/perlmutter/setup/build-qe-cpu-perlmutter.sh
#   -- or on login node (survives disconnect) --
#   nohup bash deployments/perlmutter/setup/build-qe-cpu-perlmutter.sh > runs/build-qe-cpu-login/build.log 2>&1 &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# ---- Configuration ----------------------------------------------------------
QE_VERSION="${QE_VERSION:-develop}"      # git tag (e.g. "7.4") or "develop"
QE_REPO="${QE_REPO:-https://gitlab.com/QEF/q-e.git}"

QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
SRC_DIR="${SRC_DIR:-${QE_PREFIX}/src}"
BUILD_DIR="${BUILD_DIR:-${QE_PREFIX}/build-cpu}"
INSTALL_DIR="${INSTALL_DIR:-${QE_PREFIX}/install-cpu}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NCORES="${NCORES:-128}"  # Perlmutter CPU node has 128 cores
else
  NCORES="${NCORES:-16}"   # shared login node
fi

# ---- Create output directory ------------------------------------------------
mkdir -p "${PROJ}/runs/build-qe-cpu-${SLURM_JOB_ID:-login}" 2>/dev/null || true

echo "=========================================="
echo "Quantum ESPRESSO CPU-only build on Perlmutter"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "QE version: ${QE_VERSION}"
echo "Build dir:  ${BUILD_DIR}"
echo "Install:    ${INSTALL_DIR}"
echo "=========================================="

# ---- Load modules -----------------------------------------------------------
echo ""
echo "--- Loading Perlmutter modules (HydraGNN-aligned stack) ---"

# Use the same module versions as the HydraGNN Perlmutter installer so QE-CPU
# binaries are ABI-compatible with anything built/run from that environment.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/perlmutter-module-stack.sh"
load_perlmutter_modules

# QE-specific extras on top of HydraGNN stack
module load cray-fftw
module load cmake/3.30.5 2>/dev/null || module load cmake 2>/dev/null || true
module load git/2.47.0   2>/dev/null || module load git   2>/dev/null || true

echo ""
echo "--- Loaded modules ---"
module list
echo ""
echo "--- Compiler versions ---"
gfortran --version 2>&1 | head -1
gcc --version 2>&1 | head -1
echo "CRAY_FFTW_PREFIX=${CRAY_FFTW_PREFIX:-(unset)}"
echo ""

# ---- Clone or update source -------------------------------------------------
if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "Cloning QE ${QE_VERSION} from ${QE_REPO} ..."
    mkdir -p "$(dirname "${SRC_DIR}")"
    if [[ "${QE_VERSION}" == "develop" ]]; then
        git clone --branch develop "${QE_REPO}" "${SRC_DIR}"
    else
        git clone --branch "qe-${QE_VERSION}" "${QE_REPO}" "${SRC_DIR}"
    fi
else
    echo "Source already present at ${SRC_DIR}, skipping clone."
    echo "  HEAD = $(cd "${SRC_DIR}" && git rev-parse --short HEAD)"
fi

# ---- Configure with CMake ---------------------------------------------------
if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
  echo "Removing old build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "--- Running CMake configure ---"

cmake \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_Fortran_COMPILER=ftn \
    \
    -DQE_ENABLE_MPI=ON \
    -DQE_ENABLE_MPI_MODULE=OFF \
    -DQE_ENABLE_OPENMP=ON \
    \
    -DQE_ENABLE_SCALAPACK=OFF \
    \
    -DQE_FFTW_VENDOR=FFTW3 \
    -DFFTW3_ROOT="${CRAY_FFTW_PREFIX}" \
    \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    \
    "${SRC_DIR}" 2>&1 | tee "${BUILD_DIR}/cmake.log"

echo ""
echo "--- CMake configure done ---"

# ---- Build ------------------------------------------------------------------
echo "--- Building QE CPU targets with make -j${NCORES} ---"
cd "${BUILD_DIR}"
make -j"${NCORES}" pw ph pp cp neb hp ld1 2>&1 | tee "${BUILD_DIR}/build.log"
MAKE_RC=$?

if (( MAKE_RC != 0 )); then
  echo "BUILD FAILED with rc=${MAKE_RC}"
  echo "Check build.log: ${BUILD_DIR}/build.log"
  exit ${MAKE_RC}
fi

echo ""
echo "--- Build complete ---"

# ---- Install ----------------------------------------------------------------
echo "--- Installing executables to ${INSTALL_DIR}/bin ---"
mkdir -p "${INSTALL_DIR}/bin"
cp "${BUILD_DIR}"/bin/*.x "${INSTALL_DIR}/bin/" 2>/dev/null || {
  echo "WARNING: No executables found in ${BUILD_DIR}/bin"
  find "${BUILD_DIR}" -maxdepth 3 -name "*.x" -type f -exec cp {} "${INSTALL_DIR}/bin/" \; || true
}
echo "  installed $(ls "${INSTALL_DIR}/bin/" | wc -l) executables"

echo ""
echo "=========================================="
echo "CPU build finished: $(date)"
echo "Executables in: ${INSTALL_DIR}/bin/"
ls "${INSTALL_DIR}/bin/" 2>/dev/null | head -10
echo "=========================================="
echo ""
echo "To use QE CPU executables, set PATH:"
echo "  module load PrgEnv-gnu cray-fftw"
echo "  export PATH=${INSTALL_DIR}/bin:\$PATH"
echo ""
echo "To run on compute nodes:"
echo "  srun -N1 -n64 ${INSTALL_DIR}/bin/pw.x -in your-input.in"
