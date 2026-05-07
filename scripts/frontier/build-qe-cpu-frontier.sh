#!/bin/bash
#SBATCH -J build-qe-cpu
#SBATCH -A mat746
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/build-qe-cpu-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/build-qe-cpu-%j/job-%j.err

# =============================================================================
# Build Quantum ESPRESSO CPU-only on Frontier
#
# Compiler: PrgEnv-gnu (gfortran 13 / gcc 13)
# MPI:      cray-mpich
# FFTW:     cray-fftw
# BLAS/LAPACK: cray-libsci (implicit via ftn/cc wrappers)
# ScaLAPACK: disabled (optional, can be re-enabled with SCALAPACK_DIR)
#
# GPU notes:
#   - QE's AMD OpenMP offload path requires nvfortran (not available on Frontier)
#   - gfortran -foffload=amdgcn-amdhsa not configured on Frontier
#   - Use the OLCF module 'quantum-espresso/7.1' for a CUDA GPU build (nvfortran)
#
# Usage:
#   sbatch run-scripts/build-qe-gpu-frontier.sh
#   -- or on login node (survives disconnect) --
#   nohup bash run-scripts/build-qe-gpu-frontier.sh > runs/build-qe-login/build.log 2>&1 &
# =============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
QE_VERSION="develop"                    # git tag (e.g. "7.4") or "develop"
QE_REPO="https://gitlab.com/QEF/q-e.git"
BASE_DIR="/lustre/orion/mat746/proj-shared"
SRC_DIR="${BASE_DIR}/quantum-espresso/src"
BUILD_DIR="${BASE_DIR}/quantum-espresso/build-cpu"
INSTALL_DIR="${BASE_DIR}/quantum-espresso/install-cpu"

NCORES=64   # cores available on a Frontier node for make -j

# ---- Create output directory (Slurm needs it for log files) -----------------
mkdir -p "$(dirname "${SLURM_JOB_ID:+/lustre/orion/mat746/proj-shared/runs/build-qe-gpu-${SLURM_JOB_ID}}")" 2>/dev/null || true

echo "=========================================="
echo "Quantum ESPRESSO CPU build on Frontier"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "QE version: ${QE_VERSION}"
echo "Build dir:  ${BUILD_DIR}"
echo "Install:    ${INSTALL_DIR}"
echo "=========================================="

# ---- Load modules -----------------------------------------------------------
module reset

module load PrgEnv-gnu/8.6.0            # gfortran 13 / gcc 13
module load cray-fftw/3.3.10.9          # Cray-optimised FFTW3 (linked via ftn wrapper)
module load cmake/3.30.5
module load git/2.47.0

echo ""
echo "--- Loaded modules ---"
module list
echo "--- Compiler versions ---"
ftn --version 2>&1 | head -1
cc  --version 2>&1 | head -1
echo "CRAY_FFTW_PREFIX=${CRAY_FFTW_PREFIX}"
echo ""

# ---- Clone or update source -------------------------------------------------
if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "Cloning QE ${QE_VERSION} from ${QE_REPO} ..."
    mkdir -p "$(dirname "${SRC_DIR}")"
    if [[ "${QE_VERSION}" == "develop" ]]; then
        git clone --depth=1 --branch develop "${QE_REPO}" "${SRC_DIR}"
    else
        git clone --depth=1 --branch "qe-${QE_VERSION}" "${QE_REPO}" "${SRC_DIR}"
    fi
else
    echo "Source already present at ${SRC_DIR}, skipping clone."
fi

# ---- Configure with CMake ---------------------------------------------------
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
    "${SRC_DIR}"

echo ""
echo "--- CMake configure done ---"

# ---- Build ------------------------------------------------------------------
echo "--- Building QE with make -j${NCORES} ---"
make -j${NCORES} pw ph pp cp 2>&1 | tee "${BUILD_DIR}/build.log"

echo ""
echo "--- Build complete ---"

# ---- Install ----------------------------------------------------------------
echo "--- Installing to ${INSTALL_DIR} ---"
make install 2>&1 | tee "${BUILD_DIR}/install.log"

echo ""
echo "=========================================="
echo "Build finished: $(date)"
echo "Executables in: ${INSTALL_DIR}/bin/"
ls "${INSTALL_DIR}/bin/" 2>/dev/null || ls "${BUILD_DIR}/bin/"
echo "=========================================="
echo ""
echo "To use QE, load the same modules and add to PATH:"
echo "  module load PrgEnv-gnu/8.6.0 cray-fftw/3.3.10.9"
echo "  export PATH=${INSTALL_DIR}/bin:\$PATH"
