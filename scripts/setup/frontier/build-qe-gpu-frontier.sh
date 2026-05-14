#!/bin/bash
#SBATCH -J build-qe-gpu
#SBATCH -A mat746
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/build-qe-gpu-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/build-qe-gpu-%j/job-%j.err

# =============================================================================
# Build Quantum ESPRESSO with AMD MI250X (gfx90a) OpenMP target offload
# on Frontier.
#
# Toolchain:
#   PrgEnv-cray + cce  ............... Cray Fortran/C/C++ compilers
#   craype-accel-amd-gfx90a .......... enables OpenMP target offload to MI250X
#   rocm  ............................ rocFFT, rocBLAS, rocSOLVER
#   cray-fftw  ....................... CPU-side FFTW3 (fallback when offload off)
#   cray-libsci  ..................... CPU BLAS/LAPACK (linked implicitly)
#   cray-mpich  ..................... GPU-aware MPI (loaded by PrgEnv-cray)
#
# QE GPU offload is enabled with the develop-branch CMake flag
#     -DQE_ENABLE_OFFLOAD=ON
# which targets pwscf hot kernels (FFT batches, dgemm calls, eigensolve)
# through the OpenMP 5.x `target` directive. ROCm libraries are picked up
# via the cray wrappers when craype-accel-amd-gfx90a is loaded.
#
# Where to run this:
#   COMPILATION DOES NOT REQUIRE A GPU. The Cray + ROCm toolchain is fully
#   available on Frontier login nodes and cross-compiles gfx90a device code
#   without an MI250X being present. Login-node build is the recommended path.
#
# Usage:
#   # Login-node build (recommended) — survives disconnect via nohup:
#   mkdir -p runs/build-qe-gpu-login
#   nohup bash scripts/setup/frontier/build-qe-gpu-frontier.sh \
#         > runs/build-qe-gpu-login/build.log 2>&1 &
#
#   # Or as a batch job (sbatch headers below remain valid):
#   sbatch scripts/setup/frontier/build-qe-gpu-frontier.sh
#
# Use the develop branch (or qe-7.4+); older QE lacks QE_ENABLE_OFFLOAD.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=/lustre/orion/mat746/proj-shared/matsim-agents
PROJ="$(dirname "${REPO}")"

# ---- Configuration ----------------------------------------------------------
QE_VERSION="${QE_VERSION:-develop}"      # git tag (e.g. "7.4") or "develop"
QE_REPO="${QE_REPO:-https://gitlab.com/QEF/q-e.git}"
BASE_DIR="${PROJ}"
SRC_DIR="${BASE_DIR}/quantum-espresso/src"
BUILD_DIR="${BASE_DIR}/quantum-espresso/build-gpu"
INSTALL_DIR="${BASE_DIR}/quantum-espresso/install-gpu"

# Compute parallelism for compilation. Frontier login nodes are shared,
# so default to a modest count; raise via `NCORES=64 bash …` if needed.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NCORES="${NCORES:-64}"   # dedicated compute node
else
  NCORES="${NCORES:-16}"   # shared login node — be a good neighbour
fi

# AMD GPU architecture for Frontier MI250X
AMDGPU_TARGETS="gfx90a"

# ---- Create output directory (used for sbatch log files) --------------------
mkdir -p "${PROJ}/runs/build-qe-gpu-${SLURM_JOB_ID:-login}" 2>/dev/null || true

echo "=========================================="
echo "Quantum ESPRESSO GPU (gfx90a) build on Frontier"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "QE version:  ${QE_VERSION}"
echo "Source:      ${SRC_DIR}"
echo "Build dir:   ${BUILD_DIR}"
echo "Install:     ${INSTALL_DIR}"
echo "Target arch: ${AMDGPU_TARGETS}"
echo "=========================================="

# ---- Load modules -----------------------------------------------------------
module reset

# Switch to PrgEnv-cray (needed for OpenMP target offload to AMD GPUs).
# Pin cce to 21.0.0 — the default cce/18.0.1 hits an internal compiler error
# (ftn-7991 in /workspace/crayftn/pdgcs/v_fei.c) on PW/src/gen_at_d{j,y}.f90
# when -target-accel=amd_gfx90a is in effect. cce 21 fixes it.
module load PrgEnv-cray
module load cce/21.0.0
module load craype-accel-amd-gfx90a      # enables -fopenmp offload to gfx90a
module load rocm                         # rocFFT, rocBLAS, rocSOLVER
module load cray-fftw                    # CPU-side FFTW3 headers
module load cmake/3.30.5
module load git/2.47.0

# QE GPU build needs unbuffered HIP and host-callable rocFFT/rocBLAS.
# Cray's compiler wrapper picks these up when craype-accel-amd-gfx90a is loaded;
# we still export ROCM_PATH explicitly so CMake's FindHIP can locate them.
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-${ROCM_VERSION:-default}}"

echo ""
echo "--- Loaded modules ---"
module list
echo "--- Compiler versions ---"
ftn --version 2>&1 | head -1
cc  --version 2>&1 | head -1
CC  --version 2>&1 | head -1
echo "ROCM_PATH       = ${ROCM_PATH}"
echo "CRAY_FFTW_PREFIX= ${CRAY_FFTW_PREFIX:-unset}"
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
    echo "  HEAD = $(cd "${SRC_DIR}" && git rev-parse --short HEAD)"
fi

# ---- Configure with CMake ---------------------------------------------------
# Use a clean build dir to avoid stale CPU-build cache files.
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "--- Running CMake configure ---"

# Notes on the GPU-specific flags (QE develop API, post-2025 rename):
#   QE_GPU_ARCHS=gfx90a   : selects MI250X. QE auto-derives QE_GPU="openmp;rocm",
#                            which enables OpenMP target offload AND links
#                            rocFFT/rocBLAS/rocSOLVER for hot kernels.
#   QE_ENABLE_HDF5=OFF    : skip HDF5 to avoid cray-hdf5 module clashes;
#                           flip to ON if needed.
#   QE_ENABLE_LIBXC=OFF   : skip LibXC for the first GPU build; functionals are
#                           still available via QE's internal XC modules.
#   QE_ENABLE_SCALAPACK=OFF: ScaLAPACK does not have a GPU-aware path here; rely
#                            on QE's internal LAXlib offload kernels.
#   FFTW3_ROOT            : CPU FFTW3 headers \u2014 QE still uses these for the
#                           non-offloaded code paths.

cmake \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_Fortran_COMPILER=ftn \
    \
    -DQE_ENABLE_MPI=ON \
    -DQE_ENABLE_MPI_MODULE=OFF \
    -DQE_ENABLE_OPENMP=ON \
    \
    -DQE_GPU_ARCHS=gfx90a \
    \
    -DQE_ENABLE_SCALAPACK=OFF \
    -DQE_ENABLE_HDF5=OFF \
    -DQE_ENABLE_LIBXC=OFF \
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
# Build the targets that benefit most from GPU offload first; pp/cp can be
# added once the pw/ph build is verified.
echo "--- Building QE GPU targets with make -j${NCORES} ---"
make -j"${NCORES}" pw ph pp 2>&1 | tee "${BUILD_DIR}/build.log"

echo ""
echo "--- Build complete ---"

# ---- Install ----------------------------------------------------------------
echo "--- Installing to ${INSTALL_DIR} ---"
make install 2>&1 | tee "${BUILD_DIR}/install.log"

echo ""
echo "=========================================="
echo "GPU build finished: $(date)"
echo "Executables in: ${INSTALL_DIR}/bin/"
ls "${INSTALL_DIR}/bin/" 2>/dev/null || ls "${BUILD_DIR}/bin/"
echo "=========================================="
echo ""
echo "To run pw.x with GPU offload, on a compute node:"
echo "  module load PrgEnv-cray cce craype-accel-amd-gfx90a rocm cray-fftw"
echo "  export OMP_NUM_THREADS=7        # 7 cores per GCD on Frontier"
echo "  export OMP_TARGET_OFFLOAD=MANDATORY   # fail loudly if offload misroutes"
echo "  export MPICH_GPU_SUPPORT_ENABLED=1"
echo "  srun -N1 -n8 -c7 --gpus-per-node=8 --gpu-bind=closest \\"
echo "       ${INSTALL_DIR}/bin/pw.x -in your-input.in"
