#!/bin/bash
#SBATCH -J build-qe-gpu
#SBATCH -p gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -C gpu
#SBATCH -o runs/build-qe-gpu-%j.out
#SBATCH -e runs/build-qe-gpu-%j.err

# =============================================================================
# Build Quantum ESPRESSO with NVIDIA A100 GPU support on Perlmutter
#
# Toolchain (aligned with HydraGNN Perlmutter installer):
#   PrgEnv-gnu/8.5.0, cpe/24.07, cray-mpich/8.1.30 ............ host toolchain + GPU-aware MPI
#   craype-accel-nvidia80 ..................................... A100 (sm_80) target
#   cudatoolkit/12.9 .......................................... CUDA libraries (matches PyTorch cu129)
#   gcc-native/13.2 ........................................... modern host C/C++ compiler
#   NVHPC 25.5 (nvfortran/nvc/nvc++) .......................... NVIDIA Fortran compiler bundled with CUDA 12.9
#   cray-fftw ................................................. CPU-side FFTW3
#   cray-libsci ............................................... BLAS/LAPACK (implicit via ftn wrapper)
#
# QE GPU offload is enabled with:
#     -DQE_GPU_ARCHS=sm_80       (A100 GPU architecture)
#     -DQE_ENABLE_OFFLOAD=ON     (enables CUDA-based GPU offloading)
#
# Advantages over Frontier (AMD MI250X):
#   - No internal compiler errors (ICE) to work around
#   - NVIDIA CUDA toolchain is mature and stable on Perlmutter
#   - Faster compilation (~20-30 min instead of 45+ min with workarounds)
#
# Where to run this:
#   CAN RUN ON LOGIN NODE or as a batch job. GPU not required for compilation
#   (CUDA cross-compile is available on login nodes).
#
# Usage:
#   # Login-node build (recommended — survives disconnect via nohup):
#   mkdir -p runs/build-qe-gpu-login
#   nohup bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh \
#         > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &
#
#   # Or as a batch job (sbatch headers above remain valid):
#   sbatch deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
#
# Use the develop branch or qe-7.4+; older versions may lack GPU support.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# ---- Configuration ----------------------------------------------------------
QE_VERSION="${QE_VERSION:-develop}"      # git tag (e.g. "7.4") or "develop"
QE_REPO="${QE_REPO:-https://gitlab.com/QEF/q-e.git}"

# Where QE source/build/install live. Default: under matsim-agents/external/
# (gitignored). Override with:
#   QE_PREFIX=/some/other/path bash deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
SRC_DIR="${SRC_DIR:-${QE_PREFIX}/src}"
BUILD_DIR="${BUILD_DIR:-${QE_PREFIX}/build-gpu}"
INSTALL_DIR="${INSTALL_DIR:-${QE_PREFIX}/install-gpu}"

# Compute parallelism for compilation. Perlmutter login nodes are shared,
# so default to modest count; raise via `NCORES=64 bash …` if on compute node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NCORES="${NCORES:-64}"   # dedicated compute node
else
  NCORES="${NCORES:-16}"   # shared login node
fi

# NVIDIA GPU architecture for Perlmutter A100
NVIDIA_GPU_ARCH="sm_80"

# ---- Create output directory ------------------------------------------------
mkdir -p "${PROJ}/runs/build-qe-gpu-${SLURM_JOB_ID:-login}" 2>/dev/null || true

echo "=========================================="
echo "Quantum ESPRESSO GPU (A100) build on Perlmutter"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "QE version:  ${QE_VERSION}"
echo "Source:      ${SRC_DIR}"
echo "Build dir:   ${BUILD_DIR}"
echo "Install:     ${INSTALL_DIR}"
echo "Target arch: ${NVIDIA_GPU_ARCH}"
echo "=========================================="

# ---- Load modules -----------------------------------------------------------
echo ""
echo "--- Loading Perlmutter NVIDIA compiler modules ---"

# Load the module stack for Perlmutter with NVIDIA compilers (nvfortran, nvc, nvc++)
source "${SCRIPT_DIR}/perlmutter-module-stack.sh" 2>/dev/null || {
  echo "ERROR: perlmutter-module-stack.sh not found"
  exit 1
}


# Call the function to load NVIDIA compilers
if declare -f load_perlmutter_modules_nvidia >/dev/null; then
  load_perlmutter_modules_nvidia
else
  echo "ERROR: load_perlmutter_modules_nvidia function not found in perlmutter-module-stack.sh"
  exit 1
fi

# Force NVHPC 25.5 (which bundles CUDA 12.9) so the QE GPU build links against
# the same CUDA runtime that HydraGNN's PyTorch (torch 2.11.0+cu129) uses.
# Mismatched CUDA majors between PyTorch and QE cause libcuda*.so symbol
# conflicts when both are loaded in the same job.
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
export CUDA_MATH_LIB_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib"
export NVFORTRAN_FLAGS="-cuda_home=${CUDA_HOME}"
# Ensure CUDA and CUDA math libraries are found by linker
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_MATH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_MATH_LIB_DIR}:${LIBRARY_PATH:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64 -L${CUDA_MATH_LIB_DIR} ${LDFLAGS:-}"

# Load FFTW (CPU-side, needed even for GPU build)
module load cray-fftw || echo "WARNING: cray-fftw not found"

# Verify NVIDIA toolchain
if ! command -v nvfortran &>/dev/null; then
  echo "ERROR: nvfortran not found. Check PrgEnv-nvidia is loaded."
  exit 1
fi

echo ""
echo "--- Loaded modules ---"
module list
echo ""
echo "--- Compiler versions ---"
nvfortran --version 2>&1 | head -1
nvc++ --version 2>&1 | head -1
nvc --version 2>&1 | head -1
echo "CUDA_HOME=${CUDA_HOME:-(unset)}"
echo "CRAY_FFTW_PREFIX=${CRAY_FFTW_PREFIX:-(unset)}"
echo ""

# ---- Clone or update source -------------------------------------------------
# NOTE: full clone (no --depth=1) so QE's GitInfo.cmake `git describe` succeeds
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
# Reuse cached objects when CLEAN_BUILD is empty; otherwise wipe build dir.
if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
  echo "Removing old build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "--- Running CMake configure ---"

# Ensure FFTW3 is available (should be from cray-fftw module)
if [[ -z "${FFTW3_ROOT:-}" ]] && [[ -z "${CRAY_FFTW_PREFIX:-}" ]]; then
  # Try to find it manually
  if [[ -d "/opt/cray/pe/fftw/3.3.10.11" ]]; then
    FFTW3_ROOT="/opt/cray/pe/fftw/3.3.10.11"
  elif [[ -d "/opt/cray/pe/fftw" ]]; then
    FFTW3_ROOT="$(find /opt/cray/pe/fftw -maxdepth 1 -type d | tail -1)"
  else
    echo "WARNING: FFTW3 location not found; CMake may fail"
    FFTW3_ROOT=""
  fi
fi

# Use FFTW3_ROOT if set, otherwise use CRAY_FFTW_PREFIX
FFTW_PATH="${FFTW3_ROOT:-${CRAY_FFTW_PREFIX:-}}"
FFTW_ARCH="${CRAY_CPU_TARGET:-x86_milan}"
FFTW_INCLUDE_DIR="${FFTW_PATH}/${FFTW_ARCH}/include"
FFTW_LIB_DIR="${FFTW_PATH}/${FFTW_ARCH}/lib"
if [[ ! -f "${FFTW_INCLUDE_DIR}/fftw3.f03" ]]; then
  FFTW_INCLUDE_DIR="${FFTW_PATH}/x86_milan/include"
  FFTW_LIB_DIR="${FFTW_PATH}/x86_milan/lib"
fi
echo "  FFTW3 path: ${FFTW_PATH:-not found}"
echo "  FFTW3 include: ${FFTW_INCLUDE_DIR}"
echo "  FFTW3 lib dir: ${FFTW_LIB_DIR}"
FFTW_LIBS="${FFTW_LIB_DIR}/libfftw3.so;${FFTW_LIB_DIR}/libfftw3_threads.so"

# Notes on the GPU-specific flags (QE develop API):
#   QE_GPU_ARCHS=sm_80    : selects NVIDIA A100. QE auto-derives QE_GPU=cuda,
#                           which enables CUDA offload and links libcufft, 
#                           libcublas, libcusolver.
#   QE_ENABLE_HDF5=OFF    : skip HDF5 for first GPU build (can be enabled if needed)
#   QE_ENABLE_LIBXC=OFF   : skip LibXC; functionals available via QE internal XC
#   QE_ENABLE_SCALAPACK=OFF : not needed for GPU offload
#   FFTW3_ROOT            : CPU-side FFTW3 headers for non-offloaded code paths

cmake \
    -DCMAKE_C_COMPILER=nvc \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_Fortran_COMPILER=nvfortran \
    \
    -DQE_ENABLE_MPI=ON \
    -DQE_ENABLE_MPI_MODULE=OFF \
    -DQE_ENABLE_OPENMP=ON \
    \
    -DQE_GPU_ARCHS="${NVIDIA_GPU_ARCH}" \
    \
    -DQE_ENABLE_SCALAPACK=OFF \
    -DQE_ENABLE_HDF5=OFF \
    -DQE_ENABLE_LIBXC=OFF \
    \
    -DQE_FFTW_VENDOR=FFTW3 \
  -DFFTW3_ROOT="${FFTW_PATH}" \
  -DFFTW3_INCLUDE_DIRS="${FFTW_INCLUDE_DIR}" \
  -DFFTW3_LIBRARIES="${FFTW_LIBS}" \
  -DCMAKE_Fortran_FLAGS="-I${FFTW_INCLUDE_DIR}" \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
  -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_HOME}/lib64 -L${CUDA_MATH_LIB_DIR} -lpthread" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_HOME}/lib64 -L${CUDA_MATH_LIB_DIR} -lpthread" \
    \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    \
    "${SRC_DIR}" 2>&1 | tee "${BUILD_DIR}/cmake.log"

echo ""
echo "--- CMake configure done ---"

# ---- Build ------------------------------------------------------------------
# On Perlmutter with NVIDIA toolchain, compilation is straightforward (no ICE workarounds needed).
# Build the main user-facing executables.
QE_TARGETS=(pw cp ph pp neb hp ld1 pwall pwcond tddfpt upf xspectra epw kcw all_currents)

echo "--- Building QE GPU targets: ${QE_TARGETS[*]} ---"
cd "${BUILD_DIR}"
make -j"${NCORES}" "${QE_TARGETS[@]}" 2>&1 | tee "${BUILD_DIR}/build.log"
MAKE_RC=$?

if (( MAKE_RC != 0 )); then
  echo "BUILD FAILED with rc=${MAKE_RC}"
  echo "Check build.log for details: ${BUILD_DIR}/build.log"
  exit ${MAKE_RC}
fi

echo ""
echo "--- Build complete ---"

# ---- Install ----------------------------------------------------------------
echo "--- Installing executables from ${BUILD_DIR}/bin to ${INSTALL_DIR}/bin ---"
mkdir -p "${INSTALL_DIR}/bin"
cp "${BUILD_DIR}"/bin/*.x "${INSTALL_DIR}/bin/" 2>/dev/null || {
  echo "WARNING: No executables found in ${BUILD_DIR}/bin"
  echo "Attempting to find and copy built binaries..."
  find "${BUILD_DIR}" -maxdepth 3 -name "*.x" -type f -exec cp {} "${INSTALL_DIR}/bin/" \; || true
}
echo "  installed $(ls "${INSTALL_DIR}/bin/" | wc -l) executables"

echo ""
echo "========================================="
echo "GPU build finished: $(date)"
echo "Executables in: ${INSTALL_DIR}/bin/"
ls "${INSTALL_DIR}/bin/" 2>/dev/null || echo "  (no executables found)"
echo ""

# ---- Verify GPU linkage for pw.x ----
if [[ -x "${INSTALL_DIR}/bin/pw.x" ]]; then
  echo "--- Verifying CUDA linkage of pw.x ---"
  ldd "${INSTALL_DIR}/bin/pw.x" 2>/dev/null | grep -iE "cuda|cublas|cufft|cusolver" | sed 's/^/  /' || echo "  (no CUDA libraries detected in ldd output)"
fi

echo "=========================================="
echo ""
echo "To run pw.x with GPU offload on a compute node:"
echo "  # 1. Load modules (same as build)"
echo "  source ${SCRIPT_DIR}/perlmutter-module-stack.sh"
echo "  load_perlmutter_modules_gpu"
echo ""
echo "  # 2. Set environment for GPU offload"
echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # if running on multiple GPUs"
echo ""
echo "  # 3. Run with appropriate parallelism (A100 = 108 cores, 1 GCD per GPU)"
echo "  srun -N1 -n8 -c14 --gpus-per-node=8 --gpu-bind=closest \\"
echo "       ${INSTALL_DIR}/bin/pw.x -in your-input.in"
echo ""
echo "For CPU-only runs (fallback if GPU unavailable):"
echo "  srun -N1 -n64 \\"
echo "       ${INSTALL_DIR}/bin/pw.x -in your-input.in"
