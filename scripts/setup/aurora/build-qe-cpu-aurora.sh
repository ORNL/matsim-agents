#!/usr/bin/env bash
# =============================================================================
# build-qe-cpu-aurora.sh
#
# Build Quantum ESPRESSO (CPU-focused) on Aurora using the same style as the
# Frontier/Perlmutter QE setup scripts.
#
# What this script does:
#   1) Loads Aurora modules/toolchain
#   2) Clones/updates Quantum ESPRESSO source
#   3) Configures with CMake + MPI wrappers
#   4) Builds selected QE targets
#   5) Installs binaries to an install prefix
#
# Usage:
#   bash scripts/setup/aurora/build-qe-cpu-aurora.sh
#
# Optional overrides:
#   QE_VERSION=develop|7.4
#   QE_REPO=https://gitlab.com/QEF/q-e.git
#   QE_PREFIX=/path/to/quantum-espresso
#   SRC_DIR=/path/to/src
#   BUILD_DIR=/path/to/build-cpu
#   INSTALL_DIR=/path/to/install-cpu
#   NCORES=32
#   CLEAN_BUILD=1
#   C_COMPILER=mpicc
#   CXX_COMPILER=mpicxx
#   FORTRAN_COMPILER=mpifort
# =============================================================================

set -euo pipefail
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "ERROR: Could not detect matsim-agents repo root from ${SCRIPT_DIR}" >&2
    exit 1
fi

PROJ="$(dirname "${REPO}")"

QE_VERSION="${QE_VERSION:-develop}"
QE_REPO="${QE_REPO:-https://gitlab.com/QEF/q-e.git}"
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
SRC_DIR="${SRC_DIR:-${QE_PREFIX}/src}"
BUILD_DIR="${BUILD_DIR:-${QE_PREFIX}/build-cpu}"
INSTALL_DIR="${INSTALL_DIR:-${QE_PREFIX}/install-cpu}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

if [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    NCORES="${NCORES:-${SLURM_CPUS_ON_NODE}}"
elif command -v nproc >/dev/null 2>&1; then
    NCORES="${NCORES:-$(nproc)}"
else
    NCORES="${NCORES:-8}"
fi

C_COMPILER="${C_COMPILER:-mpicc}"
CXX_COMPILER="${CXX_COMPILER:-mpicxx}"
FORTRAN_COMPILER="${FORTRAN_COMPILER:-mpifort}"

log()  { printf '\033[1;34m[qe-aurora]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[qe-aurora]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[qe-aurora]\033[0m %s\n' "$*" >&2; exit 1; }

init_modules() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi

    command -v module >/dev/null 2>&1 || die "module command not found"
}

load_aurora_modules() {
    init_modules
    module reset

    # Aurora frameworks module provides MPI toolchain and Python stack.
    module load frameworks
    module load cmake >/dev/null 2>&1 || true
    module load git >/dev/null 2>&1 || true
}

ensure_tool() {
    local tool="$1"
    command -v "$tool" >/dev/null 2>&1 || die "Required tool not found: $tool"
}

log "=========================================="
log "Quantum ESPRESSO CPU build on Aurora"
log "Date:       $(date)"
log "Host:       $(hostname)"
log "QE version: ${QE_VERSION}"
log "Source:     ${SRC_DIR}"
log "Build dir:  ${BUILD_DIR}"
log "Install:    ${INSTALL_DIR}"
log "NCORES:     ${NCORES}"
log "=========================================="

load_aurora_modules

log "Loaded modules:"
module list 2>&1 || true

ensure_tool git
ensure_tool cmake
ensure_tool "$C_COMPILER"
ensure_tool "$CXX_COMPILER"
ensure_tool "$FORTRAN_COMPILER"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    log "Cloning QE ${QE_VERSION} from ${QE_REPO}"
    mkdir -p "$(dirname "${SRC_DIR}")"
    if [[ "${QE_VERSION}" == "develop" ]]; then
        git clone --branch develop "${QE_REPO}" "${SRC_DIR}"
    else
        git clone --branch "qe-${QE_VERSION}" "${QE_REPO}" "${SRC_DIR}"
    fi
else
    log "QE source already present at ${SRC_DIR}"
    log "HEAD: $(cd "${SRC_DIR}" && git rev-parse --short HEAD)"
fi

if [[ "${CLEAN_BUILD}" == "1" ]]; then
    log "Removing previous build directory ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

log "Configuring with CMake"
cmake \
    -DCMAKE_C_COMPILER="${C_COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DCMAKE_Fortran_COMPILER="${FORTRAN_COMPILER}" \
    \
    -DQE_ENABLE_MPI=ON \
    -DQE_ENABLE_MPI_MODULE=OFF \
    -DQE_ENABLE_OPENMP=ON \
    \
    -DQE_ENABLE_SCALAPACK=OFF \
    -DQE_ENABLE_HDF5=OFF \
    -DQE_ENABLE_LIBXC=OFF \
    \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    \
    "${SRC_DIR}" 2>&1 | tee "${BUILD_DIR}/cmake.log"

QE_TARGETS=(pw cp ph pp neb hp ld1)

log "Building targets: ${QE_TARGETS[*]}"
make -j"${NCORES}" "${QE_TARGETS[@]}" 2>&1 | tee "${BUILD_DIR}/build.log"

log "Installing"
if make install 2>&1 | tee "${BUILD_DIR}/install.log"; then
    :
else
    warn "make install failed; attempting fallback binary copy"
    mkdir -p "${INSTALL_DIR}/bin"
    cp "${BUILD_DIR}"/bin/*.x "${INSTALL_DIR}/bin/" 2>/dev/null || true
fi

log "=========================================="
log "Build finished: $(date)"
log "Installed binaries:"
ls "${INSTALL_DIR}/bin" 2>/dev/null || ls "${BUILD_DIR}/bin" 2>/dev/null || warn "No QE binaries found"
log "=========================================="

log "To use this build:"
log "  export PATH=${INSTALL_DIR}/bin:\$PATH"
