#!/usr/bin/env bash
#SBATCH -J build-vasp-gpu
#SBATCH -A mat746
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/build-vasp-gpu-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/build-vasp-gpu-%j/job-%j.err

# =============================================================================
# build-vasp-gpu-frontier.sh
#
# Build VASP 6.6 with OpenMP GPU offloading to AMD MI250X (gfx90a) on Frontier.
#
# Toolchain mirrors the matsim-agents Python venv module stack:
#   cpe/24.07  rocm/7.1.1  amd-mixed/7.1.1
# with these additions required for VASP GPU compilation:
#   PrgEnv-cray + cce ............... Cray Fortran compiler + OpenMP offload
#   craype-accel-amd-gfx90a ......... routes ftn/cc to gfx90a device code
#   cray-fftw ....................... FFTW3 CPU-side FFTs
# NOTE: PrgEnv-gnu (used by the Python venv) is replaced by PrgEnv-cray;
# PrgEnv-gnu does not support OpenMP target offload via the ftn wrapper.
#
# COMPILATION DOES NOT REQUIRE A GPU — the Cray+ROCm toolchain cross-compiles
# gfx90a device code without a physical MI250X present. Login-node builds
# (recommended) survive disconnects when launched via nohup.
#
# Usage
# -----
#   # Login-node build (recommended):
#   mkdir -p /lustre/orion/mat746/proj-shared/runs/build-vasp-gpu-login
#   nohup bash scripts/setup/frontier/build-vasp-gpu-frontier.sh \
#         > /lustre/orion/mat746/proj-shared/runs/build-vasp-gpu-login/build.log 2>&1 &
#
#   # Or as a SLURM batch job:
#   sbatch scripts/setup/frontier/build-vasp-gpu-frontier.sh
#
# Configurable environment overrides
# ------------------------------------
#   VASP_SRC_TGZ      Path to vasp.6.6.0.tgz    (default: external/vasp6/src/vasp.6.6.0.tgz)
#   VASP_ROOT         Extracted VASP source dir  (default: external/vasp6/src/vasp.6.6.0)
#   PREFIX            Build sub-directory name   (default: build)
#   NCORES            Parallel make jobs         (default: nproc or 16 on login)
#   CLEAN_BUILD       1=wipe build dirs first    (default: 0)
#   VASP_TARGET       all|std|gam|ncl            (default: all)
#   ROCM_MODULE       ROCm module to load        (default: rocm/7.1.1)
#   AMD_MIXED_MODULE  amd-mixed module to load   (default: amd-mixed/7.1.1)
#   VASP_HDF5_ROOT    Path to HDF5 install       (optional; enables -DVASP_HDF5)
#
# NOTE on ROCm version: rocm/7.1.1 matches the matsim-agents Python venv.
# If the linker reports cray-mpich GTL ABI errors (libamdhip64.so.6 vs .7),
# rebuild with:
#   ROCM_MODULE=rocm/6.2.4 AMD_MIXED_MODULE=amd-mixed/6.2.4 \
#     bash scripts/setup/frontier/build-vasp-gpu-frontier.sh
# =============================================================================

set -euo pipefail
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "ERROR: Could not detect matsim-agents repo root from ${SCRIPT_DIR}" >&2
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
VASP_SRC_TGZ="${VASP_SRC_TGZ:-${REPO}/external/vasp6/src/vasp.6.6.0.tgz}"
VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.0}"
PREFIX="${PREFIX:-build}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
VASP_TARGET="${VASP_TARGET:-all}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.1.1}"
AMD_MIXED_MODULE="${AMD_MIXED_MODULE:-amd-mixed/7.1.1}"
MAKEFILE_INCLUDE="${REPO}/external/vasp6/makefile.include.frontier-gpu"

# Parallel jobs: use SLURM allocation if available, else be a polite login-node
# neighbour (16 cores).  Override with NCORES=N.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    NCORES="${NCORES:-${SLURM_CPUS_ON_NODE:-64}}"
elif command -v nproc >/dev/null 2>&1; then
    RAW_NPROC="$(nproc)"
    NCORES="${NCORES:-$((RAW_NPROC < 16 ? RAW_NPROC : 16))}"
else
    NCORES="${NCORES:-16}"
fi

BUILD_TARGETS=()

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[vasp-frontier]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[vasp-frontier]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[vasp-frontier]\033[0m %s\n' "$*" >&2; exit 1; }

ensure_file() {
    local path="$1"
    [[ -e "${path}" ]] || die "Required path not found: ${path}"
}

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
    fi
    command -v module >/dev/null 2>&1 || die "module command not found"
}

# ── Print banner ──────────────────────────────────────────────────────────────
log "=========================================="
log "VASP GPU build on Frontier (AMD MI250X)"
log "Date:         $(date)"
log "Host:         $(hostname)"
log "VASP tarball: ${VASP_SRC_TGZ}"
log "VASP root:    ${VASP_ROOT}"
log "PREFIX:       ${PREFIX}"
log "Target:       ${VASP_TARGET}"
log "NCORES:       ${NCORES}"
log "ROCm module:  ${ROCM_MODULE}"
log "=========================================="

ensure_file "${VASP_SRC_TGZ}"
ensure_file "${MAKEFILE_INCLUDE}"

# ── Extract VASP tarball if needed ────────────────────────────────────────────
if [[ ! -d "${VASP_ROOT}" ]]; then
    log "Extracting ${VASP_SRC_TGZ} ..."
    mkdir -p "$(dirname "${VASP_ROOT}")"
    tar -xzf "${VASP_SRC_TGZ}" -C "$(dirname "${VASP_ROOT}")"
    [[ -d "${VASP_ROOT}" ]] || die "Extraction succeeded but ${VASP_ROOT} not found — check tarball layout"
    log "Extracted to ${VASP_ROOT}"
else
    log "VASP source already present at ${VASP_ROOT} — skipping extraction"
fi

ensure_file "${VASP_ROOT}/makefile"

# ── Install makefile.include ──────────────────────────────────────────────────
log "Installing makefile.include (Frontier GPU template) ..."
cp "${MAKEFILE_INCLUDE}" "${VASP_ROOT}/makefile.include"
log "  → ${VASP_ROOT}/makefile.include"

# ── Load modules ─────────────────────────────────────────────────────────────
#
# Module stack keeps the same cpe/ROCm versions as the matsim-agents Python
# venv (cpe/24.07, rocm/7.1.1, amd-mixed/7.1.1).  The only differences from
# the Python venv stack are:
#   • PrgEnv-cray replaces PrgEnv-gnu  (needed for ftn + OpenMP offload)
#   • cce is loaded explicitly         (pins Cray Fortran 18.0.1)
#   • craype-accel-amd-gfx90a added    (routes compilation to gfx90a device)
#   • cray-fftw added                  (FFTW3 headers/libs for VASP)
#   • miniforge3 not needed for compile
#
init_modules
module reset
module load cpe/24.07
module load PrgEnv-cray                 # Cray Fortran (ftn) + OpenMP target offload
module load cce                         # default 18.0.1 on Frontier
module load craype-accel-amd-gfx90a    # enables gfx90a OpenMP offload device code
module load "${ROCM_MODULE}"            # rocBLAS, rocFFT, rocSOLVER, RCCL, hipcc
module load "${AMD_MIXED_MODULE}"       # AMD mixed stack (matches Python venv)
module load cray-fftw                   # FFTW3 CPU-side FFTs
module unload darshan-runtime 2>/dev/null || true

log "Loaded modules:"
module list 2>&1 || true

# ── Verify key tools are present ─────────────────────────────────────────────
log "Compiler versions:"
ftn --version 2>&1 | head -1
cc  --version 2>&1 | head -1
"${ROCM_PATH:-/opt/rocm}/bin/hipcc" --version 2>&1 | head -2 || warn "hipcc not found at ROCM_PATH"

log "Environment:"
log "  ROCM_PATH        = ${ROCM_PATH:-<unset>}"
log "  CRAY_FFTW_PREFIX = ${CRAY_FFTW_PREFIX:-<unset>}"

[[ -n "${ROCM_PATH:-}" ]]        || die "ROCM_PATH is unset after module load — check ${ROCM_MODULE}"
[[ -n "${CRAY_FFTW_PREFIX:-}" ]] || die "CRAY_FFTW_PREFIX is unset after module load — check cray-fftw"

# ── Export env vars that makefile.include reads via ?= ────────────────────────
export ROCM_PATH
export FFTW_ROOT="${CRAY_FFTW_PREFIX}"
# HDF5 is optional — export VASP_HDF5_ROOT before calling this script to enable
if [[ -n "${VASP_HDF5_ROOT:-}" ]]; then
    log "HDF5 support enabled: VASP_HDF5_ROOT=${VASP_HDF5_ROOT}"
    export VASP_HDF5_ROOT
fi

# ── Determine build targets ───────────────────────────────────────────────────
case "${VASP_TARGET}" in
    all)         BUILD_TARGETS=(std gam ncl) ;;
    std|gam|ncl) BUILD_TARGETS=("${VASP_TARGET}") ;;
    *)           die "VASP_TARGET must be one of: all, std, gam, ncl" ;;
esac

cd "${VASP_ROOT}"

if [[ "${CLEAN_BUILD}" == "1" ]]; then
    log "Cleaning previous build directories under ${VASP_ROOT}/${PREFIX}"
    rm -rf "${VASP_ROOT}/${PREFIX}/std" \
           "${VASP_ROOT}/${PREFIX}/gam" \
           "${VASP_ROOT}/${PREFIX}/ncl"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
for target in "${BUILD_TARGETS[@]}"; do
    log "Building target '${target}' with -j${NCORES} ..."
    make PREFIX="${PREFIX}" DEPS=1 MODS=1 -j"${NCORES}" "${target}"
    log "Finished target '${target}'"
done

# ── Report built executables ──────────────────────────────────────────────────
log "=========================================="
log "Build finished: $(date)"
log "=========================================="
for exe in vasp_std vasp_gam vasp_ncl; do
    EXE_PATH="${VASP_ROOT}/bin/${exe}"
    if [[ -x "${EXE_PATH}" ]]; then
        log "  OK  ${EXE_PATH}"
    else
        warn "  MISSING  ${EXE_PATH}"
    fi
done
