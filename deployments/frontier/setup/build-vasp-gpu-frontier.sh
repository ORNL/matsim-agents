#!/usr/bin/env bash
#SBATCH -J build-vasp-gpu
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# =============================================================================
# build-vasp-gpu-frontier.sh
#
# Prefer OLCF's own pre-built VASP module when one is available and working;
# only build VASP 6.6 from source (OpenMP GPU offload to AMD MI250X / gfx90a)
# as a fallback when the facility module is missing or fails a smoke test.
# See VASP_FACILITY_MODULE / VASP_SKIP_FACILITY_MODULE below.
#
# Toolchain mirrors the matsim-agents Python venv module stack:
#   cpe/24.07  rocm/6.2.0  amd-mixed/6.2.0
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
#   mkdir -p /path/to/project/runs/build-vasp-gpu-login
#   nohup bash deployments/frontier/setup/build-vasp-gpu-frontier.sh \
#         > /path/to/project/runs/build-vasp-gpu-login/build.log 2>&1 &
#
#   # Or as a SLURM batch job:
#   sbatch deployments/frontier/setup/build-vasp-gpu-frontier.sh
#
# Configurable environment overrides
# ------------------------------------
#   VASP_FACILITY_MODULE      OLCF module to try first   (default: vasp/6.6.1-gpu)
#   VASP_SKIP_FACILITY_MODULE 1=skip module, build from source (default: 0)
#   VASP_SRC_TGZ      Path to vasp.6.6.1.tgz    (default: external/vasp6/src/vasp.6.6.1.tgz)
#   VASP_ROOT         Extracted VASP source dir  (default: external/vasp6/src/vasp.6.6.1)
#   PREFIX            Build sub-directory name   (default: build)
#   NCORES            Parallel make jobs         (default: nproc or 16 on login)
#   CLEAN_BUILD       1=wipe build dirs first    (default: 0)
#   VASP_TARGET       all|std|gam|ncl            (default: all)
#   ROCM_MODULE       ROCm module to load        (default: rocm/6.2.0)
#   AMD_MIXED_MODULE  amd-mixed module to load   (default: amd-mixed/6.2.0)
#   VASP_HDF5_ROOT    Path to HDF5 install       (optional; enables -DVASP_HDF5)
#
# If OLCF's vasp/6.6.1-gpu module is present and passes a smoke test, this
# script uses it directly and exits WITHOUT building anything (see
# _vasp-step-frontier.sh, which loads the same module at runtime instead of
# the manual PrgEnv-cray/rocm/6.2.0 stack below). Set
# VASP_SKIP_FACILITY_MODULE=1 to always build from source regardless.
#
# NOTE on ROCm version: the GPU device link runs cce's llvm-link + lld (LLVM 18
# for cce 18.x) over ROCm's device bitcode (amdgcn/bitcode/*.bc).  Two
# constraints pin the ROCm version:
#   1. The bitcode must be produced by LLVM <= 18, else llvm-link fails with
#      "Invalid attribute group entry (Producer 'LLVM20' Reader 'LLVM 18')".
#      -> rules out ROCm 6.4+ (LLVM 19) and 7.x (LLVM 20).
#   2. The device libs must use the NON-overloaded llvm.amdgcn.readfirstlane
#      intrinsic; ROCm 6.2.4+ backported the typed .i32 form (an LLVM-19
#      feature) into their 18.0.0git fork, which cce 18's lld cannot lower
#      ("undefined symbol: llvm.amdgcn.readfirstlane.i32").
#      -> rules out ROCm 6.2.4 and 6.3.1.
# rocm/6.2.0 is the newest ROCm satisfying BOTH (LLVM 18 bitcode + old
# intrinsic).  rocm/6.1.3 (LLVM 17) is a fallback.
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
VASP_FACILITY_MODULE="${VASP_FACILITY_MODULE:-vasp/6.6.1-gpu}"
VASP_SKIP_FACILITY_MODULE="${VASP_SKIP_FACILITY_MODULE:-0}"
VASP_SRC_TGZ="${VASP_SRC_TGZ:-${REPO}/external/vasp6/src/vasp.6.6.1.tgz}"
VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.1}"
PREFIX="${PREFIX:-build}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
VASP_TARGET="${VASP_TARGET:-all}"
# MODS=1 runs VASP's bulk `mods` pass: a single `ftn -homp -c <all sources>` whose
# objects the link step then reuses.  Under cce 18.x that bulk compile aborts with
# ftn-7032 on vdw_nl.f90 ("Unsupported OpenMP construct Calls") and bypasses the
# per-file OBJECTS_O*/-hnoomp workarounds.  MODS=0 skips it so each source is
# compiled individually via the O-group rules (in .depend order), which lets the
# workarounds -- including vdw_nl -> -hnoomp -- take effect.
VASP_MODS="${VASP_MODS:-0}"
ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.0}"
AMD_MIXED_MODULE="${AMD_MIXED_MODULE:-amd-mixed/6.2.0}"
# Tracked next to this script (external/ is gitignored, so the recipe lives here).
MAKEFILE_INCLUDE="${MAKEFILE_INCLUDE:-${SCRIPT_DIR}/makefile.include.frontier-gpu}"

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

# Returns 0 and sets FACILITY_BIN_DIR if the OLCF-provided module "$1" exists,
# loads cleanly, ships all three executables, and passes a smoke test.
try_facility_module() {
    local mod="$1" bindir out smoke_dir smoke_log
    init_modules
    if ! module avail "${mod}" 2>&1 | grep -qF "${mod}"; then
        warn "Facility module '${mod}' not found via 'module avail' — will build from source"
        return 1
    fi
    # Load in a subshell: never let a failed/partial load pollute this
    # script's environment before the from-source build's own `module reset`.
    if ! out="$(
        set -e
        module reset >/dev/null 2>&1 || true
        module load "${mod}" >/dev/null 2>&1
        command -v vasp_std
    )"; then
        warn "Failed to load facility module '${mod}' — will build from source"
        return 1
    fi
    bindir="$(dirname "${out}")"
    for exe in vasp_std vasp_gam vasp_ncl; do
        [[ -x "${bindir}/${exe}" ]] || { warn "Facility module '${mod}' missing/non-executable ${exe} — will build from source"; return 1; }
    done
    # Smoke test: a working install prints VASP's "I REFUSE TO CONTINUE"
    # banner and exits when no INCAR is present; a broken one (bad dynamic
    # linking, wrong GPU arch, etc.) errors out before reaching that banner.
    smoke_dir="$(mktemp -d)"
    smoke_log="${smoke_dir}/smoke.log"
    (
        module reset >/dev/null 2>&1 || true
        module load "${mod}" >/dev/null 2>&1
        cd "${smoke_dir}"
        timeout 30 "${bindir}/vasp_std"
    ) > "${smoke_log}" 2>&1 || true
    if ! grep -q "I REFUSE TO CONTINUE" "${smoke_log}" 2>/dev/null; then
        warn "Facility module '${mod}' smoke test failed — see ${smoke_log} — will build from source"
        rm -rf "${smoke_dir}"
        return 1
    fi
    rm -rf "${smoke_dir}"
    FACILITY_BIN_DIR="${bindir}"
    return 0
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

# ── Prefer OLCF's own pre-built VASP module ──────────────────────────────────
FACILITY_BIN_DIR=""
if [[ "${VASP_SKIP_FACILITY_MODULE}" == "1" ]]; then
    log "VASP_SKIP_FACILITY_MODULE=1 — skipping facility-module check, building from source."
elif try_facility_module "${VASP_FACILITY_MODULE}"; then
    log "=========================================="
    log "Using OLCF-provided VASP module: ${VASP_FACILITY_MODULE}"
    for exe in vasp_std vasp_gam vasp_ncl; do
        log "  OK  ${FACILITY_BIN_DIR}/${exe}"
    done
    log "Point VASP_BIN at ${FACILITY_BIN_DIR}/vasp_std (or vasp_gam/vasp_ncl)."
    log "Runtime module stack: 'module load ${VASP_FACILITY_MODULE}' (see _vasp-step-frontier.sh)."
    log "To force a from-source build instead, re-run with VASP_SKIP_FACILITY_MODULE=1."
    log "=========================================="
    exit 0
else
    log "Facility module unavailable/unusable — building VASP from source instead."
fi

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
#   • cce is loaded explicitly         (Cray Fortran 18.x; the only cce with
#                                       working gfx90a offload on this system --
#                                       cce >=19 lacks libopenacc.pc for craype)
#   • craype-accel-amd-gfx90a added    (routes compilation to gfx90a device)
#   • cray-fftw added                  (FFTW3 headers/libs for VASP)
#   • miniforge3 not needed for compile
#
init_modules
module reset
module load cpe/24.07
module load PrgEnv-cray                 # Cray Fortran (ftn) + OpenMP target offload
module load cce                         # cce 18.x (only version with working gfx90a offload)
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
    # VASP's makefile uses `ifdef MODS` (tests defined-ness, not value), so MODS
    # must be OMITTED -- not set to 0 -- to skip the bulk `mods` pass.
    mods_arg=()
    [[ "${VASP_MODS}" == "1" ]] && mods_arg=(MODS=1)
    log "Building target '${target}' with -j${NCORES} (MODS=${VASP_MODS}) ..."
    make PREFIX="${PREFIX}" DEPS=1 "${mods_arg[@]}" -j"${NCORES}" "${target}"
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
