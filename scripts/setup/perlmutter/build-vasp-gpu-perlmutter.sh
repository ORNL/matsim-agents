#!/usr/bin/env bash
# =============================================================================
# build-vasp-gpu-perlmutter.sh
#
# Build VASP 6.6.0 on NERSC Perlmutter (NVIDIA A100, sm_80) using the OpenACC
# GPU offload port driven by NVHPC 25.5 (CUDA 12.9), cray-mpich, cray-libsci
# (BLAS/LAPACK/scaLAPACK) and cray-fftw.
#
# Source layout convention (see README):
#   ${REPO}/external/vasp6/src/vasp.6.6.0/...
#
# Usage:
#   bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh
#
# Optional overrides:
#   VASP_ROOT=/path/to/vasp.6.6.0
#   PREFIX=build
#   NCORES=16
#   CLEAN_BUILD=0|1
#   VASP_TARGET=all|std|gam|ncl
#   REGENERATE_MAKEFILE=0|1   # force-rewrite makefile.include even if present
#   GPU_ARCH=cc80             # A100=cc80; H100=cc90; L40=cc89
#   CUDA_VER=12.9             # must match NVHPC 25.5 bundled CUDA
# =============================================================================

set -euo pipefail
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "ERROR: Could not detect matsim-agents repo root from ${SCRIPT_DIR}" >&2
    exit 1
fi

VASP_ROOT="${VASP_ROOT:-${REPO}/external/vasp6/src/vasp.6.6.0}"
PREFIX="${PREFIX:-build}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
VASP_TARGET="${VASP_TARGET:-all}"
REGENERATE_MAKEFILE="${REGENERATE_MAKEFILE:-0}"
GPU_ARCH="${GPU_ARCH:-cc80}"
CUDA_VER="${CUDA_VER:-12.9}"
# scaLAPACK from netlib (built by build-scalapack-perlmutter.sh). Required
# for multi-node runs (parallel diagonalization). Set SCALAPACK_ROOT="" to
# disable and fall back to a single-node build with -DnoSCALAPACK.
SCALAPACK_ROOT="${SCALAPACK_ROOT-${REPO}/external/scalapack/install}"
SCALAPACK_AUTOBUILD="${SCALAPACK_AUTOBUILD:-1}"
BUILD_TARGETS=()

if [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    NCORES="${NCORES:-${SLURM_CPUS_ON_NODE}}"
elif command -v nproc >/dev/null 2>&1; then
    NCORES="${NCORES:-$(nproc)}"
else
    NCORES="${NCORES:-8}"
fi

log()  { printf '\033[1;34m[vasp-pm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[vasp-pm]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[vasp-pm]\033[0m %s\n' "$*" >&2; exit 1; }

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

load_perlmutter_nvhpc_modules() {
    init_modules

    # Reuse the QE-aligned helper: PrgEnv-gnu + cpe/24.07 + cray-mpich/8.1.30
    # + cudatoolkit/12.9 + cray-fftw + nvfortran on PATH (NVHPC 25.5 SDK).
    # PrgEnv-nvhpc/8.5.0 is *not* used because it only pairs with cpe/25.09
    # (cray-mpich/9.0.1), which would break ABI alignment with the matsim venv.
    local stack="${REPO}/scripts/setup/perlmutter/perlmutter-module-stack.sh"
    [[ -f "${stack}" ]] || die "perlmutter-module-stack.sh not found at ${stack}"
    # shellcheck disable=SC1090
    source "${stack}"
    load_perlmutter_modules_nvidia

    # cmake is needed only by some companion projects, not by VASP's classic make.
    ml cmake 2>/dev/null || true

    # NVHPC 25.5 ships qd (quadruple-precision emulation) under
    # ${NVROOT}/compilers/extras/qd, which the makefile.include resolves at
    # build-time via NVROOT=$(which nvfortran ...). Make sure nvfortran is on PATH.
    if ! command -v nvfortran >/dev/null 2>&1; then
        export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin:${PATH}"
    fi
    command -v nvfortran >/dev/null 2>&1 \
        || die "nvfortran not found on PATH after module loads"
    command -v nvc >/dev/null 2>&1 \
        || die "nvc not found on PATH (NVHPC compiler set incomplete)"

    log "Module environment:"
    module list 2>&1 | sed 's/^/  /'
    log "Compilers:"
    log "  nvfortran : $(command -v nvfortran)"
    log "  nvc       : $(command -v nvc)"
    log "  nvc++     : $(command -v nvc++)"
    log "  MPICH_DIR : ${MPICH_DIR:-<unset>}"
    log "  FFTW_ROOT : ${FFTW_ROOT:-<unset>}"
}

write_makefile_include() {
    local out="${VASP_ROOT}/makefile.include"
    if [[ -f "${out}" && "${REGENERATE_MAKEFILE}" != "1" ]]; then
        log "makefile.include already present (set REGENERATE_MAKEFILE=1 to overwrite)"
        return 0
    fi

    log "Generating makefile.include for Perlmutter (GPU=${GPU_ARCH}, CUDA=${CUDA_VER})"

    # Resolve cray module-provided paths (must be set before we expand the heredoc).
    local mpich_dir="${MPICH_DIR:-}"
    local fftw_root="${FFTW_ROOT:-${CRAY_FFTW_PREFIX:-}}"
    local libsci_prefix="${CRAY_LIBSCI_PREFIX:-}"
    [[ -n "${mpich_dir}"     ]] || die "MPICH_DIR not set; cray-mpich module not loaded"
    [[ -n "${fftw_root}"     ]] || die "FFTW_ROOT not set; cray-fftw module not loaded"
    [[ -n "${libsci_prefix}" ]] || die "CRAY_LIBSCI_PREFIX not set; cray-libsci module not loaded"

    # cray-mpich GTL (GPU transport layer) for nvidia80 (A100). Only present
    # in cray-mpich 8.1.x; the path lives at <cray-mpich-root>/gtl/lib.
    local mpich_root="${CRAY_MPICH_ROOTDIR:-${MPICH_DIR%/ofi/*}}"
    local gtl_dir="${mpich_root}/gtl/lib"
    [[ -d "${gtl_dir}" ]] || warn "cray-mpich GTL dir not found at ${gtl_dir}; GPU-aware MPI link may fail"

    # cray-mpich ships per-compiler subtrees (gnu/, nvidia/, intel/, ...). Each
    # has Fortran .mod files compiled with that vendor's compiler. PrgEnv-gnu
    # selects the gnu/12.3 variant by default, but its mpi.mod is GFortran-
    # built and unreadable by nvfortran ("Corrupt or Old Module file").
    # Switch to the nvidia subdir (latest) which provides nvfortran-compatible
    # mpi.mod plus libmpi.so / libmpifort.so with the same C ABI.
    local mpich_nvidia
    mpich_nvidia="$(ls -d ${mpich_root}/ofi/nvidia/*/include/mpi.mod 2>/dev/null | sort -V | tail -1)"
    if [[ -n "${mpich_nvidia}" && -f "${mpich_nvidia}" ]]; then
        mpich_dir="${mpich_nvidia%/include/mpi.mod}"
        log "  Switching MPICH_DIR to nvfortran-compatible variant: ${mpich_dir}"
    else
        warn "No nvidia-flavored cray-mpich found; nvfortran will likely reject mpi.mod"
    fi

    # Pick libsci variant compatible with NVHPC compilers (NVIDIA build of
    # cray-libsci). cray-libsci 25.09 does NOT ship scaLAPACK under NVIDIA,
    # so we build with -DnoSCALAPACK (single-node multi-GPU still works;
    # multi-node parallel diag is disabled).
    local libsci_nvidia_root="${libsci_prefix%/GNU/*}"
    libsci_nvidia_root="${libsci_nvidia_root%/CRAY/*}"
    libsci_nvidia_root="${libsci_nvidia_root%/INTEL/*}"
    libsci_nvidia_root="${libsci_nvidia_root%/AOCC/*}"
    local libsci_nvidia
    libsci_nvidia="$(ls -d ${libsci_nvidia_root}/NVIDIA/*/x86_64 2>/dev/null | sort -V | tail -1)"
    [[ -n "${libsci_nvidia}" && -d "${libsci_nvidia}" ]] \
        || die "NVIDIA variant of cray-libsci not found under ${libsci_nvidia_root}/NVIDIA/"

    log "makefile.include paths:"
    log "  MPICH_DIR     = ${mpich_dir}"
    log "  GTL dir       = ${gtl_dir}"
    log "  FFTW_ROOT     = ${fftw_root}"
    log "  libsci NVIDIA = ${libsci_nvidia}"

    # Resolve scaLAPACK (required for multi-node parallel diag).
    local sca_def="-DnoSCALAPACK"
    local sca_lib=""
    local sca_inc=""
    if [[ -n "${SCALAPACK_ROOT}" ]]; then
        local sca_lib_path=""
        for cand in "${SCALAPACK_ROOT}/lib/libscalapack.a" "${SCALAPACK_ROOT}/lib64/libscalapack.a"; do
            [[ -f "${cand}" ]] && { sca_lib_path="${cand}"; break; }
        done
        if [[ -n "${sca_lib_path}" ]]; then
            sca_def="-DscaLAPACK"
            sca_lib="${sca_lib_path}"
            log "  scaLAPACK    = ${sca_lib_path}  (multi-node enabled)"
        else
            warn "scaLAPACK not found under ${SCALAPACK_ROOT} -> single-node build (-DnoSCALAPACK)"
            warn "  build it first: bash scripts/setup/perlmutter/build-scalapack-perlmutter.sh"
        fi
    else
        log "  scaLAPACK    = <disabled>  (single-node only)"
    fi

    cat > "${out}" <<EOF
# =============================================================================
# makefile.include for VASP 6.6.0 on NERSC Perlmutter (A100, NVHPC OpenACC)
# Auto-generated by scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh
# Toolchain: PrgEnv-gnu + nvfortran/nvc/nvc++ (NVHPC 25.5)
#            + cray-mpich/8.1.30 (GPU-aware via libmpi_gtl_cuda)
#            + cray-libsci/25.09 (NVIDIA variant, no scaLAPACK -> -DnoSCALAPACK)
#            + cray-fftw/3.3.10.11 + CUDA ${CUDA_VER}
# =============================================================================

CPP_OPTIONS = -DHOST=\\"LinuxNV\\" \\
              -DMPI -DMPI_INPLACE -DMPI_BLOCK=8000 -Duse_collective \\
              ${sca_def} \\
              -DCACHE_SIZE=4000 \\
              -Davoidalloc \\
              -Dvasp6 \\
              -Dtbdyn \\
              -Dqd_emulate \\
              -Dfock_dblbuf \\
              -D_OPENMP \\
              -DACC_OFFLOAD \\
              -DNVCUDA \\
              -DUSENCCL

CPP         = nvfortran -Mpreprocess -Mfree -Mextend -E \$(CPP_OPTIONS) \$*\$(FUFFIX)  > \$*\$(SUFFIX)

# A100 = cc80, H100 = cc90; CUDA bundled with NVHPC 25.5 = 12.9
GPU         = -gpu=${GPU_ARCH},cuda${CUDA_VER}

# Direct NVHPC compiler invocation (we do NOT use Cray ftn/cc wrappers because
# PrgEnv-nvhpc requires cpe/25.09 / cray-mpich/9.0.1 which would break ABI
# alignment with the matsim venv).
CC          = nvc       -mp -acc \$(GPU)
FC          = nvfortran -mp -acc \$(GPU)
FCL         = nvfortran -mp -acc \$(GPU) -c++libs

FREE        = -Mfree
FFLAGS      = -Mbackslash -Mlarge_arrays
OFLAG       = -fast
DEBUG       = -Mfree -O0 -traceback

# Cray-MPICH (GPU-aware): include + link via NVHPC's bundled MPI shim is
# unreliable, so point at cray-mpich explicitly. The GTL lib provides the
# CUDA-aware transport.
MPI_INC     = -I${mpich_dir}/include
MPI_LIB     = -L${mpich_dir}/lib -lmpi -lmpifort \\
              -L${gtl_dir} -lmpi_gtl_cuda

# CUDA math libraries from NVHPC; NCCL is bundled.
LLIBS       = \$(MPI_LIB) -cudalib=cublas,cusolver,cufft,nccl -cuda
INCS        = \$(MPI_INC)

# O1/O2 source overrides (kept identical to upstream nvhpc_omp_acc template)
SOURCE_O1  := pade_fit.o minimax_dependence.o wave_window.o
SOURCE_O2  := pead.o

# vasp.5.lib companion build
CPP_LIB     = \$(CPP)
FC_LIB      = \$(FC)
CC_LIB      = \$(CC)
CFLAGS_LIB  = -O -w
FFLAGS_LIB  = -O1 -Mfixed
FREE_LIB    = \$(FREE)
OBJECTS_LIB = linpack_double.o

# Parser library
CXX_PARS    = nvc++ --no_warnings

LLIBS       += -lstdc++

# Cross-compile guard. On Perlmutter compute and login nodes are both Milan.
VASP_TARGET_CPU ?= -tp host
FFLAGS     += \$(VASP_TARGET_CPU)

# NVHPC root + qd (mandatory): autodetected from nvfortran on PATH.
NVROOT      =\$(shell which nvfortran | awk -F /compilers/bin/nvfortran '{ print \$\$1 }')
QD         ?= \$(NVROOT)/compilers/extras/qd
LLIBS      += -L\$(QD)/lib -lqdmod -lqd
INCS       += -I\$(QD)/include/qd

# BLAS / LAPACK from NVIDIA-built cray-libsci. scaLAPACK from netlib build
# (see build-scalapack-perlmutter.sh) -> required for multi-node parallel diag.
BLAS        = -L${libsci_nvidia}/lib -lsci_nvidia_mp
LAPACK      =
SCALAPACK   = ${sca_lib}
LLIBS      += \$(SCALAPACK) \$(LAPACK) \$(BLAS)

# FFTW from cray-fftw (the module exports FFTW_ROOT / FFTW_INC).
LLIBS      += -L${fftw_root}/lib -lfftw3 -lfftw3_omp
INCS       += -I${fftw_root}/include
EOF

    log "Wrote ${out}"
}

# ============================================================================

log "=========================================="
log "VASP GPU build on Perlmutter (A100)"
log "Date:        $(date)"
log "Host:        $(hostname)"
log "VASP root:   ${VASP_ROOT}"
log "PREFIX:      ${PREFIX}"
log "Target:      ${VASP_TARGET}"
log "NCORES:      ${NCORES}"
log "GPU arch:    ${GPU_ARCH}"
log "CUDA ver:    ${CUDA_VER}"
log "CLEAN_BUILD: ${CLEAN_BUILD}"
log "=========================================="

ensure_file "${VASP_ROOT}"
ensure_file "${VASP_ROOT}/makefile"

load_perlmutter_nvhpc_modules

# Auto-build scaLAPACK if requested and not present (multi-node parallel diag).
if [[ -n "${SCALAPACK_ROOT}" && "${SCALAPACK_AUTOBUILD}" == "1" ]]; then
    if [[ ! -f "${SCALAPACK_ROOT}/lib/libscalapack.a" \
       && ! -f "${SCALAPACK_ROOT}/lib64/libscalapack.a" ]]; then
        log "scaLAPACK not found at ${SCALAPACK_ROOT}; auto-building"
        bash "${REPO}/scripts/setup/perlmutter/build-scalapack-perlmutter.sh" \
            || die "scaLAPACK build failed; rerun with SCALAPACK_ROOT='' for a single-node build"
    fi
fi

write_makefile_include

# Sanity: cray-fftw must export FFTW_ROOT (or equivalent) so the makefile resolves.
if [[ -z "${FFTW_ROOT:-}" && -n "${CRAY_FFTW_PREFIX:-}" ]]; then
    export FFTW_ROOT="${CRAY_FFTW_PREFIX}"
fi
[[ -n "${FFTW_ROOT:-}" ]] || warn "FFTW_ROOT is not set in the environment; cray-fftw may be missing."
log "FFTW_ROOT = ${FFTW_ROOT:-<unset>}"

cd "${VASP_ROOT}"

case "${VASP_TARGET}" in
    all)
        BUILD_TARGETS=(std gam ncl)
        ;;
    std|gam|ncl)
        BUILD_TARGETS=("${VASP_TARGET}")
        ;;
    *)
        die "VASP_TARGET must be one of: all, std, gam, ncl"
        ;;
esac

if [[ "${CLEAN_BUILD}" == "1" ]]; then
    log "Removing previous build directories under ${VASP_ROOT}/${PREFIX}"
    rm -rf "${VASP_ROOT}/${PREFIX}/std" \
           "${VASP_ROOT}/${PREFIX}/gam" \
           "${VASP_ROOT}/${PREFIX}/ncl"
fi

for target in "${BUILD_TARGETS[@]}"; do
    log "Invoking make target '${target}'"
    make PREFIX="${PREFIX}" DEPS=1 MODS=1 -j"${NCORES}" "${target}"
done

log "Build finished: $(date)"
for exe in vasp_std vasp_gam vasp_ncl; do
    if [[ -x "${VASP_ROOT}/bin/${exe}" ]]; then
        log "Built ${exe}: ${VASP_ROOT}/bin/${exe}"
    else
        warn "Missing ${exe}: ${VASP_ROOT}/bin/${exe}"
    fi
done
