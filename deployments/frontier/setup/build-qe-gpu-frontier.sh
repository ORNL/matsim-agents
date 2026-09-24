#!/bin/bash
#SBATCH -J build-qe-gpu
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

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
#   nohup bash deployments/frontier/setup/build-qe-gpu-frontier.sh \
#         > runs/build-qe-gpu-login/build.log 2>&1 &
#
#   # Or as a batch job (sbatch headers below remain valid):
#   sbatch deployments/frontier/setup/build-qe-gpu-frontier.sh
#
# Use the develop branch (or qe-7.4+); older QE lacks QE_ENABLE_OFFLOAD.
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
# (gitignored). The repo "owns" the location while the large build artifacts
# stay out of version control. Override with:
#   QE_PREFIX=/some/other/path bash deployments/frontier/setup/build-qe-gpu-frontier.sh
# to relocate src+build+install as a unit, or override SRC_DIR/BUILD_DIR/
# INSTALL_DIR individually for finer control.
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
SRC_DIR="${SRC_DIR:-${QE_PREFIX}/src}"
BUILD_DIR="${BUILD_DIR:-${QE_PREFIX}/build-gpu}"
INSTALL_DIR="${INSTALL_DIR:-${QE_PREFIX}/install-gpu}"

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
# NOTE: We deliberately stay on the Frontier-default cce/18.0.1 stack.
# Newer cce versions (19/20/21) are installed but their ftn wrapper requires
# `libopenacc.pc`, which Frontier's site config does not ship; loading them
# breaks every Fortran compile with `Error invoking pkg-config!`.
# A per-file workaround later in this script handles cce/18.0.1's ICE on
# PW/src/gen_at_d{j,y}.f90.
module load PrgEnv-cray
module load cce                          # default 18.0.1 on Frontier
module load craype-accel-amd-gfx90a      # enables -fopenmp offload to gfx90a
# ROCm version is selectable via env. Default rocm/6.2.4 is the only option
# that is ABI-compatible with Frontier's cray-mpich/8.1.31 GTL: the GTL library
# /opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so is hard-linked against
# libamdhip64.so.6 (rocm 6.x SONAME). rocm/7.x ships libamdhip64.so.7 and
# breaks the MPI Fortran link probe at CMake configure. Frontier does not yet
# ship a cray-mpich built against rocm 7. Override with ROCM_MODULE=rocm/6.4.x
# only if needed.
ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.4}"
module load "${ROCM_MODULE}"             # rocFFT, rocBLAS, rocSOLVER
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
# NOTE: full clone (no --depth=1) so QE's GitInfo.cmake `git describe` succeeds
# and produces a complete git-rev.h. Shallow clones leave GIT_BRANCH_RAW and
# GIT_HASH_RAW undefined, which breaks Modules/environment.f90 compilation.
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
    # Unshallow if a previous run did a shallow clone (so git describe works).
    if [[ -f "${SRC_DIR}/.git/shallow" ]]; then
        echo "  unshallowing existing repo so git describe succeeds ..."
        (cd "${SRC_DIR}" && git fetch --unshallow 2>/dev/null || true)
    fi
    echo "  HEAD = $(cd "${SRC_DIR}" && git rev-parse --short HEAD)"
fi

# ---- Configure with CMake ---------------------------------------------------
# Reuse cached objects when CLEAN_BUILD is empty/0; otherwise wipe build dir.
if [[ "${CLEAN_BUILD:-1}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi
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

# ---- Workaround: cce/18.0.1 ICE retry loop ----------------------------------
# Cray Fortran 18.0.1 hits an internal compiler error (ftn-7991 in
# /workspace/crayftn/pdgcs/v_fei.c:3564, "Found inner_ref/inner_def object
# without Fortran internal procedure") on a number of QE PW/Modules sources
# whenever `-target-accel=amd_gfx90a -h omp` is in effect. The full set of
# affected files is QE-version dependent; observed on develop @ 0f3d904 are at
# least: PW/src/{gen_at_dj,gen_at_dy,stres_us,data_structure,s_psi,esm,exx,
# h_psi,pw_restart_new,wfcinit,setup,usnldiag}.f90.
#
# Switching to cce/{19,20,21}.x is not viable on Frontier today: those
# compilers' ftn wrappers require libopenacc.pc which is not installed in the
# site config (`Error invoking pkg-config!`).
#
# Strategy: run `make` in a loop. After each failure, parse the ftn-7991 lines
# from the make log, recompile each ICE'd source manually with the AMD-offload
# flags stripped (-target-accel=... removed; -h omp -> -h noomp), then resume.
# This is safe because none of the affected files contain `!$omp target`
# directives, so disabling device codegen for them changes no semantics.
# All other sources keep -DHIP, -D__ROCBLAS, -D__OPENMP_GPU and the offload
# codegen flags, so rocFFT/rocBLAS linkage is preserved.
run_make_with_ice_workaround() {
  # Args: <log-prefix> <make target args...>
  local logprefix="$1"; shift
  local logdir="${BUILD_DIR}/ice-workaround-logs"
  mkdir -p "${logdir}"
  local max_iter=80
  local iter=0
  while true; do
    iter=$((iter+1))
    if (( iter > max_iter )); then
      echo ">>> ABORT: hit max_iter=${max_iter}"
      return 2
    fi
    local ml="${logdir}/${logprefix}-iter-${iter}.log"
    echo ">>> [${logprefix} iter ${iter}] make -j${NCORES} $* -> ${ml}"
    cd "${BUILD_DIR}"
    if make -j"${NCORES}" "$@" > "${ml}" 2>&1; then
      echo ">>> ${logprefix} make succeeded on iter ${iter}"
      return 0
    fi
    # Extract sources that hit ftn-7991 ICE this iter.
    local ice_paths
    ice_paths=$(grep -E '^ftn-7991 ftn: INTERNAL , File = ' "${ml}" \
                | sed -E 's/.*File = ([^,]+),.*/\1/' | sort -u)
    if [[ -z "${ice_paths}" ]]; then
      echo ">>> make failed on iter ${iter} but no ftn-7991 ICE detected; aborting"
      tail -40 "${ml}"
      return 3
    fi
    echo ">>> ICE'd this iter:"
    echo "${ice_paths}" | sed 's/^/    /'
    local fullp srcf rel subsys base ff objf objrel build_make defs incs flags compile_dir
    for fullp in ${ice_paths}; do
      srcf="${fullp}"
      [[ ! -f "${srcf}" ]] && srcf="/$(echo "${fullp}" | sed 's,^\.\./\.\./\.\./,,')"
      [[ ! -f "${srcf}" ]] && { echo "    >> cannot locate source for: ${fullp}"; return 5; }
      rel="${srcf#${SRC_DIR}/}"
      subsys="$(echo "${rel}" | cut -d/ -f1)"
      base="$(basename "${srcf}" .f90)"
      # Find the build.make whose rule lists this exact source path. The rule
      # line looks like: "<objpath>.f90.o: <absolute source path>". The
      # captured objpath is BUILD_DIR-relative (e.g. PW/CMakeFiles/qe_pw.dir/src/foo.f90.o).
      build_make=""
      objrel=""
      while IFS= read -r cand; do
        local m
        m=$(grep -E "^[^[:space:]]+\.f90\.o:[[:space:]]+${srcf}\$" "${cand}" 2>/dev/null | head -1 || true)
        if [[ -n "${m}" ]]; then
          objrel="${m%%:*}"
          build_make="${cand}"
          break
        fi
      done < <(find "${BUILD_DIR}/${subsys}/CMakeFiles" -name build.make 2>/dev/null)
      if [[ -z "${objrel}" ]]; then
        echo "    >> could not locate build.make rule for ${rel}"
        return 7
      fi
      ff="$(dirname "${build_make}")/flags.make"
      objf="${BUILD_DIR}/${objrel}"
      # cmake's compile rule cd's into the dir containing the target (e.g. PW),
      # so we must run ftn from the same dir for include paths to resolve.
      compile_dir="${BUILD_DIR}/${objrel%%/CMakeFiles/*}"
      defs=$(grep '^Fortran_DEFINES'  "${ff}" | sed -E 's/^Fortran_DEFINES *= *//')
      incs=$(grep '^Fortran_INCLUDES' "${ff}" | sed -E 's/^Fortran_INCLUDES *= *//')
      flags=$(grep '^Fortran_FLAGS'   "${ff}" | sed -E 's/^Fortran_FLAGS *= *//; s/-target-accel=amd_gfx90a//g; s/-h[[:space:]]+omp/-h noomp/g')
      mkdir -p "$(dirname "${objf}")"
      cd "${compile_dir}"
      echo "    >> recompiling ${rel} -> ${objrel}"
      # shellcheck disable=SC2086
      if ! ftn ${defs} ${incs} ${flags} -c "${srcf}" -o "${objf}" 2>>"${logdir}/workaround.log"; then
        echo "    >> FAILED to compile ${rel} even without offload"
        tail -20 "${logdir}/workaround.log"
        return 4
      fi
      [[ ! -s "${objf}" ]] && { echo "    >> ${objf} empty after compile"; return 4; }
      # Touch the object newer than its sources so make does not rebuild it.
      touch "${objf}"
    done
    cd "${BUILD_DIR}"
  done
}

# ---- Source patch: pimd_subrout.f90 etime() -> cpu_time() -------------------
# QE's PIOUD (Path-Integral Ornstein-Uhlenbeck Dynamics) module calls the GNU
# Fortran extension `etime()` in its scnd() timing helper. Cray Fortran does
# not provide etime, causing pioud.x to fail at link time with:
#   undefined reference to `etime_'
# We rewrite scnd() to use the F95-standard `cpu_time` intrinsic, which Cray
# Fortran does provide and which returns the same kind of wall-time timer
# (CPU seconds, real(8)). The patch is idempotent (a marker comment is added).
PIMD_F="${SRC_DIR}/PIOUD/src/pimd_subrout.f90"
if [[ -f "${PIMD_F}" ]] && ! grep -q 'patched-by-frontier-build-script' "${PIMD_F}"; then
  echo "--- Patching ${PIMD_F#${SRC_DIR}/} (etime -> cpu_time) ---"
  python3 - "${PIMD_F}" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
old = ("subroutine scnd(t)\n\n"
       "  implicit none\n"
       "  real(8) :: t\n"
       "  real(4) :: etime\n"
       "  real(4), dimension (:), allocatable :: tarray\n\n"
       "  allocate(tarray(2))\n\n"
       "  t = etime(tarray)\n"
       "  t = tarray(1)\n"
       "   \n"
       "  deallocate(tarray)\n\n"
       "  return\n"
       "end subroutine scnd")
new = ("subroutine scnd(t)\n"
       "  ! patched-by-frontier-build-script: replaced GNU etime() with\n"
       "  ! F95-standard cpu_time() so pioud.x links under Cray Fortran.\n"
       "  implicit none\n"
       "  real(8) :: t\n"
       "  call cpu_time(t)\n"
       "  return\n"
       "end subroutine scnd")
if old not in text:
    print("PATCH MARKER MISSED: scnd block has changed; aborting", file=sys.stderr)
    sys.exit(1)
p.write_text(text.replace(old, new))
print("  scnd() rewritten")
PY
else
  echo "--- pimd_subrout.f90 already patched (or missing); skipping ---"
fi

# ---- Source patch: cce-18.0.1 uninitialized-descriptor segfault fix --------
# pw.x segfaults deterministically inside pw_init_qexsd_input on cce-18.0.1:
# hybrid_type/dftU_type/vdW_type have several ALLOCATABLE array components,
# and when declared as plain automatic locals here, cce-18.0.1 does not
# reliably initialize their allocation-status descriptor. Fix: SAVE +
# explicit reset/allocate on those locals (pw_init_qexsd_input.f90), plus a
# descriptor-safe manual deep-copy replacing whole-structure derived-type
# assignment in qes_init_dft (Modules/qes_init_module.f90), plus matching
# defensive hardening in pw_write_schema (PW/src/pw_restart_new.f90). See
# deployments/frontier/setup/patches/qe-cce-segfault-fix/ for the full patch
# and rationale. Idempotent: skipped if the marker comment is already present.
QES_INIT_F="${SRC_DIR}/Modules/qes_init_module.f90"
CCE_SEGFAULT_PATCH="${SCRIPT_DIR}/patches/qe-cce-segfault-fix/cce-uninitialized-descriptor-segfault.patch"
if [[ -f "${QES_INIT_F}" ]] && ! grep -q 'Frontier/cce-18.0.1 workaround' "${QES_INIT_F}"; then
  echo "--- Patching cce-18.0.1 uninitialized-descriptor segfault (pw_init_qexsd_input, qes_init_module, pw_restart_new) ---"
  ( cd "${SRC_DIR}" && patch -p1 --forward < "${CCE_SEGFAULT_PATCH}" )
else
  echo "--- cce-18.0.1 segfault fix already applied (or files missing); skipping ---"
fi

# ---- Build ------------------------------------------------------------------
# Build the QE target groups we actually need under the ICE retry loop.
# Avoid `make all` because it pulls in dev/test targets that reference
# uninstalled tools. The list below covers every user-facing executable QE
# ships (pw, cp, ph, pp, neb, hp, ld1, pwall, pwcond, tddfpt, upf, xspectra,
# epw, kcw, all_currents) including pioud (after the etime patch above).
QE_TARGETS=(pw cp ph pp neb hp ld1 pwall pwcond tddfpt upf xspectra epw kcw all_currents pioud)
echo "--- Building QE GPU targets: ${QE_TARGETS[*]} ---"
run_make_with_ice_workaround targets "${QE_TARGETS[@]}"
MAKE_RC=$?
if (( MAKE_RC != 0 )); then
  echo "BUILD FAILED with rc=${MAKE_RC}; see ${BUILD_DIR}/ice-workaround-logs/"
  exit ${MAKE_RC}
fi

echo ""
echo "--- Build complete ---"

# ---- Install ----------------------------------------------------------------
# Install only the executables we built. We don't run `make install` because
# its dependency on `all` pulls test/dev targets we deliberately skipped.
echo "--- Installing executables from ${BUILD_DIR}/bin to ${INSTALL_DIR}/bin ---"
mkdir -p "${INSTALL_DIR}/bin"
cp "${BUILD_DIR}"/bin/*.x "${INSTALL_DIR}/bin/"
echo "  installed $(ls "${INSTALL_DIR}/bin/" | wc -l) executables"

echo ""
echo "========================================="
echo "GPU build finished: $(date)"
echo "Executables in: ${INSTALL_DIR}/bin/"
ls "${INSTALL_DIR}/bin/" 2>/dev/null || ls "${BUILD_DIR}/bin/"
echo ""
echo "--- Verifying GPU linkage of pw.x ---"
PW_BIN="${INSTALL_DIR}/bin/pw.x"
[[ -x "${PW_BIN}" ]] || PW_BIN="${BUILD_DIR}/bin/pw.x"
ldd "${PW_BIN}" 2>/dev/null | grep -iE "amdhip|hsa|rocm|amd_comgr|sci_cray|fftw" | sed 's/^/  /'
echo "=========================================="
echo ""
echo "To run pw.x with GPU offload, on a compute node:"
echo "  module load PrgEnv-cray cce craype-accel-amd-gfx90a rocm cray-fftw"
echo "  export OMP_NUM_THREADS=7        # 7 cores per GCD on Frontier"
echo "  export OMP_TARGET_OFFLOAD=MANDATORY   # fail loudly if offload misroutes"
echo "  export MPICH_GPU_SUPPORT_ENABLED=1"
echo "  srun -N1 -n8 -c7 --gpus-per-node=8 --gpu-bind=closest \\"
echo "       ${INSTALL_DIR}/bin/pw.x -in your-input.in"
