#!/bin/bash
# Clone, build and install LAMMPS (GPL-2.0) with CMake, then probe the installed executable.
# Validated on Frontier only (GCC 14.2, Cray MPICH 8.1.32; see build-lammps-frontier.sh).
#
# Usage:   bash deployments/common/setup/lammps/build-lammps.sh
#          On Frontier use deployments/frontier/setup/lammps/build-lammps-frontier.sh (CPU) or
#          build-lammps-mliap-frontier.sh (ML-IAP + PYTHON), which load the toolchain first.
# Env:     LAMMPS_ROOT       base directory (default: <repo>/external/lammps)
#          LAMMPS_SRC_DIR    checkout (default: $LAMMPS_ROOT/src)
#          LAMMPS_PREFIX     install prefix (default: $LAMMPS_ROOT/install)
#          LAMMPS_TAG        upstream tag (default: stable_22Jul2025_update6)
#          LAMMPS_COMMIT     expected commit of the tag (default: the validated one for the
#                            default tag; empty skips the check for other tags)
#          LAMMPS_PACKAGES   space-separated packages (default below)
#          LAMMPS_MPI        on|off (default: on)
#          LAMMPS_CXX, LAMMPS_CC  compilers (default: mpicxx/mpicc with MPI, else g++/gcc)
#          LAMMPS_CMAKE      cmake executable (default: cmake from PATH)
#          LAMMPS_CMAKE_ARGS extra CMake arguments, e.g. "-D PKG_ML-IAP=on -D MLIAP_ENABLE_PYTHON=on"
#          LAMMPS_JOBS       parallel build jobs (default: 16)
#          LAMMPS_PROBE_CMD  launcher prefix for the '-h' probe (default: none; on systems where
#                            an MPI binary cannot start on a login node, run the probe in a job)
#          LAMMPS_VARIANT    cpu (default) or mliap. mliap adds PYTHON and ML-IAP with Python
#                            coupling, builds shared libraries against LAMMPS_PYTHON, and uses
#                            build-mliap/, install-mliap/ and activate-mliap.sh so the cpu
#                            build is left untouched.
#          LAMMPS_PYTHON     interpreter for the mliap variant (default: <repo>/.venv/bin/python);
#                            it must provide a shared libpython, NumPy and Cython
# Needs internet for the clone, so run it on a login node. No allocation is required.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(CDPATH= cd -P -- "${SCRIPT_DIR}/../../../.." && pwd -P)"

UPSTREAM="https://github.com/lammps/lammps.git"
DEFAULT_TAG="stable_22Jul2025_update6"
TAG="${LAMMPS_TAG:-${DEFAULT_TAG}}"
if [[ "${TAG}" == "${DEFAULT_TAG}" ]]; then
    COMMIT="${LAMMPS_COMMIT-9c5ab448c78a14fd534619622162ba418d6a1fb1}"
else
    COMMIT="${LAMMPS_COMMIT:-}"
fi
ROOT="${LAMMPS_ROOT:-${REPO}/external/lammps}"
SRC="${LAMMPS_SRC_DIR:-${ROOT}/src}"
VARIANT="${LAMMPS_VARIANT:-cpu}"
MPI="${LAMMPS_MPI:-on}"
CMAKE="${LAMMPS_CMAKE:-cmake}"
# MANYBODY: EAM/Tersoff/SW; EXTRA-FIX: electron stopping for cascades; EXTRA-PAIR: more pairs.
PACKAGES="${LAMMPS_PACKAGES:-MANYBODY EXTRA-PAIR EXTRA-FIX KSPACE MOLECULE RIGID OPENMP}"
VARIANT_ARGS=()
case "${VARIANT}" in
    cpu)
        SUFFIX=""
        SHARED=off
        ;;
    mliap)
        SUFFIX="-mliap"
        SHARED=on
        PY_BIN="${LAMMPS_PYTHON:-${REPO}/.venv/bin/python}"
        [[ -x "${PY_BIN}" ]] || { echo "LAMMPS_PYTHON not executable: ${PY_BIN}" >&2; exit 2; }
        PY_LIBDIR="$("${PY_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
        PY_VER="$("${PY_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("VERSION"))')"
        [[ -e "${PY_LIBDIR}/libpython${PY_VER}.so" ]] || {
            echo "no shared ${PY_LIBDIR}/libpython${PY_VER}.so; ML-IAP Python needs one" >&2
            exit 2
        }
        "${PY_BIN}" -c 'import numpy, Cython' || { echo "${PY_BIN} lacks numpy/Cython" >&2; exit 2; }
        PACKAGES="${PACKAGES} PYTHON ML-IAP"
        PY_DIR="$(dirname "${PY_BIN}")"
        PY_INC="$("${PY_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
        # Pin the shared libpython; FindPython would otherwise pick a static archive if present.
        VARIANT_ARGS=(-D MLIAP_ENABLE_PYTHON=on -D Python_EXECUTABLE="${PY_BIN}"
            -D Python_LIBRARY="${PY_LIBDIR}/libpython${PY_VER}.so"
            -D Python_INCLUDE_DIR="${PY_INC}"
            -D Cythonize_EXECUTABLE="${PY_DIR}/cythonize")
        ;;
    *) echo "LAMMPS_VARIANT must be cpu or mliap" >&2; exit 2 ;;
esac
PREFIX="${LAMMPS_PREFIX:-${ROOT}/install${SUFFIX}}"
BUILD="${ROOT}/build${SUFFIX}"
ACTIVATE="${ROOT}/activate${SUFFIX}.sh"
RECORD="${ROOT}/lammps${SUFFIX}-build-record.txt"
HELP="${ROOT}/lmp${SUFFIX}-help.txt"
if [[ "${MPI}" == "on" ]]; then
    CXX_BIN="${LAMMPS_CXX:-mpicxx}"
    CC_BIN="${LAMMPS_CC:-mpicc}"
else
    CXX_BIN="${LAMMPS_CXX:-g++}"
    CC_BIN="${LAMMPS_CC:-gcc}"
fi

for tool in git make "${CMAKE}" "${CXX_BIN}" "${CC_BIN}"; do
    command -v "${tool}" >/dev/null || { echo "required tool not found: ${tool}" >&2; exit 2; }
done

if [[ ! -d "${SRC}/.git" ]]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone --branch "${TAG}" --depth 1 "${UPSTREAM}" "${SRC}"
fi
HEAD_TAG="$(git -C "${SRC}" describe --tags --exact-match 2>/dev/null || true)"
if [[ "${HEAD_TAG}" != "${TAG}" ]]; then
    echo "${SRC} is not at ${TAG} (found '${HEAD_TAG:-no tag}'); remove it or set LAMMPS_TAG" >&2
    exit 2
fi
HEAD_COMMIT="$(git -C "${SRC}" rev-parse HEAD)"
if [[ -n "${COMMIT}" && "${HEAD_COMMIT}" != "${COMMIT}" ]]; then
    echo "${TAG} resolves to ${HEAD_COMMIT}, expected ${COMMIT}; upstream moved the tag" >&2
    exit 2
fi

PKG_ARGS=()
for p in ${PACKAGES}; do
    PKG_ARGS+=("-D" "PKG_${p}=on")
done
# shellcheck disable=SC2206
EXTRA=(${LAMMPS_CMAKE_ARGS:-})

rm -rf "${BUILD}"
"${CMAKE}" -S "${SRC}/cmake" -B "${BUILD}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
    -D CMAKE_CXX_COMPILER="${CXX_BIN}" -D CMAKE_C_COMPILER="${CC_BIN}" \
    -D BUILD_MPI="${MPI}" -D BUILD_OMP=on -D LAMMPS_EXCEPTIONS=on \
    -D BUILD_SHARED_LIBS="${SHARED}" -D CMAKE_INSTALL_LIBDIR=lib \
    "${PKG_ARGS[@]}" "${VARIANT_ARGS[@]}" "${EXTRA[@]}"
"${CMAKE}" --build "${BUILD}" --parallel "${LAMMPS_JOBS:-16}"
"${CMAKE}" --install "${BUILD}"

EXE="${PREFIX}/bin/lmp"
POT="${PREFIX}/share/lammps/potentials"
[[ -x "${EXE}" ]] || { echo "install produced no ${EXE}" >&2; exit 1; }
[[ -d "${POT}" ]] || { echo "install produced no ${POT}" >&2; exit 1; }

cat > "${ACTIVATE}" <<EOF
# Generated by build-lammps.sh; source it before running the self-built LAMMPS.
export LAMMPS_EXECUTABLE="${EXE}"
export LAMMPS_POTENTIALS="${POT}"
export PATH="${PREFIX}/bin\${PATH:+:\${PATH}}"
EOF
if [[ "${VARIANT}" == "mliap" ]]; then
    # The embedded interpreter must resolve the same stdlib and libpython it was linked with.
    cat >> "${ACTIVATE}" <<EOF
export LAMMPS_VARIANT=mliap
export LAMMPS_PYTHON="${PY_BIN}"
export PYTHONHOME="$("${PY_BIN}" -c 'import sys; print(sys.base_prefix)')"
export PYTHONPATH="${SRC}/python\${PYTHONPATH:+:\${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PY_LIBDIR}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
fi

{
    echo "lammps_tag=${TAG}"
    echo "lammps_commit=$(git -C "${SRC}" rev-parse HEAD)"
    echo "variant=${VARIANT}"
    [[ "${VARIANT}" == "mliap" ]] && echo "python=$("${PY_BIN}" -c 'import sys; print(sys.version.split()[0])') (${PY_BIN})"
    echo "cxx=$("${CXX_BIN}" --version | head -1)"
    echo "mpi=${MPI}"
    echo "packages=${PACKAGES}"
    echo "extra_cmake_args=${LAMMPS_CMAKE_ARGS:-}"
    echo "lmp_sha256=$(sha256sum "${EXE}" | cut -d' ' -f1)"
} > "${RECORD}"
cat "${RECORD}"

# shellcheck disable=SC1090
# shellcheck disable=SC2086
if ! (source "${ACTIVATE}" && ${LAMMPS_PROBE_CMD:-} "${EXE}" -h > "${HELP}" 2>&1); then
    echo "probe '${EXE} -h' failed (see ${HELP}); if MPI cannot start here," >&2
    echo "run the probe inside a job or set LAMMPS_PROBE_CMD" >&2
    exit 1
fi
grep -q "Large-scale Atomic/Molecular Massively Parallel Simulator" "${HELP}" || {
    echo "probe printed no LAMMPS banner (see ${HELP})" >&2
    exit 1
}
grep -A3 '^Installed packages:' "${HELP}"
if [[ "${VARIANT}" == "mliap" ]]; then
    grep -q "ML-IAP" "${HELP}" && grep -q "PYTHON" "${HELP}" || {
        echo "ML-IAP or PYTHON missing from the installed packages (see ${HELP})" >&2
        exit 1
    }
fi
echo "Environment for jobs: source ${ACTIVATE}"
