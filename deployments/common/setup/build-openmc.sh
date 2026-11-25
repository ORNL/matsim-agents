#!/bin/bash
# Clone, build and install OpenMC (MIT) for activation/transmutation, fetch nuclear data, smoke-test.
# Validated on Frontier only (serial build, GCC 12.3, Cray HDF5 1.14.3.9, Python 3.11).
#
# Usage:   source .venv/bin/activate && bash deployments/common/setup/build-openmc.sh
# Env:     OPENMC_DIR          checkout/build directory (default: <repo>/external/openmc)
#          OPENMC_TAG          upstream tag (default: v0.15.3, the last release for Python 3.11)
#          OPENMC_DATA_DIR     nuclear-data cache, outside Git (default: <repo>/.hpc-build/openmc-data)
#          OPENMC_CXX/OPENMC_CC  compilers (default: g++/gcc from PATH)
#          OPENMC_HDF5_ROOT    HDF5 prefix (default: $HDF5_ROOT, else $HDF5_DIR)
#          OPENMC_SKIP_DATA=1  do not download the 3.4 GB cross sections or the 27 MB chain
#          OPENMC_SKIP_SMOKE=1 skip the Co-59 activation check
#          OPENMC_RUNTIME_LIBS  extra LD_LIBRARY_PATH entries for the smoke test (on Frontier:
#                      the HDF5 lib, GCC snos/lib64 and MPICH lib directories; see the docs)
# Needs internet (clone, pip, data), so run it on a login node. No allocation is required.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(CDPATH= cd -P -- "${SCRIPT_DIR}/../../.." && pwd -P)"

UPSTREAM="https://github.com/openmc-dev/openmc.git"
TAG="${OPENMC_TAG:-v0.15.3}"
SRC="${OPENMC_DIR:-${REPO}/external/openmc}"
DATA="${OPENMC_DATA_DIR:-${REPO}/.hpc-build/openmc-data}"
CXX_BIN="${OPENMC_CXX:-g++}"
CC_BIN="${OPENMC_CC:-gcc}"
HDF5="${OPENMC_HDF5_ROOT:-${HDF5_ROOT:-${HDF5_DIR:-}}}"

# ENDF/B-VIII.0 libraries published by the OpenMC team (openmc.org/data); no checksums are given.
XS_URL="https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz"
CHAIN_URL="https://anl.box.com/shared/static/nyezmyuofd4eqt6wzd626lqth7wvpprr.xml"
XS_BYTES=3383607420
CHAIN_BYTES=27526672

for tool in git cmake make curl tar xz "${CXX_BIN}" "${CC_BIN}" python; do
    command -v "${tool}" >/dev/null || { echo "required tool not found: ${tool}" >&2; exit 2; }
done
[[ -n "${VIRTUAL_ENV:-}" ]] || { echo "activate the matsim-agents virtualenv first" >&2; exit 2; }
[[ -n "${HDF5}" && -f "${HDF5}/include/hdf5.h" ]] || {
    echo "set OPENMC_HDF5_ROOT to an HDF5 prefix with include/hdf5.h (on Frontier:" >&2
    echo "  module load cray-hdf5; the prefix is \$HDF5_ROOT)" >&2
    exit 2
}
PYV="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${TAG}" == v0.15.* && "${PYV}" != 3.1[1-9] ]]; then
    echo "OpenMC ${TAG} needs Python >= 3.11; this is ${PYV}" >&2
    exit 2
fi

# 1. Source, with the vendored submodules the C++ library needs.
if [[ ! -d "${SRC}/.git" ]]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone --branch "${TAG}" --depth 1 --recurse-submodules --shallow-submodules \
        "${UPSTREAM}" "${SRC}"
fi
HEAD_TAG="$(git -C "${SRC}" describe --tags --exact-match 2>/dev/null || true)"
if [[ "${HEAD_TAG}" != "${TAG}" ]]; then
    echo "${SRC} is not at ${TAG} (found '${HEAD_TAG:-no tag}'); remove it or set OPENMC_TAG" >&2
    exit 2
fi

# 2. Python package. --no-deps keeps the environment's numpy/scipy; the explicit list is the
#    runtime dependency set from OpenMC's pyproject.toml.
python -m pip install --quiet numpy h5py scipy matplotlib pandas lxml uncertainties \
    setuptools endf ipython
python -m pip install --quiet --no-deps "${SRC}"

# 3. C++ library (libopenmc.so), serial: no MPI, no OpenMP, no DAGMC or libMesh.
BUILD="${SRC}/build"
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${CXX_BIN}" -DCMAKE_C_COMPILER="${CC_BIN}" \
    -DHDF5_ROOT="${HDF5}" -DOPENMC_USE_MPI=off -DOPENMC_USE_OPENMP=off
cmake --build "${BUILD}" --parallel "${OPENMC_JOBS:-8}"
LIB="${BUILD}/lib/libopenmc.so"
[[ -f "${LIB}" ]] || LIB="$(find "${BUILD}" -name 'libopenmc.so' | head -1)"
[[ -f "${LIB}" ]] || { echo "build produced no libopenmc.so" >&2; exit 1; }
PKG="$(python -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("openmc").origin))')"
mkdir -p "${PKG}/lib"
cp -f "${LIB}" "${PKG}/lib/libopenmc.so"

# 4. Nuclear data (login node only). Re-run safely: complete files are kept.
mkdir -p "${DATA}"
XS_XML="${DATA}/endfb-viii.0-hdf5/cross_sections.xml"
CHAIN="${DATA}/chain_endfb80_pwr.xml"
if [[ "${OPENMC_SKIP_DATA:-0}" != "1" ]]; then
    if [[ ! -f "${CHAIN}" || "$(stat -c %s "${CHAIN}")" != "${CHAIN_BYTES}" ]]; then
        curl -fL --retry 3 -o "${CHAIN}" "${CHAIN_URL}"
    fi
    [[ "$(stat -c %s "${CHAIN}")" == "${CHAIN_BYTES}" ]] || {
        echo "chain file has an unexpected size (upstream may have changed)" >&2
        exit 1
    }
    if [[ ! -f "${XS_XML}" ]]; then
        ARCHIVE="${DATA}/endfb80_hdf5.tar.xz"
        curl -fL --retry 3 -C - -o "${ARCHIVE}" "${XS_URL}"
        [[ "$(stat -c %s "${ARCHIVE}")" == "${XS_BYTES}" ]] || {
            echo "cross-section archive has an unexpected size (upstream may have changed)" >&2
            exit 1
        }
        tar -xJf "${ARCHIVE}" -C "${DATA}"
        rm -f "${ARCHIVE}"
    fi
fi

# 5. Record what was built; data hashes are local, the publisher gives none.
RECORD="${DATA}/openmc-build-record.txt"
{
    echo "openmc_tag=${TAG}"
    echo "openmc_commit=$(git -C "${SRC}" rev-parse HEAD)"
    echo "python=$(python --version 2>&1)"
    echo "cxx=$("${CXX_BIN}" --version | head -1)"
    echo "hdf5=${HDF5}"
    echo "libopenmc_sha256=$(sha256sum "${PKG}/lib/libopenmc.so" | cut -d' ' -f1)"
    [[ -f "${CHAIN}" ]] && echo "chain_sha256=$(sha256sum "${CHAIN}" | cut -d' ' -f1)"
    [[ -f "${XS_XML}" ]] && echo "cross_sections_xml_sha256=$(sha256sum "${XS_XML}" | cut -d' ' -f1)"
} > "${RECORD}"
cat "${RECORD}"

# 6. Smoke test: Co-59 -> Co-60 against the analytic inventory.
if [[ "${OPENMC_SKIP_SMOKE:-0}" != "1" && -f "${XS_XML}" && -f "${CHAIN}" ]]; then
    LD_LIBRARY_PATH="${OPENMC_RUNTIME_LIBS:+${OPENMC_RUNTIME_LIBS}:}${HDF5}/lib:${LD_LIBRARY_PATH:-}" \
        OPENMC_CROSS_SECTIONS="${XS_XML}" OPENMC_CHAIN_FILE="${CHAIN}" \
        python "${SCRIPT_DIR}/openmc-smoke.py"
fi

echo ""
echo "Use it with:"
echo "  export OPENMC_CROSS_SECTIONS=${XS_XML}"
echo "  export OPENMC_CHAIN_FILE=${CHAIN}"
echo "  # on Frontier, importing openmc.deplete also needs the HDF5, GCC and MPICH libraries on"
echo "  # LD_LIBRARY_PATH; see docs/external/openmc.md"
