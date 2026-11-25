#!/bin/bash
# Build SPECTRA-PKA (serial Fortran, GPL-3.0) on a Frontier login node and run the upstream smoke test.
#
# Usage:   bash deployments/frontier/setup/build-spectra-pka-frontier.sh
# Env:     SPECTRA_PKA_DIR     checkout/build directory (default: <repo>/external/SPECTRA-PKA)
#          SPECTRA_PKA_COMMIT  upstream commit to build (default: the one validated on Frontier)
#          SPECTRA_PKA_FC      Fortran compiler (default: gfortran; validated with gfortran 7.5)
#          SPECTRA_PKA_ALLOW_OTHER_COMMIT=1  build a checkout that is not at SPECTRA_PKA_COMMIT
# Needs internet for the clone, so run it on a login node. No scheduler allocation is required.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(CDPATH= cd -P -- "${SCRIPT_DIR}/../../.." && pwd -P)"

UPSTREAM="https://github.com/fispact/SPECTRA-PKA.git"
COMMIT="${SPECTRA_PKA_COMMIT:-fff9e3f}"
SRC="${SPECTRA_PKA_DIR:-${REPO}/external/SPECTRA-PKA}"
FC="${SPECTRA_PKA_FC:-gfortran}"

command -v "${FC}" >/dev/null || { echo "Fortran compiler not found: ${FC}" >&2; exit 2; }
command -v make >/dev/null || { echo "make not found" >&2; exit 2; }

if [[ ! -d "${SRC}/.git" ]]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone "${UPSTREAM}" "${SRC}"
    git -C "${SRC}" checkout --quiet "${COMMIT}"
fi

HEAD_COMMIT="$(git -C "${SRC}" rev-parse HEAD)"
if [[ "${HEAD_COMMIT}" != "${COMMIT}"* && "${SPECTRA_PKA_ALLOW_OTHER_COMMIT:-0}" != "1" ]]; then
    echo "${SRC} is at ${HEAD_COMMIT}, expected ${COMMIT}." >&2
    echo "Check it out, or set SPECTRA_PKA_ALLOW_OTHER_COMMIT=1 to build it anyway." >&2
    exit 2
fi

echo "Compiler: $("${FC}" --version | head -1)"
echo "Source:   ${SRC} (${HEAD_COMMIT})"

# The Makefile has no module ordering, so a parallel build fails: never use make -j.
make -C "${SRC}" distclean
make -C "${SRC}" FF="${FC}" LD="${FC}"
EXE="${SRC}/spectra-pka"
[[ -x "${EXE}" ]] || { echo "build produced no executable" >&2; exit 1; }

# Smoke test on the bundled Zr data in a scratch copy; it checks the build, not any campaign.
WORK="${SPECTRA_PKA_SMOKE_DIR:-${REPO}/.hpc-build/frontier/spectra-pka-smoke}"
rm -rf "${WORK}"
mkdir -p "${WORK}"
TEST="${SRC}/manual/test"
REF="${TEST}/example_results"
cp -r "${TEST}/ZR.in" "${TEST}/fluxes_specter.dat" "${TEST}/example_data" "${WORK}/"
(cd "${WORK}" && timeout 300 "${EXE}" ZR.in > run.log 2>&1)

tail -1 "${WORK}/run.log" | grep -q "END as expected" || {
    echo "smoke test failed: missing 'END as expected' (see ${WORK}/run.log)" >&2
    exit 1
}
cmp -s "${WORK}/ZR.indexes" "${REF}/ZR.indexes" || {
    echo "smoke test failed: ZR.indexes differs from upstream" >&2
    exit 1
}
# The shipped ZR.out predates the current format (energy digits, dpa columns): informational.
if cmp -s "${WORK}/ZR.out" "${REF}/ZR.out"; then
    OUT_MATCH="byte-identical to upstream example_results"
else
    OUT_MATCH="differs from upstream example_results (older output format; see docs/external/spectra-pka.md)"
fi

{
    echo "commit=${HEAD_COMMIT}"
    echo "compiler=$("${FC}" --version | head -1)"
    echo "executable=${EXE}"
    echo "sha256=$(sha256sum "${EXE}" | cut -d' ' -f1)"
    echo "smoke_indexes=identical"
    echo "smoke_out=${OUT_MATCH}"
} > "${WORK}/build-record.txt"

cat "${WORK}/build-record.txt"
echo ""
echo "Use it with:  export SPECTRA_PKA_EXECUTABLE_PATH=${EXE}"
