#!/usr/bin/env bash
# =============================================================================
# build-vasp-gpu-aurora.sh
#
# Build VASP 6.6.0 on Aurora using the locally prepared Intel GPU offload
# makefile.include.
#
# Usage:
#   bash deployments/aurora/setup/build-vasp-gpu-aurora.sh
#
# Optional overrides:
#   VASP_ROOT=/path/to/vasp.6.6.0
#   PREFIX=build
#   NCORES=32
#   CLEAN_BUILD=0
#   VASP_TARGET=all|std|gam|ncl
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
BUILD_TARGETS=()

if [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    NCORES="${NCORES:-${SLURM_CPUS_ON_NODE}}"
elif command -v nproc >/dev/null 2>&1; then
    NCORES="${NCORES:-$(nproc)}"
else
    NCORES="${NCORES:-8}"
fi

log()  { printf '\033[1;34m[vasp-aurora]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[vasp-aurora]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[vasp-aurora]\033[0m %s\n' "$*" >&2; exit 1; }

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
    module load frameworks
}

ensure_file() {
    local path="$1"
    [[ -e "${path}" ]] || die "Required path not found: ${path}"
}

log "=========================================="
log "VASP GPU build on Aurora"
log "Date:       $(date)"
log "Host:       $(hostname)"
log "VASP root:  ${VASP_ROOT}"
log "PREFIX:     ${PREFIX}"
log "Target:     ${VASP_TARGET}"
log "NCORES:     ${NCORES}"
log "=========================================="

ensure_file "${VASP_ROOT}"
ensure_file "${VASP_ROOT}/makefile"
ensure_file "${VASP_ROOT}/makefile.include"

load_aurora_modules

log "Loaded modules:"
module list 2>&1 || true

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
    rm -rf "${VASP_ROOT}/${PREFIX}/std" "${VASP_ROOT}/${PREFIX}/gam" "${VASP_ROOT}/${PREFIX}/ncl"
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
