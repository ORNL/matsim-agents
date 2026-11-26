#!/bin/bash
# Build LAMMPS on Frontier from source (CPU, MPI + OpenMP) into <repo>/.hpc-build/frontier/lammps/:
#   src/ (clone), build/, install/ (bin/lmp, share/lammps/potentials), activate.sh, build record.
# Use this where no LAMMPS module exists or a package is missing; otherwise the module works
# (see docs/external/lammps.md).
#
# Usage (login node, internet needed, no allocation):
#   bash deployments/frontier/setup/build-lammps-frontier.sh
# Env:  the LAMMPS_* variables of ../../common/setup/build-lammps.sh are passed through.
#       LAMMPS_GCC_MODULE, LAMMPS_MPICH_MODULE, LAMMPS_CMAKE_MODULE override the toolchain.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(CDPATH= cd -P -- "${SCRIPT_DIR}/../../.." && pwd -P)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lammps-toolchain-frontier.sh"

export LAMMPS_ROOT="${LAMMPS_ROOT:-${REPO}/.hpc-build/frontier/lammps}"
export LAMMPS_CXX="${LAMMPS_CXX:-CC}"
export LAMMPS_CC="${LAMMPS_CC:-cc}"
export XALT_EXECUTABLE_TRACKING=no

bash "${SCRIPT_DIR}/../../common/setup/build-lammps.sh"

# Jobs must load the same toolchain the binary was linked against.
cat >> "${LAMMPS_ROOT}/activate.sh" <<EOF
source "${SCRIPT_DIR}/lammps-toolchain-frontier.sh"
export XALT_EXECUTABLE_TRACKING=no
EOF
