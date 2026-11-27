#!/usr/bin/env bash
# Install mpi4py into a matsim-owned environment, built against the facility MPI.
# Multi-rank LAMMPS ML-IAP runs gather atoms with mpi4py. Usage: install-mpi4py.sh <python>
# Env: FACILITY (perlmutter uses the Cray `cc` wrapper, as HydraGNN's installer does),
# MPICC (override), MPI4PY_VERSION (default matches HydraGNN's pin).
set -Eeuo pipefail

PYTHON="${1:?usage: install-mpi4py.sh <python>}"
FACILITY="${FACILITY:-unknown}"
MPI4PY_VERSION="${MPI4PY_VERSION:-4.1.1}"

if [[ -z "${MPICC:-}" ]]; then
    if [[ "${FACILITY}" == "perlmutter" ]]; then
        MPICC=cc
    else
        MPICC="$(command -v mpicc || command -v cc || true)"
    fi
fi
command -v "${MPICC:-}" >/dev/null 2>&1 || {
    printf '\033[1;31m[mpi4py]\033[0m no MPI compiler wrapper found; load the MPI module or set MPICC\n' >&2
    exit 1
}

# --ignore-installed: environments that inherit the base .venv would otherwise reuse its copy.
# --no-binary: the PyPI manylinux wheel links a generic libmpi, not the facility MPI.
MPICC="${MPICC}" "${PYTHON}" -m pip install --ignore-installed \
    --no-binary mpi4py "mpi4py==${MPI4PY_VERSION}"
# The MPI runtime libraries are on the loader path only once the facility modules are loaded.
"${PYTHON}" -c "from mpi4py import MPI; print('mpi4py', MPI.Get_library_version().splitlines()[0])" \
    || printf '\033[1;33m[mpi4py]\033[0m installed, but importing MPI failed; check the MPI module and LD_LIBRARY_PATH\n' >&2
