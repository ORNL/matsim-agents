#!/usr/bin/env bash
# fix_h5py_system_conflict_aurora.sh
#
# Problem
# -------
# On Aurora, the system Python site-packages at
#   /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages/
# ships h5py 3.15.1, but the HDF5 shared library it was compiled against
# (libhdf5.so.310) is NOT present in the default LD_LIBRARY_PATH on login/UAN nodes:
#
#   ImportError: libhdf5.so.310: cannot open shared object file: No such file or directory
#
# Root cause
# ----------
# When a venv is created with --system-site-packages (or without --copies), pip
# queries the system site-packages first. If h5py is not explicitly installed
# *inside* the venv, Python resolves the import to the system copy, which needs
# the missing libhdf5.so.310.
#
# Triggered by
# ------------
# Installing mace-torch pulls h5py as a dependency. Because h5py is already
# "visible" from the system site-packages, pip skips reinstalling it into the
# venv. At runtime the broken system copy is used.
#
# Fix
# ---
# Force-reinstall h5py from PyPI inside the target venv. The PyPI wheel bundles
# its own HDF5 shared library (libhdf5 is statically linked or wheel-local),
# so no external libhdf5.so is needed.
#
# This fix is SAFE for HydraGNN: HydraGNN does not directly depend on h5py,
# and the reinstalled version (3.16+) is fully backward-compatible.
#
# Usage
# -----
#   bash fix_h5py_system_conflict_aurora.sh [<path/to/venv>]
#
# If no venv path is given the currently-activated venv ($VIRTUAL_ENV) is used.

set -Eeuo pipefail

VENV="${1:-${VIRTUAL_ENV:-}}"

if [[ -z "$VENV" ]]; then
  echo "ERROR: no venv path given and no venv is currently active."
  echo "Usage: bash $0 /path/to/venv"
  exit 1
fi

echo "Venv: $VENV"
echo "Python: $("${VENV}/bin/python" --version)"

echo ""
echo "Reinstalling h5py from PyPI into the venv (--no-deps to preserve pinned numpy)..."
"${VENV}/bin/pip" install --force-reinstall --no-deps h5py

echo ""
echo "Verifying import..."
"${VENV}/bin/python" -c "import h5py; print(f'  h5py {h5py.__version__} OK  ({h5py.__file__})')"
echo "Done."
