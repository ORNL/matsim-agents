#!/bin/bash
# Opt in only for dedicated matsim MACE inference workers; see the runbook.
# Children inherit this flag, and each calculator factory applies the shim
# before importing MACE. No installed package or global environment is changed.
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/mace-inference.sh PYTHON SCRIPT_OR_MODULE [ARGS...]" >&2
    exit 2
fi
export MATSIM_MACE_SKIP_TRANSFORMERS=1
exec "$@"
