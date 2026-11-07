#!/bin/bash
# Default runtime for the tested Aurora facility and custom oneAPI binaries.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${init}" ]]; then
      # shellcheck disable=SC1090
      source "${init}"
      break
    fi
  done
fi
command -v module >/dev/null 2>&1 || { echo "Aurora VASP requires environment modules" >&2; exit 2; }
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/aurora-module-stack.sh"
load_aurora_oneapi_modules
module load "${VASP_MPICH_MODULE:-mpich/prd/5.0.0.aurora_test.51a9474}"
