#!/bin/bash
# Single source of the HydraGNN code location for every job script. Source it; do not run it.
#
#   source "$REPO/deployments/common/hydragnn-env.sh"
#   LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_BEST6_LOGDIR}
#   hydragnn_export_pythonpath
#
# The code (HYDRAGNN_ROOT) is the checkout pinned in benchmarks/e4_mlip/frozen-source.json. It is
# checked out at that commit with no uncommitted changes to tracked files, otherwise this script
# fails. The BEST6 weights (HYDRAGNN_BEST6_LOGDIR) are data and live in a different tree on
# purpose; src/matsim_agents/backends/mlip/hydragnn_sources.json ties their checkpoint hash to
# the same commit, and matsim-agents re-checks that when it loads them.
#
# Overrides: HYDRAGNN_ROOT (legacy HYDRAGNN_DIR), HYDRAGNN_WEIGHTS. Any override must still
# resolve to the pinned commit.

_hg_common=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hg_repo=$(cd "${_hg_common}/../.." && pwd)
_hg_proj=${PROJ:-$(dirname "${_hg_repo}")}
_hg_pin="${_hg_repo}/benchmarks/e4_mlip/frozen-source.json"

HYDRAGNN_PINNED_COMMIT=$(grep -o '"commit": *"[0-9a-f]\{40\}"' "${_hg_pin}" | grep -o '[0-9a-f]\{40\}')
if [[ -z "${HYDRAGNN_PINNED_COMMIT}" ]]; then
  echo "hydragnn-env: no 40-character commit in ${_hg_pin}" >&2
  return 1 2>/dev/null || exit 1
fi

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-${HYDRAGNN_DIR:-${_hg_proj}/HydraGNN-e4b-frozen-${HYDRAGNN_PINNED_COMMIT:0:8}}}
HYDRAGNN_DIR=${HYDRAGNN_ROOT}
HYDRAGNN_EXAMPLE=${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26
_hg_best6=examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64
if [[ -z "${HYDRAGNN_WEIGHTS:-}" ]]; then
  # Aurora keeps BEST6 inside the pinned checkout; Frontier and Perlmutter keep it in the old tree.
  if [[ -d "${HYDRAGNN_ROOT}/${_hg_best6}" ]]; then
    HYDRAGNN_WEIGHTS=${HYDRAGNN_ROOT}
  else
    HYDRAGNN_WEIGHTS=${_hg_proj}/HydraGNN
  fi
fi
HYDRAGNN_BEST6_LOGDIR=${HYDRAGNN_WEIGHTS}/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64
export HYDRAGNN_ROOT HYDRAGNN_DIR HYDRAGNN_EXAMPLE HYDRAGNN_WEIGHTS HYDRAGNN_BEST6_LOGDIR

_hg_actual=$(git -C "${HYDRAGNN_ROOT}" rev-parse HEAD 2>/dev/null || true)
if [[ "${_hg_actual}" != "${HYDRAGNN_PINNED_COMMIT}" ]]; then
  echo "hydragnn-env: ${HYDRAGNN_ROOT} is at '${_hg_actual:-not a git checkout}', not the pinned" \
    "${HYDRAGNN_PINNED_COMMIT} (see docs/hydragnn-frozen-source.md)" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -n "$(git -C "${HYDRAGNN_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "hydragnn-env: ${HYDRAGNN_ROOT} has uncommitted changes to tracked files" >&2
  return 1 2>/dev/null || exit 1
fi

hydragnn_export_pythonpath() {
  export PYTHONPATH="${HYDRAGNN_EXAMPLE}:${HYDRAGNN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

unset _hg_common _hg_repo _hg_proj _hg_pin _hg_actual _hg_best6
