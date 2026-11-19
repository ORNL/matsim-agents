#!/bin/bash

# Print the UMA model_name of a YAML's mlip block (variables expanded, paths
# resolved) with the given Python, which must import matsim_agents.
mlip_uma_model_name() {
  local python="$1" config="$2"
  "$python" -c '
import sys

from matsim_agents.backends.mlip.config import load_mlip_config

print(load_mlip_config(sys.argv[1]).model_name)
' "$config"
}

# Export the durable UMA bundle root and check the bundle of MODEL_NAME (default
# MATSIM_UMA_MODEL_NAME, then uma-s-1p1). A model_name that is an existing file
# or directory is a local checkpoint and needs no bundle.
configure_uma_model_artifacts() {
  local repo="$1" proj durable_root model_name bundle
  proj="$(dirname "${repo}")"
  durable_root="${MATSIM_MODEL_ARTIFACTS_ROOT:-${proj}/models/artifacts}"
  MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-${durable_root}/uma}"
  model_name="${2:-${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}}"
  export MATSIM_UMA_ARTIFACT_DIR
  [[ ! -e "${model_name}" ]] || return 0
  bundle="${MATSIM_UMA_ARTIFACT_DIR}/${model_name}"
  [[ -s "${bundle}/checkpoint.pt" && -s "${bundle}/atom_refs.yaml" ]] || {
    echo "ERROR: durable UMA bundle is missing or incomplete: ${bundle}" >&2
    echo "Run deployments/perlmutter/download/download-uma-perlmutter.sh first." >&2
    return 2
  }
}

configure_mace_model_artifacts() {
  local repo="$1" proj durable_root
  proj="$(dirname "${repo}")"
  durable_root="${MATSIM_MODEL_ARTIFACTS_ROOT:-${proj}/models/artifacts}"
  MACE_ARTIFACT_DIR="${MACE_ARTIFACT_DIR:-${durable_root}/mace}"
  XDG_CACHE_HOME="${MACE_ARTIFACT_DIR}"
  MACE_CACHE="${MACE_ARTIFACT_DIR}/mace"
  mkdir -p "${MACE_CACHE}"
  export MACE_ARTIFACT_DIR XDG_CACHE_HOME MACE_CACHE
}