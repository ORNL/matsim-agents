#!/bin/bash

configure_uma_model_artifacts() {
  local repo="$1" proj durable_root model_name bundle
  proj="$(dirname "${repo}")"
  durable_root="${MATSIM_MODEL_ARTIFACTS_ROOT:-${proj}/models/artifacts}"
  MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-${durable_root}/uma}"
  model_name="${MATSIM_UMA_MODEL_NAME:-uma-s-1p1}"
  bundle="${MATSIM_UMA_ARTIFACT_DIR}/${model_name}"
  [[ -s "${bundle}/checkpoint.pt" && -s "${bundle}/atom_refs.yaml" ]] || {
    echo "ERROR: durable UMA bundle is missing or incomplete: ${bundle}" >&2
    echo "Run deployments/perlmutter/download/download-uma-perlmutter.sh first." >&2
    return 2
  }
  export MATSIM_UMA_ARTIFACT_DIR
}

configure_mace_model_artifacts() {
  local repo="$1" proj durable_root legacy_root
  proj="$(dirname "${repo}")"
  durable_root="${MATSIM_MODEL_ARTIFACTS_ROOT:-${proj}/models/artifacts}"
  legacy_root="${proj}/models/mace_cache"
  MACE_ARTIFACT_DIR="${MACE_ARTIFACT_DIR:-${durable_root}/mace}"
  if [[ ! -d "${MACE_ARTIFACT_DIR}/mace" && -d "${legacy_root}/mace" ]]; then
    mkdir -p "$(dirname "${MACE_ARTIFACT_DIR}")"
    mv "${legacy_root}" "${MACE_ARTIFACT_DIR}"
  fi
  XDG_CACHE_HOME="${MACE_ARTIFACT_DIR}"
  MACE_CACHE="${MACE_ARTIFACT_DIR}/mace"
  mkdir -p "${MACE_CACHE}"
  export MACE_ARTIFACT_DIR XDG_CACHE_HOME MACE_CACHE
}