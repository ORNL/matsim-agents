#!/bin/bash

# Resolve repository root from a script directory, with a fallback path.
resolve_repo_root() {
  local script_dir=$1
  local fallback_repo=$2
  local repo
  repo="$(cd "${script_dir}/../../.." 2>/dev/null && pwd)"
  if [[ ! -f "${repo}/pyproject.toml" ]]; then
    repo="${fallback_repo}"
  fi
  printf '%s\n' "${repo}"
}

# Create run/output directories and export RUN_DIR/OUTPUT_DIR.
init_run_dirs() {
  local proj_root=$1
  local run_prefix=$2
  local job_id=$3
  RUN_DIR="${proj_root}/runs/${run_prefix}-${job_id}"
  OUTPUT_DIR="${RUN_DIR}/outputs"
  mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"
  export RUN_DIR OUTPUT_DIR
}