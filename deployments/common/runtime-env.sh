#!/bin/bash

# Resolve the checkout without embedding a project, account, or username.
# PROJECT_ROOT is the authoritative override for scheduler spool copies.
resolve_repo_root() {
  local script_dir=$1
  local candidate
  for candidate in \
    "${PROJECT_ROOT:-}" \
    "$(cd "${script_dir}/../../.." 2>/dev/null && pwd || true)" \
    "${SLURM_SUBMIT_DIR:-}" \
    "${PBS_O_WORKDIR:-}"; do
    if [[ -n "${candidate}" && -f "${candidate}/pyproject.toml" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "ERROR: cannot locate matsim-agents; export PROJECT_ROOT before submission" >&2
  return 2
}

init_run_dirs() {
  local proj_root=$1
  local run_prefix=$2
  local job_id=$3
  local runs_root=${RUNS_ROOT:-${proj_root}/runs}
  RUN_DIR="${runs_root}/${run_prefix}-${job_id}"
  OUTPUT_DIR="${RUN_DIR}/outputs"
  mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}"
  export RUN_DIR OUTPUT_DIR
}
