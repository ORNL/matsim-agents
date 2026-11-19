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

# Print the mlip.backend of a YAML with an `mlip:` block (PYTHON needs PyYAML).
mlip_backend() {
  local python=$1 config=$2
  "${python}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["mlip"]["backend"])' \
    "${config}"
}

# Print the Python environment that runs BACKEND. HYDRAGNN_VENV is the site's
# HydraGNN environment; UMA and MACE use their isolated environments.
mlip_backend_venv() {
  local repo=$1 backend=$2 hydragnn_venv=$3
  case "${backend}" in
    hydragnn) printf '%s\n' "${hydragnn_venv}" ;;
    uma) printf '%s\n' "${MATSIM_FAIRCHEM_VENV:-${repo}/.venv-uma}" ;;
    mace) printf '%s\n' "${MATSIM_MACE_VENV:-${repo}/.venv-mace}" ;;
    *) echo "ERROR: unsupported MLIP backend: ${backend}" >&2; return 2 ;;
  esac
}

# Print MLIP_CONFIG, or write and print a HydraGNN foundation-model selection
# (LOGDIR weighted by its BranchWeightMLP) in RUN_DIR when it is unset.
default_mlip_config() {
  local run_dir=$1 logdir=$2 branch_mlp=$3
  if [[ -n "${MLIP_CONFIG:-}" ]]; then
    printf '%s\n' "${MLIP_CONFIG}"
    return 0
  fi
  printf 'mlip:\n  backend: hydragnn\n  logdir: "%s"\n  mlp_checkpoint: "%s"\n' \
    "${logdir}" "${branch_mlp}" >"${run_dir}/mlip.yaml"
  printf '%s\n' "${run_dir}/mlip.yaml"
}

# Activate a virtualenv, or a conda environment directory without bin/activate.
activate_python_env() {
  local env=$1
  if [[ -f "${env}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${env}/bin/activate"
  else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env}"
  fi
}
