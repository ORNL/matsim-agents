#!/usr/bin/env bash
# One-shot, self-contained matsim-agents + HydraGNN environment for ALCF Aurora.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(dirname "${MATSIM_DIR}")/HydraGNN}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_REF="${HYDRAGNN_REF:-main}"
INSTALL_ROOT="${INSTALL_ROOT:-${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Aurora}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
MATSIM_EXTRAS="${MATSIM_EXTRAS:-dev,openai,ollama,anthropic,huggingface}"
RECREATE_ENV="${RECREATE_ENV:-0}"

log() { printf '\033[1;34m[aurora-install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[aurora-install]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "${MATSIM_DIR}/pyproject.toml" ]] || die "matsim-agents checkout not found: ${MATSIM_DIR}"
command -v git >/dev/null || die "git is required"

if [[ ! -d "${HYDRAGNN_DIR}/.git" ]]; then
    log "Cloning HydraGNN ${HYDRAGNN_REF} into ${HYDRAGNN_DIR}"
    git clone --branch "${HYDRAGNN_REF}" "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}"
else
    git -C "${HYDRAGNN_DIR}" diff --quiet && git -C "${HYDRAGNN_DIR}" diff --cached --quiet \
        || die "HydraGNN checkout has local changes: ${HYDRAGNN_DIR}"
    log "Updating HydraGNN checkout to ${HYDRAGNN_REF}"
    git -C "${HYDRAGNN_DIR}" fetch origin "${HYDRAGNN_REF}"
    git -C "${HYDRAGNN_DIR}" checkout --detach FETCH_HEAD
fi

HYDRAGNN_INSTALLER="${HYDRAGNN_DIR}/scripts/hpc/alcf/aurora/installation/install.sh"
[[ -f "${HYDRAGNN_INSTALLER}" ]] || die "HydraGNN Aurora installer not found: ${HYDRAGNN_INSTALLER}"

log "Installing HydraGNN and its XPU/PyG/MPI dependencies first"
INSTALL_ROOT="${INSTALL_ROOT}" VENV_PATH="${VENV_PATH}" RECREATE_ENV="${RECREATE_ENV}" \
HYDRAGNN_SRC="${HYDRAGNN_DIR}" bash "${HYDRAGNN_INSTALLER}"

PYTHON="${VENV_PATH}/bin/python"
[[ -x "${PYTHON}" ]] || die "HydraGNN did not create ${VENV_PATH}"
log "Installing non-editable HydraGNN and matsim-agents packages in that environment"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed --no-deps --force-reinstall "${HYDRAGNN_DIR}"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "${MATSIM_DIR}[${MATSIM_EXTRAS}]"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed hf_transfer
"${PYTHON}" -m pip check
"${PYTHON}" -c "import hydragnn, matsim_agents, torch; print('verified', torch.__version__)"

log "Complete. Activate with: module load frameworks && source ${VENV_PATH}/bin/activate"
