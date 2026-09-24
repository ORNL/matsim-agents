#!/usr/bin/env bash
# One-shot, self-contained matsim-agents + HydraGNN environment for OLCF Frontier.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(dirname "${MATSIM_DIR}")/HydraGNN}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_REF="${HYDRAGNN_REF:-main}"
INSTALL_ROOT="${INSTALL_ROOT:-${MATSIM_DIR}/.hpc-build/frontier}"
VENV_PATH="${VENV_PATH:-${MATSIM_DIR}/.venv}"
MATSIM_EXTRAS="${MATSIM_EXTRAS:-hydragnn,dev,openai,ollama,anthropic,huggingface}"
INSTALL_UMA="${INSTALL_UMA:-0}"
UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"
INSTALL_MACE="${INSTALL_MACE:-0}"
MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"
RECREATE_ENV="${RECREATE_ENV:-0}"
SKIP_VLLM="${SKIP_VLLM:-1}"

log() { printf '\033[1;34m[frontier-install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[frontier-install]\033[0m %s\n' "$*" >&2; exit 1; }

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

HYDRAGNN_INSTALLER="${HYDRAGNN_DIR}/scripts/hpc/olcf/frontier/installation/install-rocm72.sh"
[[ -f "${HYDRAGNN_INSTALLER}" ]] || die "HydraGNN Frontier installer not found: ${HYDRAGNN_INSTALLER}"

# HydraGNN's Frontier script derives its build root from PWD, while VENV_PATH
# remains authoritative. Keep both the build tree and environment under matsim.
mkdir -p "${INSTALL_ROOT}"
log "Installing HydraGNN and its ROCm/PyG/MPI dependencies first"
(
    cd "${INSTALL_ROOT}"
    VENV_PATH="${VENV_PATH}" RECREATE_ENV="${RECREATE_ENV}" SKIP_VLLM="${SKIP_VLLM}" \
        bash "${HYDRAGNN_INSTALLER}"
)

PYTHON="${VENV_PATH}/bin/python"
[[ -x "${PYTHON}" ]] || die "HydraGNN did not create ${VENV_PATH}"
log "Installing HydraGNN and matsim-agents wheels into the same environment"
# --no-build-isolation: avoids pip's PEP 517 isolated build env, which needs
# network/TLS to fetch build backends; those backends are pre-installed here
# instead (setuptools/wheel already present for HydraGNN's setup.py).
"${PYTHON}" -m pip install --no-build-isolation --upgrade-strategy only-if-needed --no-deps "${HYDRAGNN_DIR}"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "hatchling>=1.21"
"${PYTHON}" -m pip install --no-build-isolation --upgrade-strategy only-if-needed "${MATSIM_DIR}[${MATSIM_EXTRAS}]"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed hf_transfer
"${PYTHON}" -m pip check
"${PYTHON}" -c "import hydragnn, matsim_agents, torch; print('verified', torch.__version__)"
if [[ "${INSTALL_UMA}" == "1" ]]; then
    log "Installing UMA in its Torch-compatible matsim-owned environment"
    MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${VENV_PATH}" UMA_VENV_PATH="${UMA_VENV_PATH}" \
        RECREATE_UMA_ENV="${RECREATE_ENV}" \
        bash "${MATSIM_DIR}/deployments/common/setup/install-uma-compat.sh"
fi
if [[ "${INSTALL_MACE}" == "1" ]]; then
    log "Installing MACE in its e3nn-compatible matsim-owned environment"
    MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${VENV_PATH}" MACE_VENV_PATH="${MACE_VENV_PATH}" \
        FACILITY=frontier RECREATE_MACE_ENV="${RECREATE_ENV}" \
        bash "${MATSIM_DIR}/deployments/common/setup/install-mace-compat.sh"
fi

log "Complete. Activate with: source ${VENV_PATH}/bin/activate"
