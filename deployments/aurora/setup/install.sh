#!/usr/bin/env bash
# One-shot, self-contained matsim-agents + HydraGNN environment for ALCF Aurora.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(dirname "${MATSIM_DIR}")/HydraGNN}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_REF="${HYDRAGNN_REF:-}"
INSTALL_ROOT="${INSTALL_ROOT:-${MATSIM_DIR}/.hpc-build/aurora}"
VENV_PATH="${VENV_PATH:-${MATSIM_DIR}/.venv}"
MATSIM_EXTRAS="${MATSIM_EXTRAS:-hydragnn,dev,openai,ollama,anthropic,huggingface}"
INSTALL_UMA="${INSTALL_UMA:-0}"
UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"
INSTALL_MACE="${INSTALL_MACE:-0}"
MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"
RECREATE_ENV="${RECREATE_ENV:-0}"
REUSE_HYDRAGNN_DEPENDENCIES="${REUSE_HYDRAGNN_DEPENDENCIES:-0}"
RECREATE_MACE_ENV="${RECREATE_MACE_ENV:-${RECREATE_ENV}}"
RECREATE_UMA_ENV="${RECREATE_UMA_ENV:-${RECREATE_MACE_ENV}}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
RUN_MODEL_TESTS="${RUN_MODEL_TESTS:-0}"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks}"
HYDRAGNN_INSTALLER_REPO="${HYDRAGNN_INSTALLER_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_INSTALLER_REF="${HYDRAGNN_INSTALLER_REF:-main}"
HYDRAGNN_INSTALLER_DIR="${HYDRAGNN_INSTALLER_DIR:-${INSTALL_ROOT}/installer-source}"

log() { printf '\033[1;34m[aurora-install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[aurora-install]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "${MATSIM_DIR}/pyproject.toml" ]] || die "matsim-agents checkout not found: ${MATSIM_DIR}"
command -v git >/dev/null || die "git is required"

set +u
export ZSH_EVAL_CONTEXT="${ZSH_EVAL_CONTEXT:-}"
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
command -v module >/dev/null 2>&1 || die "Aurora environment modules are unavailable"
module reset
module load "${FRAMEWORKS_MODULE}"
set -u
FRAMEWORKS_PYTHON="$(command -v python3)"
"${FRAMEWORKS_PYTHON}" -c "import torch; print('frameworks torch', torch.__version__, torch.__file__)"
"${FRAMEWORKS_PYTHON}" - "${MATSIM_DIR}/pyproject.toml" "${MATSIM_EXTRAS}" <<'PY'
import sys
import tomllib

import torch
from packaging.requirements import Requirement

with open(sys.argv[1], "rb") as stream:
    project = tomllib.load(stream)["project"]
requirements = list(project.get("dependencies", []))
for extra in sys.argv[2].split(","):
    requirements.extend(project.get("optional-dependencies", {}).get(extra.strip(), []))
for value in requirements:
    requirement = Requirement(value)
    if requirement.name.lower().replace("_", "-") == "torch":
        if torch.__version__ not in requirement.specifier:
            raise SystemExit(
                f"Aurora frameworks provides torch {torch.__version__}, but Matsim-Agents "
                f"requires {requirement}. A compatible Aurora dependency profile is needed; "
                "refusing to replace the XPU stack with a pip wheel."
            )
PY

if [[ ! -d "${HYDRAGNN_DIR}/.git" ]]; then
    log "Cloning HydraGNN ${HYDRAGNN_REF:-main} into ${HYDRAGNN_DIR}"
    git clone --branch "${HYDRAGNN_REF:-main}" "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}"
elif [[ -n "${HYDRAGNN_REF}" ]]; then
    git -C "${HYDRAGNN_DIR}" diff --quiet && git -C "${HYDRAGNN_DIR}" diff --cached --quiet \
        || die "HydraGNN checkout has local changes: ${HYDRAGNN_DIR}"
    log "Updating HydraGNN checkout to ${HYDRAGNN_REF}"
    git -C "${HYDRAGNN_DIR}" fetch "${HYDRAGNN_REPO}" "${HYDRAGNN_REF}"
    git -C "${HYDRAGNN_DIR}" switch "${HYDRAGNN_REF}"
    git -C "${HYDRAGNN_DIR}" merge --ff-only FETCH_HEAD
else
    log "Preserving HydraGNN checkout $(git -C "${HYDRAGNN_DIR}" rev-parse --short HEAD)"
fi

[[ "$(realpath -m "${HYDRAGNN_INSTALLER_DIR}")" != "$(realpath "${HYDRAGNN_DIR}")" ]] \
    || die "Dependency source directory must not be the selected HydraGNN checkout"
if [[ ! -d "${HYDRAGNN_INSTALLER_DIR}/.git" ]]; then
    log "Fetching upstream dependencies ${HYDRAGNN_INSTALLER_REF} into ${HYDRAGNN_INSTALLER_DIR}"
    git clone --depth 1 --branch "${HYDRAGNN_INSTALLER_REF}" \
        "${HYDRAGNN_INSTALLER_REPO}" "${HYDRAGNN_INSTALLER_DIR}"
else
    git -C "${HYDRAGNN_INSTALLER_DIR}" diff --quiet \
        && git -C "${HYDRAGNN_INSTALLER_DIR}" diff --cached --quiet \
        || die "Upstream dependency checkout has local changes: ${HYDRAGNN_INSTALLER_DIR}"
    git -C "${HYDRAGNN_INSTALLER_DIR}" fetch --depth 1 origin "${HYDRAGNN_INSTALLER_REF}"
    git -C "${HYDRAGNN_INSTALLER_DIR}" checkout --detach FETCH_HEAD
fi
log "Upstream dependency revision $(git -C "${HYDRAGNN_INSTALLER_DIR}" rev-parse HEAD)"
HYDRAGNN_INSTALLER="${HYDRAGNN_INSTALLER:-${HYDRAGNN_INSTALLER_DIR}/scripts/hpc/alcf/aurora/installation/install.sh}"
[[ -f "${HYDRAGNN_INSTALLER}" ]] || die \
    "HydraGNN Aurora installer not found: ${HYDRAGNN_INSTALLER}. Set HYDRAGNN_INSTALLER to a current installer that supports INSTALL_ROOT and VENV_PATH; the selected checkout will not be switched automatically."

if [[ -x "${VENV_PATH}/bin/python" && "${RECREATE_ENV}" != "1" ]]; then
    "${VENV_PATH}/bin/python" - "${FRAMEWORKS_PYTHON}" <<'PY'
import pathlib
import sys
import torch

if pathlib.Path(sys._base_executable).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("Venv uses a different framework Python; rerun with RECREATE_ENV=1")
print("existing venv torch", torch.__version__)
PY
fi
if [[ "${RECREATE_ENV}" == "1" ]]; then
    [[ "${VENV_PATH}" = /* && "${VENV_PATH}" != "/" && "${VENV_PATH}" != "${MATSIM_DIR}" \
        && "${VENV_PATH}" != "${HYDRAGNN_DIR}" ]] || die "Refusing unsafe VENV_PATH: ${VENV_PATH}"
fi

log "Installing HydraGNN and its XPU/PyG/MPI dependencies first"
TORCH_VERSION="$("${FRAMEWORKS_PYTHON}" -c 'import torch; print(torch.__version__)')"
HYDRAGNN_PACKAGE_SRC="$("${FRAMEWORKS_PYTHON}" "${SCRIPT_DIR}/prepare_hydragnn_source.py" \
    --code "${HYDRAGNN_DIR}" --dependencies "${HYDRAGNN_INSTALLER_DIR}" \
    --output-root "${INSTALL_ROOT}" --torch-version "${TORCH_VERSION}")"
if [[ "${REUSE_HYDRAGNN_DEPENDENCIES}" == "1" ]]; then
    [[ "${RECREATE_ENV}" != "1" && -x "${VENV_PATH}/bin/python" ]] \
        || die "Dependency reuse requires an existing environment and RECREATE_ENV=0"
    log "Reusing installed HydraGNN dependencies; validating compiled PyG imports"
    "${VENV_PATH}/bin/python" -c \
        "import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv"
else
    INSTALL_ROOT="${INSTALL_ROOT}" VENV_PATH="${VENV_PATH}" RECREATE_ENV="${RECREATE_ENV}" \
    HYDRAGNN_SRC="${HYDRAGNN_PACKAGE_SRC}" bash "${HYDRAGNN_INSTALLER}"
fi

PYTHON="${VENV_PATH}/bin/python"
[[ -x "${PYTHON}" ]] || die "HydraGNN did not create ${VENV_PATH}"
log "Installing HydraGNN and matsim-agents wheels into the same environment"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "hatchling>=1.21"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed --no-deps --force-reinstall "${HYDRAGNN_PACKAGE_SRC}"
"${PYTHON}" -m pip install --no-build-isolation --upgrade-strategy only-if-needed "${MATSIM_DIR}[${MATSIM_EXTRAS}]"
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed hf_transfer
"${PYTHON}" -m pip check || log "Full pip check reported conflicts; checking selected runtime dependencies next"
"${PYTHON}" "${MATSIM_DIR}/deployments/common/setup/check-runtime-dependencies.py" \
    "matsim-agents[${MATSIM_EXTRAS}]" hydragnn
"${PYTHON}" -c "import hydragnn, matsim_agents, torch; print('verified', torch.__version__)"
if [[ "${RUN_MODEL_TESTS}" == "1" ]]; then
    INSTALL_UMA=1
    INSTALL_MACE=1
fi
if [[ "${INSTALL_UMA}" == "1" ]]; then
    log "Installing UMA in its Torch-compatible matsim-owned environment"
    MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${VENV_PATH}" UMA_VENV_PATH="${UMA_VENV_PATH}" \
        FACILITY=aurora RECREATE_UMA_ENV="${RECREATE_UMA_ENV}" \
        bash "${MATSIM_DIR}/deployments/common/setup/install-uma-compat.sh"
fi
if [[ "${INSTALL_MACE}" == "1" ]]; then
    log "Installing MACE in its e3nn-compatible matsim-owned environment"
    MATSIM_DIR="${MATSIM_DIR}" BASE_VENV="${VENV_PATH}" MACE_VENV_PATH="${MACE_VENV_PATH}" \
        FACILITY=aurora RECREATE_MACE_ENV="${RECREATE_MACE_ENV}" \
        bash "${MATSIM_DIR}/deployments/common/setup/install-mace-compat.sh"
fi
if [[ "${INSTALL_VLLM}" == "1" ]]; then
    log "Verifying the framework-provided vLLM-XPU stack"
    bash "${MATSIM_DIR}/deployments/aurora/setup/install-vllm-xpu-aurora.sh"
fi

if [[ "${RUN_MODEL_TESTS}" == "1" ]]; then
    log "Running the complete Matsim-Agents test suite"
    (
        cd "${MATSIM_DIR}"
        "${PYTHON}" -m pytest "${MATSIM_DIR}/tests"
    )
fi

log "Complete. Activate with: module load ${FRAMEWORKS_MODULE} && source ${VENV_PATH}/bin/activate"
