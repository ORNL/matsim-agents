#!/usr/bin/env bash
# =============================================================================
# install_matsim_aurora.sh
#
# Two-phase Python venv install for matsim-agents + HydraGNN on Aurora-like
# environments.
#
# Phase 1: Create venv and install HydraGNN dependencies first.
# Phase 2: Install matsim-agents and additional runtime/tooling dependencies.
#
# Usage:
#   bash install_matsim_aurora.sh
#
# Optional overrides:
#   PROJECT_DIR       Root project directory
#   MATSIM_DIR        matsim-agents checkout path
#   HYDRAGNN_DIR      HydraGNN checkout path
#   VENV_PATH         Python virtual environment path
#   PYTHON_BIN        Python binary used to create venv (default: auto-detect)
#   LLM_BACKENDS      matsim-agents extras (default: dev)
#   INSTALL_VLLM_SERVER  Install vllm package (default: 0)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../../..}"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/aurora_venv}"
LLM_BACKENDS="${LLM_BACKENDS:-dev}"
INSTALL_VLLM_SERVER="${INSTALL_VLLM_SERVER:-0}"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

pick_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "${PYTHON_BIN}"
        return 0
    fi

    for cand in python3.11 python3.10 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            echo "$cand"
            return 0
        fi
    done

    return 1
}

parse_major_minor() {
    local pybin="$1"
    "$pybin" - <<'PYEOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PYEOF
}

pip_retry() {
    local tries=3 delay=3
    for ((i=1; i<=tries; i++)); do
        if pip install --upgrade-strategy only-if-needed "$@"; then
            return 0
        fi
        warn "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
        sleep "$delay"
        delay=$((delay * 2))
    done
    return 1
}

[[ -d "${MATSIM_DIR}" ]] || die "matsim-agents directory not found: ${MATSIM_DIR}"
[[ -d "${HYDRAGNN_DIR}" ]] || die "HydraGNN directory not found: ${HYDRAGNN_DIR}"

PYTHON_BIN="$(pick_python)" || die "No suitable Python interpreter found. Set PYTHON_BIN explicitly."
PY_MM="$(parse_major_minor "${PYTHON_BIN}")"
log "Using Python interpreter: ${PYTHON_BIN} (${PY_MM})"

PY_MAJOR="${PY_MM%%.*}"
PY_MINOR="${PY_MM##*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
    die "Python >= 3.10 required, found ${PY_MM}. Set PYTHON_BIN to a newer interpreter."
fi

# -- Phase 1: HydraGNN environment --------------------------------------------
log "Phase 1: creating virtual environment at ${VENV_PATH}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

log "Upgrading pip/setuptools/wheel"
pip install -U pip setuptools wheel

HY_REQS=(
    "${HYDRAGNN_DIR}/requirements-base.txt"
    "${HYDRAGNN_DIR}/requirements-torch.txt"
    "${HYDRAGNN_DIR}/requirements-pyg.txt"
)

for req in "${HY_REQS[@]}"; do
    if [[ -f "$req" ]]; then
        log "Installing HydraGNN dependencies from $(basename "$req")"
        pip_retry -r "$req"
    else
        warn "Missing HydraGNN requirement file: $req"
    fi
done

if [[ -f "${HYDRAGNN_DIR}/setup.py" || -f "${HYDRAGNN_DIR}/pyproject.toml" ]]; then
    log "Installing HydraGNN editable package from ${HYDRAGNN_DIR}"
    pip_retry -e "${HYDRAGNN_DIR}" --no-deps
else
    die "HydraGNN project metadata not found in ${HYDRAGNN_DIR}"
fi

# -- Phase 2: matsim-agents extras --------------------------------------------
if [[ -f "${MATSIM_DIR}/pyproject.toml" ]]; then
    log "Installing matsim-agents[${LLM_BACKENDS}] editable package"
    pip_retry -e "${MATSIM_DIR}[${LLM_BACKENDS}]"
elif [[ -f "${MATSIM_DIR}/setup.py" ]]; then
    log "Installing matsim-agents editable package"
    pip_retry -e "${MATSIM_DIR}"
else
    die "matsim-agents project metadata not found in ${MATSIM_DIR}"
fi

log "Installing additional runtime/tooling dependencies"
pip_retry "langchain-core>=0.3.0" "pytest>=8.0" "pytest-cov>=5.0"
pip_retry "huggingface_hub>=1.12" "transformers>=4.45" "accelerate>=1.13"

if [[ "${INSTALL_VLLM_SERVER}" == "1" ]]; then
    log "INSTALL_VLLM_SERVER=1 -> installing vllm"
    pip_retry vllm
fi

# Re-assert HydraGNN base pins after optional extras change transitive versions.
if [[ -f "${HYDRAGNN_DIR}/requirements-base.txt" ]]; then
    log "Re-asserting HydraGNN base pins"
    pip_retry -r "${HYDRAGNN_DIR}/requirements-base.txt"
fi

log "Verifying installation imports"
python -c "import hydragnn; print('HydraGNN import: OK')"
python -c "import matsim_agents; print('matsim-agents import: OK')"
python -c "import huggingface_hub, transformers, accelerate; print('LLM tooling imports: OK')"

log "================================================================"
log "Installation complete"
log "Activate later with:"
log "  source ${VENV_PATH}/bin/activate"
log "================================================================"
