#!/usr/bin/env bash
# =============================================================================
# install_matsim_perlmutter.sh
#
# Frontier-style phased install for matsim-agents + HydraGNN on Perlmutter.
#
# Phases:
#   Phase 0  - Load Perlmutter modules
#   Phase 1  - Build fresh HydraGNN Perlmutter environment
#   Phase 2  - Activate env and install matsim-agents + runtime extras
#
# Usage:
#   bash install_matsim_perlmutter.sh [--gpu]
#
# Flags:
#   --gpu             Kept for compatibility; installer already targets A100.
#
# Configurable variables (override via environment before calling script):
#   PROJECT_DIR       Project root                     (default: script/../..)
#   MATSIM_DIR        matsim-agents checkout path      (default: PROJECT_DIR)
#   HYDRAGNN_DIR      HydraGNN checkout path           (default: ../HydraGNN)
#   HYDRAGNN_REPO     HydraGNN git remote              (default: ORNL/HydraGNN)
#   HYDRAGNN_BRANCH   HydraGNN branch                  (default: main)
#   MATSIM_REPO       matsim-agents git remote         (default: ORNL/matsim-agents)
#   VENV_PATH         Target conda env path            (default: HydraGNN-Installation-Perlmutter/hydragnn_venv)
#   PYTHON_VERSION    Python version for env creation  (default: 3.11)
#   EXPECTED_CUDA_MM  CUDA major.minor                 (default: 12.9)
#   TORCH_CUDA_TAG    PyTorch wheel tag                (default: cu129)
#   TORCH_CUDA_ARCH   GPU arch list                    (default: 8.0)
#   MAX_JOBS          Build parallelism                (default: 16)
#   LLM_BACKENDS      matsim extras                    (default: dev)
#   INSTALL_VLLM_SERVER  Install vLLM package          (default: 0)
# =============================================================================
set -euo pipefail

# -- Configurable paths --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../../..}"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_BRANCH="${HYDRAGNN_BRANCH:-main}"
MATSIM_REPO="${MATSIM_REPO:-https://github.com/ORNL/matsim-agents.git}"

DEFAULT_VENV_PATH="${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv"
VENV_PATH="${VENV_PATH:-${DEFAULT_VENV_PATH}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
EXPECTED_CUDA_MM="${EXPECTED_CUDA_MM:-12.9}"
# Pin to HydraGNN's torch==2.11.0 wheels (published as cu129). cu129 binaries
# match the loaded cudatoolkit/12.9 module exactly.
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu129}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.0}"
MAX_JOBS="${MAX_JOBS:-16}"
LLM_BACKENDS="${LLM_BACKENDS:-dev}"
INSTALL_VLLM_SERVER="${INSTALL_VLLM_SERVER:-0}"

# -- Helpers ------------------------------------------------------------------
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

PIP_FLAGS=(--upgrade-strategy only-if-needed)
pip_retry() {
    local tries=3 delay=3
    for ((i=1; i<=tries; i++)); do
        if pip install "${PIP_FLAGS[@]}" "$@"; then
            return 0
        fi
        warn "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
        sleep "$delay"
        delay=$((delay * 2))
    done
    return 1
}

# -- Parse args ----------------------------------------------------------------
USE_GPU=0
for arg in "$@"; do
    [[ "$arg" == "--gpu" ]] && USE_GPU=1
done

# -- Phase 0: Load modules -----------------------------------------------------
log "Loading Perlmutter modules..."

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        source /usr/share/Modules/init/bash
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    die "module command not found. Ensure you're running on a Perlmutter login node."
fi

if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
    source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
fi

module reset
ml nersc-default/1.0 || true
ml cpe/24.07
ml PrgEnv-gnu/8.5.0
ml cray-mpich/8.1.30
ml craype-accel-nvidia80
ml "cudatoolkit/${EXPECTED_CUDA_MM}"
ml gcc-native/13.2
ml cmake/3.30.2 || ml cmake/3.24.3 || true
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0 || true

if [[ "$USE_GPU" -eq 0 ]]; then
    warn "--gpu not provided; proceeding anyway because Perlmutter install targets A100/CUDA by default."
fi

log "Modules loaded (CUDA ${EXPECTED_CUDA_MM})."

# -- Bootstrap repos if missing (Frontier parity) ------------------------------
if [[ ! -d "${MATSIM_DIR}/.git" ]]; then
    log "Cloning matsim-agents -> ${MATSIM_DIR}"
    git clone "${MATSIM_REPO}" "${MATSIM_DIR}"
else
    log "matsim-agents already present at ${MATSIM_DIR}"
fi

if [[ ! -d "${HYDRAGNN_DIR}/.git" ]]; then
    log "Cloning HydraGNN (${HYDRAGNN_BRANCH}) -> ${HYDRAGNN_DIR}"
    git clone --branch "${HYDRAGNN_BRANCH}" "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}" \
        || git clone "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}"
else
    log "HydraGNN already present at ${HYDRAGNN_DIR}"
fi

# -- Conda shell init ----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    die "conda command not found (Miniforge module not loaded?)"
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)" 2>/dev/null || true
fi

# -- Phase 1: Build fresh HydraGNN env ----------------------------------------
SC_INSTALLER="${HYDRAGNN_DIR}/installation_DOE_supercomputers/hydragnn_installation_bash_script_perlmutter.sh"
[[ -f "${SC_INSTALLER}" ]] || die "HydraGNN Perlmutter installer not found: ${SC_INSTALLER}"

INSTALL_ROOT="$(dirname "${VENV_PATH}")"
mkdir -p "${INSTALL_ROOT}"

log "Running HydraGNN Perlmutter installer (Frontier-style delegated Phase 1)..."
log "Installer: ${SC_INSTALLER}"
INSTALL_ROOT="${INSTALL_ROOT}" \
VENV_PATH="${VENV_PATH}" \
PYTHON_VERSION="${PYTHON_VERSION}" \
EXPECTED_CUDA_MM="${EXPECTED_CUDA_MM}" \
TORCH_CUDA_TAG="${TORCH_CUDA_TAG}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH}" \
MAX_JOBS="${MAX_JOBS}" \
bash "${SC_INSTALLER}"

log "HydraGNN environment build complete."

[[ -d "${VENV_PATH}" ]] \
    || die "Expected conda env not found at: ${VENV_PATH}"

# -- Phase 2: Activate env and install matsim-agents ---------------------------
log "Activating conda env: ${VENV_PATH}"
conda activate "${VENV_PATH}"

PYTHON_VER="$(python --version 2>&1)"
log "Active Python: ${PYTHON_VER}"
[[ "${PYTHON_VER}" == *"3.11"* ]] || warn "Expected Python 3.11 - got ${PYTHON_VER}. Proceeding anyway."

pip_retry --disable-pip-version-check -U pip setuptools wheel

if [[ -f "${HYDRAGNN_DIR}/setup.py" || -f "${HYDRAGNN_DIR}/pyproject.toml" ]]; then
    log "Installing HydraGNN editable package from ${HYDRAGNN_DIR}..."
    # Keep the CUDA/PyG stack built in Phase 1 intact.
    # Re-resolving HydraGNN deps here can downgrade torch and break ABI-compatible
    # extensions (e.g., torch-scatter).
    pip_retry -e "${HYDRAGNN_DIR}" --no-deps
else
    warn "HydraGNN package metadata not found at ${HYDRAGNN_DIR}; import may fail."
fi

if [[ -f "${MATSIM_DIR}/pyproject.toml" ]]; then
    log "Installing matsim-agents[${LLM_BACKENDS}] (editable) into ${VENV_PATH}..."
    pip_retry -e "${MATSIM_DIR}[${LLM_BACKENDS}]"
elif [[ -f "${MATSIM_DIR}/setup.py" ]]; then
    log "Installing matsim-agents editable from setup.py..."
    pip_retry -e "${MATSIM_DIR}"
else
    die "matsim-agents project metadata not found at ${MATSIM_DIR}"
fi

# Keep runtime/test extras aligned across Perlmutter and Frontier installers.
log "Ensuring core runtime/test dependencies (langchain-core, pytest, pytest-cov)..."
pip_retry "langchain-core>=0.3.0" "pytest>=8.0" "pytest-cov>=5.0"

log "Ensuring runtime dependencies used by fused HydraGNN path..."
pip_retry "scikit-learn==1.5.1" "vesin==0.4.2"

log "Installing LLM tooling extras (huggingface_hub CLI + transformers + accelerate)..."
pip_retry "huggingface_hub>=1.12" "transformers>=4.45" "accelerate>=1.13"

if [[ "${INSTALL_VLLM_SERVER}" == "1" ]]; then
    log "INSTALL_VLLM_SERVER=1 -> installing vLLM server package"
    pip_retry vllm
fi

# -- Re-assert HydraGNN-pinned versions ----------------------------------------
# matsim-agents and its extras pull transitive deps (typer→click, ase→matplotlib, ...)
# that may upgrade packages HydraGNN pins. Reinstall HydraGNN's base requirements
# so the final environment matches HydraGNN's source-of-truth pins exactly.
HYDRAGNN_BASE_REQ="${HYDRAGNN_DIR}/requirements-base.txt"
if [[ -f "${HYDRAGNN_BASE_REQ}" ]]; then
    log "Re-asserting HydraGNN base pins from ${HYDRAGNN_BASE_REQ}"
    pip_retry -r "${HYDRAGNN_BASE_REQ}"
fi

# -- Verification ---------------------------------------------------------------
log "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import hydragnn; print('HydraGNN import: OK')"
python -c "import matsim_agents; print('matsim-agents import: OK')"
python -c "import huggingface_hub, transformers, accelerate; print('LLM tooling imports: OK')"

# -- Summary -------------------------------------------------------------------
log "================================================================"
log "Installation complete!"
log ""
log "To activate the environment in a new shell:"
log "  source ${SCRIPT_DIR}/setup_matsim_perlmutter.sh$(if [[ "$USE_GPU" -eq 1 ]]; then echo " --gpu"; fi)"
log ""
log "Direct conda activation path:"
log "  conda activate ${VENV_PATH}"
log ""
log "Then verify:"
log "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
log "  matsim-agents --help"
log "================================================================"
