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
#   INSTALL_ROOT      Root holding env + all build deps (default: HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter)
#   VENV_PATH         Target conda env path            (default: INSTALL_ROOT/hydragnn_venv)
#   PYTHON_VERSION    Python version for env creation  (default: 3.11)
#   EXPECTED_CUDA_MM  CUDA major.minor                 (default: 12.9)
#   TORCH_CUDA_TAG    PyTorch wheel tag                (default: cu128)
#   PYG_WHL_URL       PyG wheel index URL              (default: torch-2.8.0+cu128)
#   TORCH_CUDA_ARCH   GPU arch list                    (default: 8.0)
#   MAX_JOBS          Build parallelism                (default: 16)
#   LLM_BACKENDS      matsim extras                    (default: dev)
#   INSTALL_LLM_EXTRAS Install huggingface/transformers extras (default: 0)
#   INSTALL_VLLM_SERVER  Install vLLM package          (default: 0)
#   INSTALL_UMA        Install fairchem-core (UMA MLIP backend) (default: 0)
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

# INSTALL_ROOT is the single source of truth for where ALL build artifacts and
# the conda env live. The HydraGNN sub-installer drops its dependency build
# trees (ADIOS2, MPI4PY, DDStore, GPTL, DeepHyper, PyG) directly under
# INSTALL_ROOT, and the venv nests inside it by default. Deriving VENV_PATH from
# INSTALL_ROOT (rather than the reverse) means overriding only VENV_PATH can
# never scatter the build dirs into an unrelated parent directory.
DEFAULT_INSTALL_ROOT="${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"
INSTALL_ROOT="${INSTALL_ROOT:-${DEFAULT_INSTALL_ROOT}}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
EXPECTED_CUDA_MM="${EXPECTED_CUDA_MM:-12.9}"
# Pin to HydraGNN's torch==2.11.0 wheels (published as cu129). cu129 binaries
# match the loaded cudatoolkit/12.9 module exactly.
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-2.8.0+cu128.html}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.0}"
MAX_JOBS="${MAX_JOBS:-16}"
LLM_BACKENDS="${LLM_BACKENDS:-dev}"
INSTALL_LLM_EXTRAS="${INSTALL_LLM_EXTRAS:-0}"
INSTALL_VLLM_SERVER="${INSTALL_VLLM_SERVER:-0}"
INSTALL_UMA="${INSTALL_UMA:-0}"

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

# INSTALL_ROOT is resolved at the top of this script (build deps + env live
# here). Do NOT re-derive it from VENV_PATH, or overriding VENV_PATH alone would
# scatter the dependency build trees into VENV_PATH's parent directory.
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

if [[ "${INSTALL_LLM_EXTRAS}" == "1" ]]; then
    log "INSTALL_LLM_EXTRAS=1 -> installing LLM tooling extras (huggingface_hub + hf_transfer + transformers + accelerate)..."
    # Cap huggingface_hub<1.0: transformers<4.58 (and thus our 4.45..4.57 floor)
    # requires huggingface-hub<1.0. Pinned here to keep the resolver from picking
    # up the breaking 1.x line (e.g. 1.15.0) that ships with newer fairchem-core.
    # hf_transfer accelerates large model downloads when the package is available.
    pip_retry "huggingface_hub>=0.34.0,<1.0" "hf_transfer" "transformers>=4.45,<5.0" "accelerate>=1.13"
else
    log "INSTALL_LLM_EXTRAS=0 -> skipping optional LLM tooling extras"
fi

# pyXtal (random-symmetry seed generation for matsim_agents.discovery.seeds)
# is now a CORE matsim-agents dependency (see pyproject.toml) and is therefore
# installed with the editable matsim-agents package above. Modern pyxtal
# (>=1.0) is a pure-Python wheel with no Fortran/pyshtools build step, so no
# separate/gated install step is needed here.

if [[ "${INSTALL_VLLM_SERVER}" == "1" ]]; then
    log "INSTALL_VLLM_SERVER=1 -> installing vLLM server package"
    pip_retry vllm
fi

# fairchem-core (optional, required for the UMA MLIP backend).
# IMPORTANT: fairchem-core>=2.0 requires numpy>=2.0 + scipy>=1.15, which
# conflicts with HydraGNN's strict numpy==1.26.4 / scipy==1.14.1 pins.
# It MUST be installed in a separate venv to avoid breaking the HydraGNN env.
# INSTALL_UMA=1 creates ${INSTALL_ROOT}/fairchem_venv alongside hydragnn_venv,
# with matsim-agents installed in editable mode there too (so the UMA code path
# can reach matsim_agents.active_learning.calculator.build_uma_calculator).
UMA_VENV_PATH="${INSTALL_ROOT}/fairchem_venv"
if [[ "${INSTALL_UMA}" == "1" ]]; then
    log "INSTALL_UMA=1 -> creating separate fairchem_venv at ${UMA_VENV_PATH}..."
    # Use the HydraGNN venv's Python (3.11) — not the system Python — so the
    # new venv inherits a supported Python version for fairchem-core>=2.0.
    "${VENV_PATH}/bin/python" -m venv "${UMA_VENV_PATH}"
    "${UMA_VENV_PATH}/bin/pip" install -U pip setuptools wheel
    "${UMA_VENV_PATH}/bin/pip" install --upgrade-strategy only-if-needed \
        fairchem-core \
        || warn "fairchem-core install failed; UMA backend will be unavailable."
    # Install matsim-agents with its runtime deps (langchain-core, langgraph,
    # pydantic, ase, etc.) but without the uma extra (fairchem-core is already
    # installed above). --upgrade-strategy only-if-needed avoids overwriting
    # fairchem-core's numpy/scipy with older versions.
    "${UMA_VENV_PATH}/bin/pip" install --upgrade-strategy only-if-needed \
        -e "${MATSIM_DIR}[dev]"
    log "fairchem_venv created. To use UMA: activate ${UMA_VENV_PATH}"
else
    log "INSTALL_UMA=0 -> skipping fairchem_venv creation"
fi

# -- Re-assert HydraGNN-pinned versions ----------------------------------------
# matsim-agents and its extras can drift HydraGNN pins via transitive deps.
# Re-assert strict torch/PyG/base pin files so every scripted rebuild converges
# to HydraGNN's version contract.
HYDRAGNN_BASE_REQ="${HYDRAGNN_DIR}/requirements-base.txt"
HYDRAGNN_TORCH_REQ="${HYDRAGNN_DIR}/requirements-torch.txt"
HYDRAGNN_PYG_REQ="${HYDRAGNN_DIR}/requirements-pyg.txt"

if [[ -f "${HYDRAGNN_TORCH_REQ}" ]]; then
    log "Re-asserting HydraGNN torch pins from ${HYDRAGNN_TORCH_REQ}"
    # torch/vision/audio wheels come from the PyTorch CUDA index; non-torch deps
    # in the same file (e.g., e3nn/torchmetrics) resolve from PyPI.
    pip_retry --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
        --extra-index-url https://pypi.org/simple \
        -r "${HYDRAGNN_TORCH_REQ}"
fi

if [[ -f "${HYDRAGNN_PYG_REQ}" ]]; then
    log "Re-asserting HydraGNN PyG pins from ${HYDRAGNN_PYG_REQ}"
    # Prefer prebuilt ABI-matched wheels for the selected torch/CUDA combo.
    pip_retry --find-links "${PYG_WHL_URL}" -r "${HYDRAGNN_PYG_REQ}"
fi

if [[ -f "${HYDRAGNN_BASE_REQ}" ]]; then
    log "Re-asserting HydraGNN base pins from ${HYDRAGNN_BASE_REQ}"
    pip_retry -r "${HYDRAGNN_BASE_REQ}"
fi

log "Running pip dependency consistency check..."
pip check

# -- Verification ---------------------------------------------------------------
log "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import hydragnn; print('HydraGNN import: OK')"
python -c "import matsim_agents; print('matsim-agents import: OK')"

if [[ "${INSTALL_LLM_EXTRAS}" == "1" ]]; then
    python -c "import huggingface_hub, transformers, accelerate; print('LLM tooling imports: OK')"
else
    log "Skipping LLM tooling import verification (INSTALL_LLM_EXTRAS=0)"
fi
if [[ "${INSTALL_UMA}" == "1" ]]; then
    "${UMA_VENV_PATH}/bin/python" -c "from fairchem.core import FAIRChemCalculator; print('fairchem-core import: OK (fairchem_venv)')"
else
    log "Skipping fairchem-core import verification (INSTALL_UMA=0)"
fi

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
