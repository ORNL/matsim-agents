#!/usr/bin/env bash
# =============================================================================
# install_matsim_aurora.sh
#
# Frontier/Perlmutter-style phased install for matsim-agents + HydraGNN on
# Aurora.
#
# Phase 1: Run HydraGNN's Aurora installer to create/configure the venv and
#          install HydraGNN dependencies from installation_DOE_supercomputers.
# Phase 2: Activate that environment and install matsim-agents dependencies.
#
# Usage:
#   bash install_matsim_aurora.sh
#
# Optional overrides:
#   PROJECT_DIR       Root project directory
#   MATSIM_DIR        matsim-agents checkout path
#   HYDRAGNN_DIR      HydraGNN checkout path
#   HYDRAGNN_INSTALLER  Aurora installer script path
#   INSTALL_ROOT      Base install directory for HydraGNN Aurora installer
#   VENV_PATH         Environment path created by HydraGNN installer
#   RECREATE_ENV      Passed to HydraGNN installer (default: 0)
#   LLM_BACKENDS      matsim-agents extras (default: dev)
#   INSTALL_VLLM_SERVER  Install vllm package (default: 0)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../../..}"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"
HYDRAGNN_INSTALLER="${HYDRAGNN_INSTALLER:-${HYDRAGNN_DIR}/installation_DOE_supercomputers/hydragnn_installation_bash_script_aurora.sh}"
INSTALL_ROOT="${INSTALL_ROOT:-${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Aurora}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
RECREATE_ENV="${RECREATE_ENV:-0}"
LLM_BACKENDS="${LLM_BACKENDS:-dev}"
INSTALL_VLLM_SERVER="${INSTALL_VLLM_SERVER:-0}"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

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
[[ -f "${HYDRAGNN_INSTALLER}" ]] || die "HydraGNN Aurora installer not found: ${HYDRAGNN_INSTALLER}"

# -- Phase 1: Delegate to HydraGNN Aurora installer ----------------------------
log "Phase 1: running HydraGNN Aurora installer"
log "Installer: ${HYDRAGNN_INSTALLER}"
log "INSTALL_ROOT=${INSTALL_ROOT}"
log "VENV_PATH=${VENV_PATH}"

(
    cd "$(dirname "${HYDRAGNN_INSTALLER}")"
    INSTALL_ROOT="${INSTALL_ROOT}" \
    VENV_PATH="${VENV_PATH}" \
    RECREATE_ENV="${RECREATE_ENV}" \
    bash "$(basename "${HYDRAGNN_INSTALLER}")"
)

[[ -d "${VENV_PATH}" ]] || die "Expected Aurora environment not found at ${VENV_PATH}"

log "Activating environment: ${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

PY_MM="$(python - <<'PYEOF'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PYEOF
)"
PY_MAJOR="${PY_MM%%.*}"
PY_MINOR="${PY_MM##*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
    die "Python >= 3.10 required by matsim-agents, found ${PY_MM} in ${VENV_PATH}."
fi

log "Upgrading pip/setuptools/wheel in active environment"
pip install -U pip setuptools wheel

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
# NOTE: cap huggingface_hub below 1.0. The frameworks module on Aurora ships
# transformers 4.57.x which requires huggingface-hub<1.0; allowing the 1.x line
# (e.g. 1.15.0) breaks `import transformers`. Keep transformers<5 for the same
# compatibility window across Perlmutter/Frontier installers.
pip_retry "huggingface_hub>=0.34.0,<1.0" "transformers>=4.45,<5.0" "accelerate>=1.13"

# pyXtal (optional, used by matsim_agents.discovery.seeds for random-symmetry
# crystal generation when no AFLOW prototype matches the target composition).
# Installed with --upgrade-strategy only-if-needed so it cannot upgrade the
# frameworks-provided numpy/scipy/pymatgen/spglib/ase pins. Made non-fatal:
# pyxtal's transitive deps (notably pyshtools) sometimes fail to build on
# HPC systems without a working Fortran toolchain; the discovery code already
# warns and skips random search cleanly when pyxtal is missing.
log "Installing pyxtal (optional, for random-symmetry seed generation)"
pip_retry "pyxtal>=0.6" || warn "pyxtal install failed; random-symmetry seed generation will be unavailable."

if [[ "${INSTALL_VLLM_SERVER}" == "1" ]]; then
    log "INSTALL_VLLM_SERVER=1 -> installing vllm"
    pip_retry vllm
fi

# Strip any overlay torch / CUDA wheels that transitive deps (notably
# fairchem-core) may have dragged into the venv. Aurora is XPU-only and the
# frameworks module already ships torch 2.10/XPU + torchvision + triton-xpu
# via --system-site-packages. A venv-local CUDA torch (e.g. torch==2.8.0+cu128
# plus nvidia-*-cu12 + triton 3.4.0) silently shadows the frameworks build and
# breaks vLLM-XPU at runtime with a torchvision._meta_registrations import
# error. Uninstall is a no-op if the packages aren't present.
log "Removing any CUDA torch overlay that transitive deps may have pulled in"
pip uninstall -y \
    torch torchvision torchaudio triton \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# Strip any overlay numpy / pandas that transitive deps may have pulled into
# the venv.
#
# NOTE ON PROJECT-WIDE NUMPY PIN POLICY
# The Frontier (ROCm) and Perlmutter (CUDA) installers pin numpy==1.26.4
# because PyTorch on those systems was compiled against the numpy 1.x ABI.
# Aurora is DIFFERENT: the `frameworks/2025.3.1` module ships numpy 2.2.6 and
# every C extension in the stack is compiled against the numpy 2.x ABI:
#   pyarrow 23.0, scipy 1.17, scikit-learn 1.8, torch 2.10/XPU, IPEX 2.10.
# Pinning to numpy==1.26.4 on Aurora would cause the SAME class of ABI-
# mismatch crash that the pin is meant to prevent on other machines.
# The Aurora-correct equivalent of that pin is therefore NOT to install any
# venv-local numpy at all: `pip uninstall -y numpy` lets --system-site-packages
# expose the pre-built frameworks 2.2.6, which all C extensions expect.
#
# A venv-local numpy 2.4.x / pandas 2.3.x overlay (e.g. dragged in by
# fairchem-core or other transitive deps) silently shadows the frameworks build
# and triggers a SIGSEGV inside pyarrow.lib during the
# `import vllm.model_executor.models` chain (which transitively pulls in
# sklearn -> pandas.compat.pyarrow -> pyarrow.lib). Uninstall is a no-op when
# nothing is overlaid; let --system-site-packages expose the frameworks copy.
log "Removing any numpy/pandas overlay (Aurora frameworks ships 2.2.6 / 3.0.0)"
pip uninstall -y numpy pandas 2>/dev/null || true

log "Verifying installation imports"
python -c "import hydragnn; print('HydraGNN import: OK')"
python -c "import matsim_agents; print('matsim-agents import: OK')"
python -c "import huggingface_hub, transformers, accelerate; print('LLM tooling imports: OK')"

log "================================================================"
log "Installation complete"
log "Activate later with:"
log "  source ${VENV_PATH}/bin/activate"
log "================================================================"
