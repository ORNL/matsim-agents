#!/usr/bin/env bash
# =============================================================================
# build-mace-venv-perlmutter.sh
#
# Build a dedicated MACE virtualenv on Perlmutter (CUDA/A100) that layers on top
# of the existing HydraGNN venv via `python -m venv --system-site-packages`, so
# it REUSES torch/torchvision/PyG/HydraGNN (no multi-GB re-download) and only
# adds e3nn==0.4.4 + mace-torch into its OWN site-packages, which take
# precedence.
#
# Why e3nn==0.4.4: the MACE foundation-model checkpoints (mace_mp / mace_off)
# were serialized against e3nn 0.4.4 and fail to deserialize under the newer
# e3nn 0.6.x that HydraGNN ships. Shadowing e3nn in the MACE venv fixes this
# without disturbing the HydraGNN base venv.
#
# matsim-agents is installed into the MACE venv (--no-deps) so that
# `mlip.backend: mace` (matsim_agents.active_learning.calculator.build_mace_calculator)
# can be imported and run from this venv.
#
# Usage
# -----
#   bash deployments/perlmutter/setup/build-mace-venv-perlmutter.sh
#
# Configurable variables (override via environment before calling)
# ----------------------------------------------------------------
#   PROJECT_DIR    Root project directory        (default: dir of this script)
#   MATSIM_DIR     matsim-agents checkout path   (default: $PROJECT_DIR)
#   HYDRAGNN_DIR   HydraGNN checkout path        (default: ../HydraGNN)
#   INSTALL_ROOT   Root holding the base venv    (default: HydraGNN-Installation-Perlmutter)
#   BASE_VENV      Existing HydraGNN venv        (default: INSTALL_ROOT/hydragnn_venv)
#   MACE_VENV      Target MACE venv path         (default: INSTALL_ROOT/mace_venv)
#   MACE_TORCH_VERSION  mace-torch pin           (default: 0.3.16)
#   E3NN_VERSION        e3nn pin                 (default: 0.4.4)
# =============================================================================
set -uo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../../..}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"
MACE_TORCH_VERSION="${MACE_TORCH_VERSION:-0.3.16}"
E3NN_VERSION="${E3NN_VERSION:-0.4.4}"

INSTALL_ROOT_DEFAULT="${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter"
INSTALL_ROOT="${INSTALL_ROOT:-${INSTALL_ROOT_DEFAULT}}"
BASE_VENV="${BASE_VENV:-${INSTALL_ROOT}/hydragnn_venv}"
MACE_VENV="${MACE_VENV:-${INSTALL_ROOT}/mace_venv}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[mace]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[mace]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[mace]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "${BASE_VENV}/bin/python" ]] || die "Base HydraGNN venv not found at ${BASE_VENV}. Run install_matsim_perlmutter.sh --gpu first."

# ── Load Perlmutter modules so torch (CUDA) imports for verification ──────────
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
if command -v module >/dev/null 2>&1; then
    if [[ -f /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh ]]; then
        source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh || true
    fi
    module reset || true
    ml nersc-default/1.0 || true
    ml cpe/24.07 || true
    ml PrgEnv-gnu/8.5.0 || true
    ml cudatoolkit || true
else
    warn "module command not found; continuing (build only needs pip)."
fi

# ── Create the MACE venv (layered on the base venv) ───────────────────────────
if [[ -x "${MACE_VENV}/bin/python" ]]; then
    log "MACE venv already exists at ${MACE_VENV}; reusing (will re-run pip installs)."
else
    log "Creating MACE venv (--system-site-packages from ${BASE_VENV})..."
    "${BASE_VENV}/bin/python" -m venv --system-site-packages "${MACE_VENV}"
fi
PIP="${MACE_VENV}/bin/pip"
"${PIP}" install --upgrade pip >/dev/null 2>&1 || true

# ── Protect the inherited core so pip never re-pulls torch/numpy/scipy ────────
# Derive the exact pins from whatever the base venv ships (cu128 torch 2.8.x,
# numpy 1.26.4, etc.) so we never clobber the CUDA wheels.
CONS="$(mktemp)"
"${BASE_VENV}/bin/python" - >"${CONS}" <<'PY'
import importlib
for name in ("torch", "torchvision", "numpy", "scipy"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}=={mod.__version__}")
    except Exception:
        pass
PY
log "Pinning inherited core to protect the base venv:"
sed 's/^/    /' "${CONS}"

# ── Install e3nn 0.4.4 (shadows base 0.6.x) then mace-torch (deps inherited) ──
log "Installing e3nn==${E3NN_VERSION} into MACE venv (shadows base e3nn)..."
"${PIP}" install --no-cache-dir -c "${CONS}" "e3nn==${E3NN_VERSION}" || die "e3nn install failed."

log "Installing mace-torch==${MACE_TORCH_VERSION} (deps satisfied by inherited base)..."
"${PIP}" install --no-cache-dir --no-deps "mace-torch==${MACE_TORCH_VERSION}" || die "mace-torch install failed."

# ── Install matsim-agents into the MACE venv (no-deps: runtime deps inherited) ─
if [[ -d "${MATSIM_DIR}" ]]; then
    log "Installing matsim-agents (--no-deps) into MACE venv so mlip.backend=mace is importable..."
    "${PIP}" install --no-cache-dir --no-deps -e "${MATSIM_DIR}" || \
        warn "matsim-agents editable install failed; the mace backend may not import from this venv."
else
    warn "matsim-agents checkout not found at ${MATSIM_DIR}; skipping editable install."
fi

# ── Verify ────────────────────────────────────────────────────────────────────
log "Verifying MACE venv..."
"${MACE_VENV}/bin/python" - <<'PY'
import torch, e3nn, mace
from mace.calculators import mace_mp, mace_off, MACECalculator  # noqa: F401
print("torch  ", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("e3nn   ", e3nn.__version__)
print("mace   ", mace.__version__)
try:
    from matsim_agents.active_learning.calculator import build_mace_calculator  # noqa: F401
    print("matsim-agents mace backend import: OK")
except Exception as exc:  # noqa: BLE001
    print("matsim-agents import WARNING:", exc)
PY

log "Done. To use MACE: source ${MACE_VENV}/bin/activate"
log "Then set mlip.backend: mace (mlip.mace.family: mace_mp|mace_off, mlip.mace.model: small|medium|large)."
