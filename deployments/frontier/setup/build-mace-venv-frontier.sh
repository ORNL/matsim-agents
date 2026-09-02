#!/usr/bin/env bash
# =============================================================================
# build-mace-venv-frontier.sh
#
# Build a dedicated MACE virtualenv on Frontier (ROCm) that layers on top of the
# existing HydraGNN ROCm venv via `python -m venv --system-site-packages`, so it
# REUSES torch/torchvision/PyG/HydraGNN (no multi-GB re-download) and only adds
# e3nn==0.4.4 + mace-torch into its OWN site-packages, which take precedence.
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
#   bash deployments/frontier/setup/build-mace-venv-frontier.sh [--rocm72]
#
# Configurable variables (override via environment before calling)
# ----------------------------------------------------------------
#   PROJECT_DIR    Root project directory        (default: dir of this script)
#   MATSIM_DIR     matsim-agents checkout path   (default: $PROJECT_DIR/matsim-agents)
#   HYDRAGNN_DIR   HydraGNN checkout path        (default: $PROJECT_DIR/HydraGNN)
#   ROCM_VERSION   "7.1" (default) or "7.2"      (or pass --rocm72)
#   INSTALL_ROOT   Root holding the base venv    (default: HydraGNN-Installation-Frontier[-ROCm72])
#   BASE_VENV      Existing HydraGNN venv        (default: INSTALL_ROOT/hydragnn_venv[_rocm72])
#   MACE_VENV      Target MACE venv path         (default: INSTALL_ROOT/mace_venv[_rocm72])
#   MACE_TORCH_VERSION  mace-torch pin           (default: 0.3.16)
#   E3NN_VERSION        e3nn pin                 (default: 0.4.4)
# =============================================================================
set -uo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/../../..}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
MATSIM_DIR="${MATSIM_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-$(cd "${MATSIM_DIR}/.." && pwd)/HydraGNN}"
ROCM_VERSION="${ROCM_VERSION:-7.1}"
MACE_TORCH_VERSION="${MACE_TORCH_VERSION:-0.3.16}"
E3NN_VERSION="${E3NN_VERSION:-0.4.4}"

for arg in "$@"; do
    [[ "$arg" == "--rocm72" ]] && ROCM_VERSION="7.2"
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[mace]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[mace]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[mace]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Resolve base venv / install root for the selected ROCm version ────────────
INSTALL_ROOT_DEFAULT_71="${MATSIM_DIR}/.hpc-build/frontier"
INSTALL_ROOT_DEFAULT_72="${MATSIM_DIR}/.hpc-build/frontier"
if [[ "$ROCM_VERSION" == "7.2" ]]; then
    INSTALL_ROOT="${INSTALL_ROOT:-$INSTALL_ROOT_DEFAULT_72}"
    BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
    MACE_VENV="${MACE_VENV:-${INSTALL_ROOT}/mace_venv_rocm72}"
else
    INSTALL_ROOT="${INSTALL_ROOT:-$INSTALL_ROOT_DEFAULT_71}"
    BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
    MACE_VENV="${MACE_VENV:-${INSTALL_ROOT}/mace_venv}"
fi

[[ -x "${BASE_VENV}/bin/python" ]] || die "Base matsim venv not found at ${BASE_VENV}. Run install.sh first."

# ── Load Frontier ROCm modules so torch (HIP) imports for verification ────────
if [[ -f "${SCRIPT_DIR}/frontier-module-stack.sh" ]]; then
    log "Sourcing frontier-module-stack.sh (ROCm ${ROCM_VERSION})..."
    # shellcheck disable=SC1091
    ROCM_VERSION="${ROCM_VERSION}" source "${SCRIPT_DIR}/frontier-module-stack.sh" || \
        warn "frontier-module-stack.sh returned non-zero; continuing (build only needs pip)."
fi

# ── Network: correct CA bundle for the SUSE login node (pip default is wrong) ─
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/ca-bundle.pem}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-/etc/ssl/ca-bundle.pem}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/ca-bundle.pem}"
unset PIP_CONSTRAINT 2>/dev/null || true

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
# Derive the exact pins from whatever the base venv already ships (robust across
# ROCm 7.1 / 7.2 without hardcoding wheel tags).
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
print("torch  ", torch.__version__, "HIP/CUDA available:", torch.cuda.is_available())
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
