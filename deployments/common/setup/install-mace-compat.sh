#!/usr/bin/env bash
# Install MACE in a matsim-owned compatibility environment.
#
# MACE 0.3.16 declares e3nn==0.4.4, while the HydraGNN environment declares
# e3nn==0.5.1. A single Python environment cannot satisfy both contracts.
set -Eeuo pipefail

MATSIM_DIR="${MATSIM_DIR:?set MATSIM_DIR to the matsim-agents checkout}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
MACE_VENV_PATH="${MACE_VENV_PATH:-${MATSIM_DIR}/.venv-mace}"
MACE_TORCH_VERSION="${MACE_TORCH_VERSION:-0.3.16}"
E3NN_VERSION="${E3NN_VERSION:-0.4.4}"
FACILITY="${FACILITY:-unknown}"
RECREATE_MACE_ENV="${RECREATE_MACE_ENV:-0}"

log() { printf '\033[1;34m[mace-compat]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[mace-compat]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "${BASE_VENV}/bin/python" ]] || die "Base matsim environment not found: ${BASE_VENV}"
if [[ "${RECREATE_MACE_ENV}" == "1" && -d "${MACE_VENV_PATH}" ]]; then
    [[ "${MACE_VENV_PATH}" != "/" && "${MACE_VENV_PATH}" != "${MATSIM_DIR}" ]] \
        || die "Refusing unsafe MACE environment target: ${MACE_VENV_PATH}"
    rm -rf -- "${MACE_VENV_PATH}"
fi
if [[ ! -x "${MACE_VENV_PATH}/bin/python" ]]; then
    log "Creating ${MACE_VENV_PATH} from ${BASE_VENV}"
    if [[ "${FACILITY}" == "aurora" ]]; then
        "${BASE_VENV}/bin/python" -m venv --system-site-packages "${MACE_VENV_PATH}"
    else
        "${BASE_VENV}/bin/python" -m venv "${MACE_VENV_PATH}"
    fi
fi

PYTHON="${MACE_VENV_PATH}/bin/python"
# venv's --system-site-packages exposes the interpreter's global packages, not
# another venv. Add the matsim environment explicitly; packages installed in
# .venv-mace appear earlier on sys.path and can safely shadow e3nn only.
BASE_SITE="$("${BASE_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
MACE_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "${BASE_SITE}" >"${MACE_SITE}/matsim-base-venv.pth"
CONSTRAINTS="$(mktemp)"
trap 'rm -f -- "${CONSTRAINTS}"' EXIT
"${BASE_VENV}/bin/python" - >"${CONSTRAINTS}" <<'PY'
from importlib.metadata import version
for name in ("torch", "torchvision", "numpy", "scipy"):
    print(f"{name}=={version(name)}")
PY

"${PYTHON}" -m pip install --upgrade pip
if [[ "${FACILITY}" == "aurora" ]]; then
    # The frameworks module can expose an h5py linked to an unavailable HDF5.
    "${PYTHON}" -m pip install --force-reinstall --no-deps h5py
fi
"${PYTHON}" -m pip install -c "${CONSTRAINTS}" "e3nn==${E3NN_VERSION}"
"${PYTHON}" -m pip install -c "${CONSTRAINTS}" "mace-torch==${MACE_TORCH_VERSION}"
if [[ "${FACILITY}" == "aurora" ]]; then
    log "Guarding the optional CUDA/HIP-only OpenEquivariance import in MACE"
    "${PYTHON}" "${MATSIM_DIR}/deployments/common/setup/guard_mace_optional_backends.py" \
        --site "${MACE_SITE}"
fi
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "hatchling>=1.21"
# --no-build-isolation: avoids pip's PEP 517 isolated build env, which needs
# network/TLS to fetch hatchling; it is pre-installed above instead.
"${PYTHON}" -m pip install --no-build-isolation --no-deps --force-reinstall "${MATSIM_DIR}"

# PyTorch >=2.6 otherwise changes torch.load's default for e3nn's constants.
ACTIVATE="${MACE_VENV_PATH}/bin/activate"
MARKER="# >>> matsim-mace-compat >>>"
if ! grep -qF "${MARKER}" "${ACTIVATE}"; then
    {
        printf '\n%s\n' "${MARKER}"
        printf '%s\n' 'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1'
        printf '%s\n' '# <<< matsim-mace-compat <<<'
    } >>"${ACTIVATE}"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ "${FACILITY}" == "aurora" ]]; then
    "${PYTHON}" "${MATSIM_DIR}/deployments/common/setup/check-runtime-dependencies.py" \
        matsim-agents mace-torch
    EXPECTED_TORCH="$("${BASE_VENV}/bin/python" -c 'import torch; print(torch.__file__)')"
    "${PYTHON}" - "${EXPECTED_TORCH}" <<'PY'
import pathlib
import sys
import torch
if pathlib.Path(torch.__file__).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("MACE replaced Aurora framework Torch; rerun with RECREATE_MACE_ENV=1")
PY
fi
"${PYTHON}" - <<'PY'
import dscribe
import e3nn
import mace
import torch
from mace.calculators import (  # noqa: F401
    MACECalculator,
    mace_anicc,
    mace_mp,
    mace_off,
    mace_omol,
    mace_polar,
)
from matsim_agents.backends.mlip.calculator import build_mace_calculator  # noqa: F401

assert e3nn.__version__ == "0.4.4", e3nn.__version__
print("verified", "torch", torch.__version__, "e3nn", e3nn.__version__, "mace", mace.__version__,
    "dscribe", getattr(dscribe, "__version__", "unknown"))
PY
log "Complete. Activate MACE workflows with: source ${MACE_VENV_PATH}/bin/activate"
