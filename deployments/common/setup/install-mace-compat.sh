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
    "${BASE_VENV}/bin/python" -m venv "${MACE_VENV_PATH}"
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
import importlib
for name in ("torch", "torchvision", "numpy", "scipy"):
    try:
        module = importlib.import_module(name)
        print(f"{name}=={module.__version__}")
    except Exception:
        pass
PY

"${PYTHON}" -m pip install --upgrade pip
if [[ "${FACILITY}" == "aurora" ]]; then
    # The frameworks module can expose an h5py linked to an unavailable HDF5.
    "${PYTHON}" -m pip install --force-reinstall --no-deps h5py
fi
"${PYTHON}" -m pip install -c "${CONSTRAINTS}" "e3nn==${E3NN_VERSION}"
"${PYTHON}" -m pip install -c "${CONSTRAINTS}" "mace-torch==${MACE_TORCH_VERSION}"
"${PYTHON}" -m pip install --no-deps --force-reinstall "${MATSIM_DIR}"

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

"${PYTHON}" - <<'PY'
import e3nn
import mace
import torch
from mace.calculators import MACECalculator, mace_mp, mace_off  # noqa: F401
from matsim_agents.active_learning.calculator import build_mace_calculator  # noqa: F401

assert e3nn.__version__ == "0.4.4", e3nn.__version__
print("verified", "torch", torch.__version__, "e3nn", e3nn.__version__, "mace", mace.__version__)
PY
log "Complete. Activate MACE workflows with: source ${MACE_VENV_PATH}/bin/activate"
