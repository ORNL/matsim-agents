#!/usr/bin/env bash
# Install FairChem/UMA in a matsim-owned compatibility environment.
# HydraGNN requires torch==2.14.0 while FairChem 2.22 requires
# torch>=2.13.0,<2.14.dev0, so they cannot share one environment.
set -Eeuo pipefail

MATSIM_DIR="${MATSIM_DIR:?set MATSIM_DIR to the matsim-agents checkout}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"
UMA_MATSIM_EXTRAS="${UMA_MATSIM_EXTRAS:-uma,dev,openai,ollama,anthropic,huggingface}"
RECREATE_UMA_ENV="${RECREATE_UMA_ENV:-0}"

log() { printf '\033[1;34m[uma-compat]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[uma-compat]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "${BASE_VENV}/bin/python" ]] || die "Base matsim environment not found: ${BASE_VENV}"
if [[ "${RECREATE_UMA_ENV}" == "1" && -d "${UMA_VENV_PATH}" ]]; then
    [[ "${UMA_VENV_PATH}" != "/" && "${UMA_VENV_PATH}" != "${MATSIM_DIR}" ]] \
        || die "Refusing unsafe UMA environment target: ${UMA_VENV_PATH}"
    rm -rf -- "${UMA_VENV_PATH}"
fi
if [[ ! -x "${UMA_VENV_PATH}/bin/python" ]]; then
    log "Creating isolated environment at ${UMA_VENV_PATH}"
    "${BASE_VENV}/bin/python" -m venv "${UMA_VENV_PATH}"
fi

PYTHON="${UMA_VENV_PATH}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "hatchling>=1.21"
# --no-build-isolation: avoids pip's PEP 517 isolated build env, which needs
# network/TLS to fetch hatchling; it is pre-installed above instead.
"${PYTHON}" -m pip install --no-build-isolation --upgrade-strategy only-if-needed \
    "${MATSIM_DIR}[${UMA_MATSIM_EXTRAS}]"
"${PYTHON}" -m pip check
"${PYTHON}" - <<'PY'
import fairchem
import numpy
import scipy
import torch
from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: F401
import matsim_agents  # noqa: F401

assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 14)
print("verified", "torch", torch.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "fairchem", getattr(fairchem, "__version__", "unknown"))
PY
log "Complete. Activate UMA workflows with: source ${UMA_VENV_PATH}/bin/activate"
