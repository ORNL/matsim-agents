#!/usr/bin/env bash
# Install FairChem/UMA in a matsim-owned compatibility environment.
# HydraGNN and FairChem require incompatible e3nn versions. Aurora's Torch
# prerelease additionally requires an explicit experimental FairChem wheel.
set -Eeuo pipefail

MATSIM_DIR="${MATSIM_DIR:?set MATSIM_DIR to the matsim-agents checkout}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
UMA_VENV_PATH="${UMA_VENV_PATH:-${MATSIM_DIR}/.venv-uma}"
UMA_MATSIM_EXTRAS="${UMA_MATSIM_EXTRAS:-uma,dev,openai,ollama,anthropic,huggingface}"
RECREATE_UMA_ENV="${RECREATE_UMA_ENV:-0}"
FACILITY="${FACILITY:-unknown}"
AURORA_FAIRCHEM_VERSION="${AURORA_FAIRCHEM_VERSION:-2.23.0}"

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
    if [[ "${FACILITY}" == "aurora" ]]; then
        "${BASE_VENV}/bin/python" -m venv --system-site-packages "${UMA_VENV_PATH}"
    else
        "${BASE_VENV}/bin/python" -m venv "${UMA_VENV_PATH}"
    fi
fi

PYTHON="${UMA_VENV_PATH}/bin/python"
if [[ "${FACILITY}" == "aurora" ]]; then
    BASE_SITE="$("${BASE_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    UMA_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    printf '%s\n' "${BASE_SITE}" >"${UMA_SITE}/matsim-base-venv.pth"
    CONSTRAINTS="$(mktemp)"
    trap 'rm -f -- "${CONSTRAINTS}"' EXIT
    "${BASE_VENV}/bin/python" - >"${CONSTRAINTS}" <<'PY'
from importlib.metadata import version
for name in ("torch", "torchvision", "numpy", "scipy"):
    print(f"{name}=={version(name)}")
PY
    export PIP_CONSTRAINT="${CONSTRAINTS}"
    "${PYTHON}" -m pip install --force-reinstall --no-deps h5py
fi
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install --upgrade-strategy only-if-needed "hatchling>=1.21"
FAIRCHEM_ARGS=()
if [[ "${FACILITY}" == "aurora" ]]; then
    WHEEL_ROOT="${MATSIM_DIR}/.hpc-build/aurora/fairchem/${AURORA_FAIRCHEM_VERSION}"
    mkdir -p "${WHEEL_ROOT}/upstream"
    "${PYTHON}" -m pip download --no-deps --only-binary=:all: \
        --dest "${WHEEL_ROOT}/upstream" "fairchem-core==${AURORA_FAIRCHEM_VERSION}"
    UPSTREAM_WHEELS=("${WHEEL_ROOT}/upstream"/fairchem_core-"${AURORA_FAIRCHEM_VERSION}"-*.whl)
    [[ "${#UPSTREAM_WHEELS[@]}" == "1" && -f "${UPSTREAM_WHEELS[0]}" ]] \
        || die "Expected exactly one upstream FairChem wheel"
    TORCH_VERSION="$("${BASE_VENV}/bin/python" -c 'from importlib.metadata import version; print(version("torch"))')"
    COMPAT_WHEEL="$("${PYTHON}" "${MATSIM_DIR}/deployments/aurora/setup/prepare_fairchem_wheel.py" \
        --wheel "${UPSTREAM_WHEELS[0]}" --output "${WHEEL_ROOT}/patched" \
        --torch-version "${TORCH_VERSION}")"
    log "Experimental FairChem override: ${COMPAT_WHEEL}; provenance: ${COMPAT_WHEEL%.whl}.json"
    FAIRCHEM_ARGS=("${COMPAT_WHEEL}")
fi
# --no-build-isolation: avoids pip's PEP 517 isolated build env, which needs
# network/TLS to fetch hatchling; it is pre-installed above instead.
"${PYTHON}" -m pip install --no-build-isolation --upgrade-strategy only-if-needed \
    "${MATSIM_DIR}[${UMA_MATSIM_EXTRAS}]" "${FAIRCHEM_ARGS[@]}"
if [[ "${FACILITY}" == "aurora" ]]; then
    "${PYTHON}" -m pip check || log "Full pip check reported inherited conflicts; checking UMA runtime next"
    "${PYTHON}" "${MATSIM_DIR}/deployments/common/setup/check-runtime-dependencies.py" \
        "matsim-agents[${UMA_MATSIM_EXTRAS}]"
    EXPECTED_TORCH="$("${BASE_VENV}/bin/python" -c 'import torch; print(torch.__file__)')"
    "${PYTHON}" - "${EXPECTED_TORCH}" <<'PY'
import pathlib
import sys
import torch
if pathlib.Path(torch.__file__).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("UMA replaced Aurora framework Torch; rerun with RECREATE_UMA_ENV=1")
PY
else
    "${PYTHON}" -m pip check
fi
"${PYTHON}" - <<'PY'
import dscribe
import fairchem
import numpy
import scipy
import torch
from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: F401
import matsim_agents  # noqa: F401

assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 14)
print("verified", "torch", torch.__version__, "numpy", numpy.__version__,
    "scipy", scipy.__version__, "fairchem", getattr(fairchem, "__version__", "unknown"),
    "dscribe", getattr(dscribe, "__version__", "unknown"))
PY
log "Complete. Activate UMA workflows with: source ${UMA_VENV_PATH}/bin/activate"
