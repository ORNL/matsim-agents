#!/usr/bin/env bash
# Install vLLM in a matsim-owned compatibility environment.
# vLLM hard-pins torch==2.13.0 with CUDA-version-specific compiled kernels,
# while HydraGNN/matsim-agents require torch==2.14.0 -- they cannot share one
# environment. vLLM only ever runs as a standalone `vllm serve` HTTP process;
# matsim-agents talks to it over the OpenAI-compatible /v1 API (via the
# lightweight `openai` client package already present in the other matsim
# environments), so it does not need matsim-agents installed alongside it.
set -Eeuo pipefail

MATSIM_DIR="${MATSIM_DIR:?set MATSIM_DIR to the matsim-agents checkout}"
BASE_VENV="${BASE_VENV:-${MATSIM_DIR}/.venv}"
VLLM_VENV_PATH="${VLLM_VENV_PATH:-${MATSIM_DIR}/venv_vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
RECREATE_VLLM_ENV="${RECREATE_VLLM_ENV:-0}"

log() { printf '\033[1;34m[vllm-compat]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[vllm-compat]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "${BASE_VENV}/bin/python" ]] || die "Base matsim environment not found: ${BASE_VENV}"
if [[ "${RECREATE_VLLM_ENV}" == "1" && -d "${VLLM_VENV_PATH}" ]]; then
    [[ "${VLLM_VENV_PATH}" != "/" && "${VLLM_VENV_PATH}" != "${MATSIM_DIR}" ]] \
        || die "Refusing unsafe vLLM environment target: ${VLLM_VENV_PATH}"
    rm -rf -- "${VLLM_VENV_PATH}"
fi
if [[ ! -x "${VLLM_VENV_PATH}/bin/python" ]]; then
    log "Creating isolated environment at ${VLLM_VENV_PATH}"
    "${BASE_VENV}/bin/python" -m venv "${VLLM_VENV_PATH}"
fi

PYTHON="${VLLM_VENV_PATH}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install "vllm==${VLLM_VERSION}"
"${PYTHON}" -m pip check || log "pip check reported dependency conflicts (see above); continuing since imports are verified next"
"${PYTHON}" - <<'PY'
import torch
import vllm
assert torch.__version__.split("+")[0] == "2.13.0", torch.__version__
print("verified", "vllm", vllm.__version__, "torch", torch.__version__)
PY
log "Complete. Launch servers with: ${VLLM_VENV_PATH}/bin/vllm serve <model> ..."
