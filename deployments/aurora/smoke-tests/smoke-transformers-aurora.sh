#!/bin/bash
#PBS -N smoke-transformers-aurora
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# smoke-transformers-aurora.sh
#
# Smoke test: load a HuggingFace model via the matsim-agents `huggingface`
# provider (transformers + Accelerate device_map="auto") across the 12 PVC
# tiles of one Aurora node and emit a one-shot generation. Mirrors the
# Frontier and Perlmutter transformers smoke tests.
#
# Submit:
#   qsub deployments/aurora/smoke-tests/smoke-transformers-aurora.sh
#
# Override the model:
#   qsub -v MATSIM_MODEL_DIR=/path/to/model \
#        deployments/aurora/smoke-tests/smoke-transformers-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}"
[[ -f "${REPO}/pyproject.toml" ]] || REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
MODEL_DIR="${MATSIM_MODEL_DIR:-${PROJ}/models/Qwen2.5-72B-Instruct}"
MODEL_NAME="${MATSIM_MODEL_NAME:-$(basename "${MODEL_DIR}")}"
RUN_DIR="${RUNS_ROOT:-${PROJ}/runs}/smoke-transformers-aurora-${PBS_JOBID:-local}"
[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: model directory not found: ${MODEL_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

# ── modules + venv ──────────────────────────────────────────────────────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# FLAT: every tile is a root device, so Accelerate can spread layers over all 12.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH="${MODEL_DIR}"
ulimit -c 0

SMOKE_PY="${RUN_DIR}/smoke_transformers.py"
cat >"${SMOKE_PY}" <<'PY'
import torch
from langchain_core.messages import HumanMessage

from matsim_agents.backends.llm.provider import get_chat_model

xpu = getattr(torch, "xpu", None)
count = xpu.device_count() if xpu is not None and xpu.is_available() else 0
print(f"  torch {torch.__version__}  xpu_devices={count}")
if count == 0:
    raise SystemExit("No XPU devices visible; run through mpiexec on an Aurora compute node")

print("Building chat model...")
llm = get_chat_model()
print(f"Model ready: {llm}")
response = llm.invoke([HumanMessage(content="What is 2 + 2? Answer in one sentence.")])
print(f"\n=== Response ===\n{response.content}\n================")
print("Smoke test PASSED")
PY

echo "[$(date)] Python: $(command -v python) ($(python --version 2>&1))"
echo "[$(date)] Loading ${MODEL_NAME} (${MODEL_DIR}) via the matsim_agents HuggingFace provider ..."
# Bare Python on Aurora compute nodes cannot see the PVC tiles; use mpiexec.
mpiexec -n 1 --ppn 1 python "${SMOKE_PY}" 2>&1 | tee "${RUN_DIR}/smoke.log"
echo "[$(date)] Done."
