#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-transformers-perlmutter.sh
#
# Smoke test: load a HuggingFace model via the matsim-agents `huggingface`
# provider (transformers + Accelerate device_map="auto") across the 4 A100s
# on a single Perlmutter GPU node, and emit a one-shot generation.
#
# Submit:
#   sbatch scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh
#
# Override the model:
#   sbatch --export=ALL,MATSIM_MODEL_DIR=/path/to/model \
#          scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh
# ---------------------------------------------------------------------------
#SBATCH -A amsc001
#SBATCH -J smoke-transformers-pm
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/smoke-transformers-pm-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/smoke-transformers-pm-%j/job-%j.out

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
RUN_DIR=$PROJ/runs/smoke-transformers-pm-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── modules + venv ───────────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Model:  $MODEL_NAME ($MODEL_DIR)"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

echo "[$(date)] Loading $(basename "$MODEL_DIR") via matsim_agents HuggingFace provider ..."
python - <<PY
import os
os.environ["MATSIM_LLM_PROVIDER"] = "huggingface"
os.environ["MATSIM_HF_MODEL_PATH"] = "$MODEL_DIR"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from matsim_agents.llm import get_chat_model
from langchain_core.messages import HumanMessage

print("Building chat model...")
llm = get_chat_model()
print(f"Model ready: {llm}")

print("Invoking...")
response = llm.invoke([HumanMessage(content="What is 2 + 2? Answer in one sentence.")])
print(f"\n=== Response ===\n{response.content}\n================")
print("Smoke test PASSED")
PY

echo "[$(date)] Done."
