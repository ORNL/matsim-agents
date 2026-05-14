#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-transformers-frontier.sh
#
# Smoke test: load Qwen2.5-72B-Instruct via HuggingFace Transformers +
# Accelerate (device_map="auto") across all 8 GCDs on a single Frontier node
# and generate a short response. No vLLM required.
#
# Submit:
#   sbatch scripts/smoke-transformers-frontier.sh
#
# Or run interactively after `salloc`:
#   bash scripts/smoke-transformers-frontier.sh
# ---------------------------------------------------------------------------
#SBATCH -A mat746
#SBATCH -J smoke-transformers
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/smoke-transformers-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/smoke-transformers-%j/job-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug

set -uo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename $MODEL_DIR)}
RUN_DIR=$PROJ/runs/smoke-transformers-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── conda + modules ──────────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# MIOpen cache in /tmp to avoid Lustre locking issues
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID:-local}
mkdir -p "$MIOPEN_USER_DB_PATH"

# LD_PRELOAD of libamdhip64.so causes hipErrorInvalidKernelFile on Frontier
# Use VLLM_CUDART_SO_PATH instead so HIP runtime is discoverable without LD_PRELOAD.
export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Torch:"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

echo "[$(date)] Loading $(basename $MODEL_DIR) via matsim_agents HuggingFace provider ..."
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
