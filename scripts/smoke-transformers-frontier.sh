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
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
MODEL_DIR=$PROJ/models/Qwen2.5-72B-Instruct
RUN_DIR=$PROJ/runs/smoke-transformers-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── conda + modules ──────────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier-module-stack.sh"
load_frontier_rocm_modules
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

export LD_PRELOAD=/opt/rocm-7.1.1/lib/libamdhip64.so${LD_PRELOAD:+:${LD_PRELOAD}}

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Torch:"
python - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

echo "[$(date)] Loading Qwen2.5-72B-Instruct with device_map=auto ..."
python - <<PY
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "$MODEL_DIR"
print(f"Model path: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"Model loaded. Device map: {model.hf_device_map}")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2? Answer in one sentence."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

print("Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )
response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n=== Response ===\n{response}\n================")
print("Smoke test PASSED")
PY

echo "[$(date)] Done."
