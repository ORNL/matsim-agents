#!/bin/bash
#SBATCH -A mat746
#SBATCH -J rhea-hf
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/rhea-hf-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/rhea-hf-%j/job-%j.out
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: RHEA discovery run on Frontier using HuggingFace Transformers
# as the LLM backend (no vLLM server required).
#
# Layout:
#   • HuggingFace pipeline : all 8 GCDs via device_map="auto" (Accelerate)
#   • matsim-agents chat   : --llm-provider huggingface (loads model inline)
#
# Usage:
#   sbatch scripts/frontier/job-rhea-transformers-frontier.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=.../Qwen3-32B MATSIM_MODEL_NAME=Qwen/Qwen3-32B \
#     sbatch scripts/frontier/job-rhea-transformers-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64
MLP_CHECKPOINT=$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
RUN_DIR=$PROJ/runs/rhea-hf-$SLURM_JOB_ID
OUTPUT_DIR=$RUN_DIR/outputs

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & conda env ──────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$PROJ/matsim-agents/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

# Make HydraGNN example utilities importable (inference_fused, etc.)
export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}

# ── environment ──────────────────────────────────────────────────────────────
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Hard-block all remote fetches (compute nodes have no outbound internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Point matsim-agents at the local model directory
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH=$MODEL_DIR

# ROCm env
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_ARCH=gfx90a
TORCH_LIB=$VENV/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Model:  $MODEL_NAME  ($MODEL_DIR)"
python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# ── RHEA query ───────────────────────────────────────────────────────────────
QUERY="Propose 4 to 5 refractory high-entropy alloy compositions using elements \
from Mo, Nb, Ta, W, V, Cr, Hf, Zr, Ti that are known for combined \
high-temperature resistance and mechanical strength. For each composition \
specify the relevant crystal phases (e.g. BCC, B2, HCP) and explain the \
physical justification. Then relax each proposed structure using the MLFF \
and report the final energies and which phases are most stable."

echo "[$(date)] Submitting RHEA query to matsim-agents (HuggingFace provider) ..."
echo "$QUERY" | matsim-agents chat \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --llm-provider    huggingface \
    --llm-model       "$MODEL_DIR" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --min-atoms       64 \
    --n-orderings     2 \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
