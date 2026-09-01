#!/bin/bash
#SBATCH -J discovery-chat
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: discovery-chat run on Frontier using HuggingFace Transformers
# as the LLM backend (no vLLM server required).
#
# Layout:
#   • HuggingFace pipeline : all 8 GCDs via device_map="auto" (Accelerate)
#   • matsim-agents chat   : --llm-provider huggingface (loads model inline)
#
# Usage:
#   sbatch deployments/frontier/jobs/job-discovery-chat-frontier.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=.../Qwen3-32B MATSIM_MODEL_NAME=Qwen/Qwen3-32B \
#     sbatch deployments/frontier/jobs/job-discovery-chat-frontier.sh
#
# Override discovery prompt:
#   MATSIM_DISCOVERY_QUERY="Propose candidate materials for ..." \
#     sbatch deployments/frontier/jobs/job-discovery-chat-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64
HYDRAGNN_BRANCH_MLP_CHECKPOINT=$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
init_run_dirs "$PROJ" "discovery-chat" "${SLURM_JOB_ID}"

# ── modules & conda env ──────────────────────────────────────────────────────
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
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

# ── discovery query ──────────────────────────────────────────────────────────
if [[ -n "${MATSIM_DISCOVERY_QUERY:-}" ]]; then
    QUERY="${MATSIM_DISCOVERY_QUERY}"
else
    QUERY="$(cat <<'EOF'
Propose 3 to 5 candidate inorganic materials for high-temperature structural applications.
For each candidate, provide the formula, likely crystal family, and a brief physics-based
justification. Then relax the proposed structures with the MLFF and summarize relative
stability from final energies and residual forces.
EOF
)"
fi

echo "[$(date)] Submitting discovery query to matsim-agents (HuggingFace provider) ..."
echo "$QUERY" | matsim-agents chat \
    --logdir          "$LOGDIR" \
    --hydragnn-branch-mlp-checkpoint "$HYDRAGNN_BRANCH_MLP_CHECKPOINT" \
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
