#!/bin/bash
#SBATCH -A m5216
#SBATCH -J discovery-chat
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: discovery run on NERSC Perlmutter using HuggingFace
# Transformers as the LLM backend (no vLLM server required).
#
# Layout:
#   • HuggingFace pipeline : all 4 A100s via device_map="auto" (Accelerate)
#   • matsim-agents chat   : --llm-provider huggingface (loads model inline)
#
# Usage:
#   sbatch scripts/advanced/perlmutter/job-discovery-chat-perlmutter.sh
#
# Override model at submission:
#   MATSIM_MODEL_DIR=.../Qwen3-32B MATSIM_MODEL_NAME=Qwen/Qwen3-32B \
#     sbatch scripts/advanced/perlmutter/job-discovery-chat-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
# Slurm copies the submitted script into a spool dir before executing it, so
# BASH_SOURCE[0] does NOT point at this file's real location under sbatch --
# fall back to the known repo-absolute path when the relative lookup misses.
RUNTIME_ENV="${SCRIPT_DIR}/../common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || RUNTIME_ENV=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents/scripts/advanced/common/runtime-env.sh
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}" "/global/cfs/projectdirs/m5216/mlupopa/matsim-agents")"
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
init_run_dirs "$PROJ" "discovery-chat" "${SLURM_JOB_ID}"

# ── modules & conda env ──────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

# Make HydraGNN example utilities importable (inference_fused, etc.)
export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}

# ── environment ──────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Hard-block all remote fetches (compute nodes have no outbound internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Point matsim-agents at the local model directory
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH=$MODEL_DIR

# CUDA / NCCL knobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

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
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
