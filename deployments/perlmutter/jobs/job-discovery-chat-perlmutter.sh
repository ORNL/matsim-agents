#!/bin/bash
#SBATCH -J discovery-chat
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: discovery run on NERSC Perlmutter with the LLM loaded
# in-process through Hugging Face Transformers (no vLLM server).
#
# Placement comes from a resource layout (default
# deployments/perlmutter/resources/discovery-hf-80gb.yaml):
#   • Qwen2.5-72B in-process : all 4 A100-80GB, at most 75% of GPU 0
#   • MLIP relaxations       : one worker on GPU 0, at most 20% of its memory
#
# Usage:
#   sbatch deployments/perlmutter/jobs/job-discovery-chat-perlmutter.sh
#
# Another model or layout at submission:
#   MATSIM_RESOURCES=/path/to/layout.yaml MODEL_ROOT=$PROJ/models \
#     sbatch deployments/perlmutter/jobs/job-discovery-chat-perlmutter.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv
source "$REPO/deployments/common/hydragnn-env.sh"
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_BEST6_LOGDIR}
init_run_dirs "$PROJ" "discovery-chat" "${SLURM_JOB_ID}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJ/models}"
export PROJECT_ROOT="$REPO"
RESOURCES="${MATSIM_RESOURCES:-$REPO/deployments/perlmutter/resources/discovery-hf-80gb.yaml}"

# ── modules & conda env ──────────────────────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
# The MLIP: MLIP_CONFIG (any YAML with an `mlip:` block, e.g. examples/mlip/*.yaml)
# or the HydraGNN foundation model above. The client runs in that backend's environment.
MLIP_CONFIG="$(default_mlip_config "$RUN_DIR" "$LOGDIR")"
VENV="$(mlip_backend_venv "$REPO" "$(mlip_backend "$VENV/bin/python" "$MLIP_CONFIG")" "$VENV")"
activate_python_env "$VENV"

# Offline compute nodes: stage the selected model's downloaded artifacts.
source "$REPO/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mlip_model_artifacts "$REPO" python "$MLIP_CONFIG"

# Make HydraGNN example utilities importable (inference_fused, etc.)
hydragnn_export_pythonpath

# ── environment ──────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Hard-block all remote fetches (compute nodes have no outbound internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# CUDA / NCCL knobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Resources: $RESOURCES"
matsim-agents resources plan "$RESOURCES"
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


echo "[$(date)] Submitting discovery query to matsim-agents (in-process Hugging Face model) ..."
echo "$QUERY" | matsim-agents chat \
    --mlip-config     "$MLIP_CONFIG" \
    --output-dir      "$OUTPUT_DIR" \
    --resources       "$RESOURCES" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
