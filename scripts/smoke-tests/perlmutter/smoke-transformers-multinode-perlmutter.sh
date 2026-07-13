#!/bin/bash
# ---------------------------------------------------------------------------
# smoke-transformers-multinode-perlmutter.sh
#
# Multi-node HuggingFace smoke test on Perlmutter using torchrun + the
# integrated transformers tensor-parallel planner (``tp_plan="auto"``).
# One rank per GPU, NCCL across nodes — no DeepSpeed.
#
# Submit:
#   sbatch -N 2 scripts/smoke-tests/perlmutter/smoke-transformers-multinode-perlmutter.sh
#
# Override model:
#   sbatch --export=ALL,MATSIM_MODEL_DIR=/path/to/model \
#          scripts/smoke-tests/perlmutter/smoke-transformers-multinode-perlmutter.sh
#
# The launchers under scripts/launchers/perlmutter/ override --nodes via sbatch.
# ---------------------------------------------------------------------------
#SBATCH -A m5216
#SBATCH -J smoke-tf-mn-pm
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 00:45:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/smoke-tf-mn-pm-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/smoke-tf-mn-pm-%j/job-%j.out

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv

MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}
LOADER="${SCRIPT_DIR}/_torchrun_smoke_loader.py"

RUN_DIR=$PROJ/runs/smoke-tf-mn-pm-${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

# ── modules + venv ───────────────────────────────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

# ── torch/HF environment ─────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}

# Pass model path through to the loader
export MATSIM_HF_MODEL_PATH="$MODEL_DIR"
export MATSIM_LLM_PROVIDER=huggingface

# ── torch.distributed rendezvous (head node = SLURM_NODELIST[0]) ────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn}      # Perlmutter Slingshot
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
# Token used to silence transformers' tp_plan warning (one process per GPU is required)
export TRANSFORMERS_VERBOSITY=info

NNODES=$SLURM_JOB_NUM_NODES
NPROC_PER_NODE=$SLURM_NTASKS_PER_NODE
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))

echo "=========================================="
echo "[Perlmutter multi-node HF smoke]"
echo "Date:           $(date)"
echo "Job ID:         ${SLURM_JOB_ID:-N/A}"
echo "Nodes:          $NNODES  ($SLURM_NODELIST)"
echo "Ranks/node:     $NPROC_PER_NODE"
echo "World size:     $WORLD_SIZE"
echo "Master:         $MASTER_ADDR:$MASTER_PORT"
echo "Model:          $MODEL_NAME"
echo "Model dir:      $MODEL_DIR"
echo "Loader:         $LOADER"
echo "=========================================="

python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"local_devices={torch.cuda.device_count()}")
PY

# srun runs one task per GPU. Inside each task we have a single python process
# that initialises ``torch.distributed`` directly using the RANK/LOCAL_RANK env
# vars we export below. This avoids torchrun's per-node elastic agent (which
# would conflict with srun's own task-launching).
srun --kill-on-bad-exit=1 \
     --label \
     bash -c '
       export RANK=$SLURM_PROCID
       export LOCAL_RANK=$SLURM_LOCALID
       export WORLD_SIZE='"$WORLD_SIZE"'
       export LOCAL_WORLD_SIZE='"$NPROC_PER_NODE"'
       exec python -u '"$LOADER"'
     '

status=$?
echo "[$(date)] srun exit status: $status"
exit $status
