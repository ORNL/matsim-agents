#!/bin/bash
#SBATCH -J fte-pack4
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#SBATCH -t 06:00:00
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH --exclusive
# ---------------------------------------------------------------------------
# matsim-agents: pack up to FOUR finetune-eval campaigns onto ONE full GPU node
# (4x A100), one campaign per GPU, running concurrently. This turns the many
# single-GPU `-q shared` campaigns into a handful of full-node `-q premium`
# (or `-q regular`) jobs -- 2x charge but much higher scheduling priority and no
# wasted GPUs (all 4 are used).
#
# Each campaign REUSES the tested single-campaign launcher
# (job-finetune-eval-perlmutter.sh) unchanged: this script just pins it to a
# distinct GPU (CUDA_VISIBLE_DEVICES) and hands it per-campaign env overrides,
# so variant tags, caches and output dirs stay identical to the shared-queue
# runs.
#
# Inputs (env)
# ------------
#   BACKEND   hydragnn | uma | mace   (shared by all campaigns on this node)
#   SPECS     up to 4 campaigns separated by ';'. Each campaign is a
#             space-separated list of VAR=VAL overrides passed to the single
#             launcher (CASE=... plus any variant knobs). Example:
#
#     BACKEND=mace SPECS="\
#       CASE=lifepo4-al-001 MACE_MODEL=medium; \
#       CASE=hea-bcc-al-001 MACE_MODEL=medium; \
#       CASE=cantor-fcc-al-001 MACE_MODEL=medium MACE_LORA=1; \
#       CASE=phosphorene-2d-al-001 MACE_MODEL=large" \
#       sbatch deployments/perlmutter/launchers/job-finetune-eval-pack4-perlmutter.sh
#
# Switch to the (free) regular queue with:  sbatch -q regular ...
# Fewer than 4 specs is fine (only the GPUs needed are used).
# ---------------------------------------------------------------------------
set -uo pipefail

# Slurm copies the batch script to its spool dir, so BASH_SOURCE points there;
# resolve the repo (and the sibling launcher) from PROJECT_ROOT / a known path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
SINGLE="${REPO}/deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh"
[[ ! -f "${SINGLE}" ]] && { echo "ERROR: single-campaign launcher not found: ${SINGLE}" >&2; exit 2; }

BACKEND="${BACKEND:?set BACKEND=hydragnn|uma|mace}"
SPECS="${SPECS:?set SPECS='CASE=...; CASE=...; ...' (up to 4, ';'-separated)}"
LOGDIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Split SPECS on ';' into an array (trim surrounding whitespace per entry).
IFS=';' read -ra CAMPAIGNS <<<"${SPECS}"
n=0
for c in "${CAMPAIGNS[@]}"; do
  c="$(echo "${c}" | xargs)"      # trim
  [[ -z "${c}" ]] && continue
  CLEAN[$n]="${c}"
  n=$((n+1))
done
(( n == 0 )) && { echo "ERROR: no campaigns parsed from SPECS" >&2; exit 2; }
(( n > 4 ))  && { echo "ERROR: at most 4 campaigns per node (got ${n})" >&2; exit 2; }

echo "=========================================="
echo "[Perlmutter FT+eval PACK4]"
echo "Date:    $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}   QOS: ${SLURM_JOB_QOS:-N/A}"
echo "Backend: ${BACKEND}"
echo "Node:    $(hostname)   GPUs: ${SLURM_GPUS_PER_NODE:-?}"
echo "Campaigns (${n}):"
for i in $(seq 0 $((n-1))); do echo "  gpu${i}: ${CLEAN[$i]}"; done
echo "=========================================="

declare -a PIDS GPUOF
for i in $(seq 0 $((n-1))); do
  gpu_log="${LOGDIR}/${SLURM_JOB_NAME:-fte-pack4}-${SLURM_JOB_ID:-$$}-gpu${i}.out"
  echo "[launch] gpu${i} -> ${gpu_log}"
  # Reuse the single-campaign launcher, pinned to one GPU with per-spec env.
  CUDA_VISIBLE_DEVICES="${i}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    env BACKEND="${BACKEND}" ${CLEAN[$i]} bash "${SINGLE}" >"${gpu_log}" 2>&1 &
  PIDS[$i]=$!
  GPUOF[$i]="${CLEAN[$i]}"
done

rc_total=0
for i in $(seq 0 $((n-1))); do
  if wait "${PIDS[$i]}"; then
    echo "[done] gpu${i} OK   (${GPUOF[$i]})"
  else
    rc=$?
    echo "[done] gpu${i} FAILED rc=${rc}   (${GPUOF[$i]})" >&2
    rc_total=1
  fi
done

echo "[$(date)] PACK4 complete (backend=${BACKEND}, ${n} campaigns, overall rc=${rc_total})."
exit "${rc_total}"
