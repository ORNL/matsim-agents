#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the DFT-concurrency strong-scaling sweep: one AL iteration with a
# 32-job DFT workload at N = 1, 2, 4, 8 nodes, repeated REPS times. Each repeat
# draws an independent random workload (via AL_SEED) that is held fixed across
# node counts within the repeat, so averaging the per-N wall times over repeats
# dampens per-draw variability and yields a robust strong-scaling estimate.
#
#   deployments/perlmutter/jobs/submit-al-dft-scaling-sweep.sh
#   NODES="1 2 4" REPS=5 QOS=regular scripts/.../submit-al-dft-scaling-sweep.sh
#
# After all jobs finish, build the figure (mean +/- std over repeats) with:
#   hydragnn_venv/bin/python research/paper/manuscript/figures/plot_dft_scaling.py \
#       --runs-root $PROJ/runs --out research/paper/manuscript/figures/al_dft_scaling.pdf
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/job-al-dft-scaling-perlmutter.sh"
[[ -f "$JOB" ]] || { echo "ERROR: missing $JOB" >&2; exit 2; }

NODES="${NODES:-1 2 4 8}"
REPS="${REPS:-5}"
REP_START="${REP_START:-1}"
QOS="${QOS:-premium}"
WALL="${WALL:-}"

echo "Submitting DFT strong-scaling sweep (QOS=$QOS): N in [$NODES] x REPS=$REP_START..$REPS"
for r in $(seq "$REP_START" "$REPS"); do
  for n in $NODES; do
    jid=$(sbatch -N "$n" -q "$QOS" ${WALL:+-t "$WALL"} --export=ALL,REP="$r" --parsable "$JOB")
    echo "  N=$n  rep=$r  ->  job $jid"
  done
done
echo "Done. Track with: squeue -u \$USER -n al-dft-scaling"
