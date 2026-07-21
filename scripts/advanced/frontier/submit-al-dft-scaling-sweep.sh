#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the DFT-concurrency strong-scaling sweep on OLCF Frontier: one AL
# iteration with a fixed 32-job DFT workload at N = 1, 2, 4, 8 nodes, repeated
# REPS times. Each repeat draws an independent random workload (via AL_SEED)
# held fixed across node counts within the repeat, so averaging the per-N wall
# times over repeats dampens per-draw variability and yields a robust
# strong-scaling estimate. Mirrors the Perlmutter sweep wrapper.
#
#   scripts/advanced/frontier/submit-al-dft-scaling-sweep.sh
#   NODES="1 2 4" REPS=5 PARTITION=batch scripts/.../submit-al-dft-scaling-sweep.sh
#
# NOTE: Frontier's `debug` QOS is limited to small node counts / short walltime;
# the larger scaling points (N=4, 8) must run in the standard `batch` partition.
# Pass QOS=debug for a quick small-N smoke test.
#
# After all jobs finish, build the figure (mean +/- std over repeats) with:
#   docs/paper/figures/plot_dft_scaling.py --runs-root $PROJ/runs
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/job-al-dft-scaling-frontier.sh"
[[ -f "$JOB" ]] || { echo "ERROR: missing $JOB" >&2; exit 2; }

NODES="${NODES:-1 2 4 8}"
REPS="${REPS:-5}"
REP_START="${REP_START:-1}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-}"
WALL="${WALL:-}"

echo "Submitting DFT strong-scaling sweep (partition=$PARTITION): N in [$NODES] x REPS=$REP_START..$REPS"
for r in $(seq "$REP_START" "$REPS"); do
  for n in $NODES; do
    jid=$(sbatch -N "$n" -p "$PARTITION" ${QOS:+-q "$QOS"} ${WALL:+-t "$WALL"} --export=ALL,REP="$r" --parsable "$JOB")
    echo "  N=$n  rep=$r  ->  job $jid"
  done
done
echo "Done. Track with: squeue -u \$USER -n al-dft-scaling"
