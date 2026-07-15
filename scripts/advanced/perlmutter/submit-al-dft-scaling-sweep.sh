#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the DFT-concurrency strong-scaling sweep: one AL iteration with a
# fixed 16-job DFT workload at N = 1, 2, 4, 8 nodes.
#
#   scripts/advanced/perlmutter/submit-al-dft-scaling-sweep.sh
#   NODES="1 2 4" QOS=premium scripts/.../submit-al-dft-scaling-sweep.sh
#
# After all four jobs finish, build the figure with:
#   fairchem_venv/bin/python docs/paper/figures/plot_dft_scaling.py \
#       --runs-root $PROJ/runs --out docs/paper/figures/al_dft_scaling.pdf
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/job-al-dft-scaling-perlmutter.sh"
[[ -f "$JOB" ]] || { echo "ERROR: missing $JOB" >&2; exit 2; }

NODES="${NODES:-1 2 4 8}"
QOS="${QOS:-regular}"

echo "Submitting DFT strong-scaling sweep (QOS=$QOS): N in [$NODES]"
for n in $NODES; do
  jid=$(sbatch -N "$n" -q "$QOS" --parsable "$JOB")
  echo "  N=$n  ->  job $jid"
done
echo "Done. Track with: squeue -u \$USER -n al-dft-scaling"
