#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the DFT-concurrency strong-scaling sweep on OLCF Frontier: one AL
# iteration with a fixed 16-job DFT workload at N = 1, 2, 4, 8 nodes.
#
#   scripts/advanced/frontier/submit-al-dft-scaling-sweep.sh
#   NODES="1 2 4" PARTITION=batch scripts/.../submit-al-dft-scaling-sweep.sh
#
# NOTE: Frontier's `debug` QOS is limited to small node counts / short walltime;
# the larger scaling points (N=4, 8) must run in the standard `batch` partition.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/job-al-dft-scaling-frontier.sh"
[[ -f "$JOB" ]] || { echo "ERROR: missing $JOB" >&2; exit 2; }

NODES="${NODES:-1 2 4 8}"
PARTITION="${PARTITION:-batch}"

echo "Submitting DFT strong-scaling sweep (partition=$PARTITION): N in [$NODES]"
for n in $NODES; do
  jid=$(sbatch -N "$n" -p "$PARTITION" --parsable "$JOB")
  echo "  N=$n  ->  job $jid"
done
echo "Done. Track with: squeue -u \$USER -n al-dft-scaling"
