#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full active-learning loop (UMA + VASP by default) for every ready
# paper case.
#
# cu_bht is excluded by default because it requires a supplied CIF seed;
# add it once the seed is in place, or pass CASES explicitly.
#
# Usage:
#   deployments/perlmutter/jobs/submit-active-learning-paper-cases.sh
#   CASES="hea_bcc hea_fcc" deployments/perlmutter/jobs/submit-active-learning-paper-cases.sh
#   MLIP_CONFIG=examples/mlip/mace-mp-medium.yaml DFT_BACKEND=qe deployments/perlmutter/jobs/submit-active-learning-paper-cases.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
JOB="${SCRIPT_DIR}/job-active-learning-paper-cases-perlmutter.sh"

CASES="${CASES:-lifepo4 hea_bcc hea_fcc phosphorene zn_formate}"
if [[ -n "${MLIP_CONFIG:-}" ]]; then
  MLIP_CONFIG="$(cd "$(dirname "$MLIP_CONFIG")" && pwd)/$(basename "$MLIP_CONFIG")"
fi

echo "Submitting AL paper-case jobs: $CASES"
for c in $CASES; do
  EXPORT="ALL,CASE=$c"
  [[ -n "${MLIP_CONFIG:-}" ]] && EXPORT="$EXPORT,MLIP_CONFIG=$MLIP_CONFIG"
  [[ -n "${DFT_BACKEND:-}" ]]  && EXPORT="$EXPORT,DFT_BACKEND=$DFT_BACKEND"
  jid=$(sbatch --parsable --export="$EXPORT" "$JOB")
  echo "  $c -> job $jid"
done
echo "Done. Track with: squeue --me"
