#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full UMA+VASP active-learning loop for every ready paper case.
#
# cu_bht is excluded by default because it requires a supplied CIF seed;
# add it once the seed is in place, or pass CASES explicitly.
#
# Usage:
#   scripts/advanced/perlmutter/submit-active-learning-paper-cases.sh
#   CASES="hea_bcc hea_fcc" scripts/advanced/perlmutter/submit-active-learning-paper-cases.sh
#   MLIP_BACKEND=hydragnn DFT_BACKEND=qe scripts/advanced/perlmutter/submit-active-learning-paper-cases.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
JOB="${SCRIPT_DIR}/job-active-learning-paper-cases-perlmutter.sh"

CASES="${CASES:-lifepo4 hea_bcc hea_fcc phosphorene zn_formate}"

echo "Submitting AL paper-case jobs: $CASES"
for c in $CASES; do
  EXPORT="ALL,CASE=$c"
  [[ -n "${MLIP_BACKEND:-}" ]] && EXPORT="$EXPORT,MLIP_BACKEND=$MLIP_BACKEND"
  [[ -n "${DFT_BACKEND:-}" ]]  && EXPORT="$EXPORT,DFT_BACKEND=$DFT_BACKEND"
  jid=$(sbatch --parsable --export="$EXPORT" "$JOB")
  echo "  $c -> job $jid"
done
echo "Done. Track with: squeue --me"
