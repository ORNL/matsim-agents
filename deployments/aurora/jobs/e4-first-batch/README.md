# E4 first production batch (`e4-first-batch-01`)

Twenty approved production DFT calculations: 8 BCC refractory alloys
(HydraGNN), 4 Cr-Mn-Fe-Co-Ni FCC (GenesisHackathonOct26), and 8 LiFePO4
antisite supercells (GridAI). Each calculation gets one initial one-node,
24-hour `capacity` job; the whole batch is capped at 480 node-hours.

| File | Purpose |
|---|---|
| `batch-manifest.json` | Frozen case list, input/PAW hashes, allocations, and resource caps |
| `prepare_batch.py` | Builds the private input directories (outside the repository) |
| `job-bcc.pbs`, `job-fcc.pbs`, `job-lifepo4.pbs` | Family wrappers around `../job-e4-vasp-ab-aurora.pbs` |
| `submit_batch.py` | Validating submission helper; dry-run by default |
| `submissions.jsonl` | Append-only record of accepted PBS IDs and rejected `qsub` calls |

POTCAR bytes are never stored here. Inputs reference PAWs by label, path, and
SHA-256 hash only.

## Material protocols

| Family | Ranks x threads | KPAR / NCORE | Stages | Ionic limit | EDIFFG |
|---|---|---|---|---|---|
| BCC | 8 x 8 | 8 / - | Low -> Normal | 240 cumulative per stage | -0.01 |
| FCC | 8 x 8 | 8 / - | single | 200 | -0.02 |
| LiFePO4 | 12 x 8 | 1 / 12 | single (Accurate) | 200 | -0.01 |

`KPAR` is material-specific; the shared runner rejects inconsistent
`KPAR`/`NCORE`/rank combinations instead of inheriting BCC settings.

## Environment

PBS supplies `E4_CASE_DIR` and `E4_CALCULATION_ID` via `qsub -v`; the helper
sets both. Each family wrapper then exports the variables required by the
shared runner, for example (BCC):

```bash
REPO CASE_DIR=$E4_CASE_DIR VARIANT=alcf REPEAT=repeat-01
E4_STAGE_MODE=low-normal E4_STAGE_NSW_MAX=240 E4_EDIFFG=-0.01
E4_MPI_RANKS=8 E4_OMP_THREADS=8
E4_REQUESTED_NODES=1 E4_REQUESTED_WALLTIME_SECONDS=86400
E4_SOFT_STOP_SECONDS=82800 E4_CHECKPOINT_SECONDS=3600
```

Single-stage families also set `E4_STAGE_MODE=single`,
`E4_SINGLE_PRECISION`, and `E4_SINGLE_PREFIX`. The runner pins the ALCF
VASP 6.6.1 executable and verifies its SHA-256 before launch.

## Usage

```bash
# Validate everything; never calls qstat or qsub.
.venv/bin/python deployments/aurora/jobs/e4-first-batch/submit_batch.py

# Validate a staged subset only.
.venv/bin/python deployments/aurora/jobs/e4-first-batch/submit_batch.py \
  --only E4-FCC-0001 E4-LFP-0002

# Submit (only after explicit authorization).
.venv/bin/python deployments/aurora/jobs/e4-first-batch/submit_batch.py \
  --submit --only E4-FCC-0001 E4-LFP-0002
```

The helper revalidates the executable checksum, input and PAW hashes,
allocation-to-family mapping, PBS directives, output-directory collisions for
the selected cases, active duplicate jobs, and resource caps. It stops at the
first `qsub` failure and never retries automatically. A rejected `qsub` is
recorded but does not count as an accepted submission.

## Scheduler limits

Aurora enforces a per-user job limit across all projects. Submissions beyond
it fail with `qsub: would exceed queue generic's per-user limit` and create
no job. Queued jobs can also wait with `User has reached queue capacity
running job limit` or an overall `nodect` limit. Submit the remaining approved
cases with `--only` once capacity frees; switching allocations does not bypass
these limits.

## Qualification

Running, queued, interrupted, or unconverged calculations are not scientific
results. A case qualifies only after ionic convergence is verified from final
force norms against its family's EDIFFG, within its cumulative ionic-step
limit. Restarts preserve the calculation ID and cumulative step counts.
