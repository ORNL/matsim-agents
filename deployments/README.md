# Deployments

Machine-specific setup, launch, job, smoke-test, and configuration assets live
under one directory per supported leadership-class system.  Runtime Python code
must not import from this tree; platform selection belongs at deployment time.

- `frontier/`: OLCF Frontier (Slurm)
- `aurora/`: ALCF Aurora (PBS Pro)
- `perlmutter/`: NERSC Perlmutter (Slurm)

## Submission contract

The checked-in scripts intentionally do not contain allocation IDs, usernames,
or project filesystem paths. Pass the allocation at submission time and export
the checkout when the scheduler may spool a copy of the script:

```bash
# Frontier / Perlmutter
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD" deployments/<site>/jobs/<job>.sh

# Aurora
qsub -A <allocation> -v PROJECT_ROOT="$PWD" deployments/aurora/jobs/<job>.sh
```

Use `RUNS_ROOT` to place output on the site's scratch filesystem. Site scripts
must fail fast when required binaries, environments, checkpoints, or model
weights are absent; they must not silently fall back to another user's paths.

## Validation status

`python scripts/diagnostics/validate_deployments.py` performs the portable
checks available off-system: shell syntax, repository references, obsolete CLI
flags, embedded allocation IDs, and unsafe absolute scheduler output paths.
Successful execution on Frontier, Aurora, and Perlmutter remains a release gate:
record the module list, accelerator visibility, rank mapping, torch build, and a
short MLIP plus DFT smoke result before marking a toolchain combination validated.

Use `benchmarks/portability/` and each facility's
`job-portability-benchmark-<facility>.sh` as the shared release gate. Those
scripts record comparable machine metadata and use an identical scientific
configuration; specialized scaling and paper scripts are not portability gates.

## Shared workflow runners

Facility job files own only scheduler directives, module/venv activation, and
hardware-specific launcher geometry. They delegate scientific configuration to:

- `common/run-mlip-relaxation.sh` for the typed `matsim-agents relax` contract;
- `common/run-active-learning.sh` for validated `matsim-agents al run` jobs.

The active-learning runner defaults to collecting DFT labels without
retraining. `MATSIM_RETRAIN=1` additionally requires `MATSIM_TRAIN_SCRIPT`;
model promotion remains disabled and must happen through an explicit reviewed
workflow. Set `MATSIM_DFT_BACKEND=qe|vasp` and provide the corresponding binary,
pseudopotential/POTCAR, wrapper, and rank variables.

Files marked `LEGACY MANUSCRIPT REPRODUCTION` reproduce an earlier published
pipeline and are not templates for new campaigns. Current deployments should
use the typed relaxation, active-learning, portability, or supervisor contracts.
