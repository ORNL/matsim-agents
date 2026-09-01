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
