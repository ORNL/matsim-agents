# Deployments

Machine-specific setup, launch, job, smoke-test, and configuration assets live
under one directory per supported leadership-class system.  Runtime Python code
must not import from this tree; platform selection belongs at deployment time.

- `frontier/`: OLCF Frontier (Slurm)
- `aurora/`: ALCF Aurora (PBS Pro)
- `perlmutter/`: NERSC Perlmutter (Slurm)

## One-command Python installation

Each machine has one canonical installer. Run it from a login node with network
access:

```bash
# OLCF Frontier
bash deployments/frontier/setup/install.sh

# ALCF Aurora
bash deployments/aurora/setup/install.sh

# NERSC Perlmutter
bash deployments/perlmutter/setup/install.sh
```

Every entry point clones or updates `ORNL/HydraGNN`, runs HydraGNN's current
facility recipe under `scripts/hpc/` first, and installs non-editable HydraGNN
and matsim-agents packages into `matsim-agents/.venv`. Compiled dependency
build trees are kept under `matsim-agents/.hpc-build/`. The script
finishes with `pip check` and import checks; a successful exit therefore means
there is one Python environment for the HydraGNN-based matsim workflow.

Common overrides are `HYDRAGNN_DIR`, `HYDRAGNN_REF`, `INSTALL_ROOT`,
`VENV_PATH`, and `MATSIM_EXTRAS`. Set `INSTALL_UMA=1` to install
`fairchem-core` through the matsim-agents `uma` extra and verify its calculator
API in the same environment. Set `INSTALL_MACE=1` to have the same canonical
installer create `matsim-agents/.venv-mace`. Frontier also accepts HydraGNN's
`SKIP_VLLM` setting. Set `RECREATE_ENV=1` on Frontier or Aurora to request a
clean rebuild; HydraGNN's current Perlmutter recipe always recreates its target
environment.

The environment and its build artifacts are owned by the matsim-agents
checkout; the HydraGNN checkout is source input only. “Self-contained” applies
to Python packages. Facility modules and native GPU,
MPI, Quantum ESPRESSO, and VASP runtimes remain external system dependencies.
MACE cannot be installed in `.venv`: upstream `mace-torch==0.3.16` declares
`e3nn==0.4.4`, while HydraGNN declares `e3nn==0.5.1`. The optional
`.venv-mace` is also owned by matsim-agents and inherits the expensive
facility-specific PyTorch stack from `.venv`, but shadows e3nn in its own
site-packages. Run MACE as a separate process; do not import HydraGNN and MACE
in the same Python interpreter.
See [`docs/mace-dependencies.md`](../docs/mace-dependencies.md) for the checked
upstream constraints and the qualification boundary.
FairChem/UMA is compatible with the updated Python dependency contract, but
UMA weights are gated and accelerator execution must still be qualified on
each facility.

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
