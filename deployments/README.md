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
facility recipe first, and installs non-editable HydraGNN
and matsim-agents packages into `matsim-agents/.venv`. Compiled dependency
build trees are kept under `matsim-agents/.hpc-build/`. The script
finishes with `pip check` and import checks; a successful exit therefore means
there is one Python environment for the HydraGNN-based matsim workflow.

The delegated HydraGNN recipes are
`scripts/hpc/olcf/frontier/installation/install-rocm72.sh`,
`scripts/hpc/alcf/aurora/installation/install.sh`, and
`scripts/hpc/nersc/perlmutter/installation/install.sh` in the HydraGNN checkout.

Common overrides are `HYDRAGNN_DIR`, `HYDRAGNN_REF`, `INSTALL_ROOT`,
`VENV_PATH`, and `MATSIM_EXTRAS`. By default (`INSTALL_UMA=1`) the installer installs
`fairchem-core` through the matsim-agents `uma` extra and verifies its calculator
API in the matsim-owned `.venv-uma` compatibility environment. By default
(`INSTALL_MACE=1`) the same canonical
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
FairChem/UMA shares the NumPy/SciPy contract but not the Torch contract:
FairChem 2.22 requires Torch 2.13, while HydraGNN requires Torch 2.14.
`INSTALL_UMA=1` therefore creates the matsim-owned `.venv-uma`. UMA weights
are gated, and accelerator execution must still be qualified on each facility.

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

For Mistral Small 24B on Aurora, prepare its HF tokenizer overlay from the local
model snapshot and pass that directory to the tracked resource plan:

```bash
python deployments/aurora/setup/prepare_mistral_tokenizer.py \
	"$MODEL_ROOT/Mistral-Small-24B-Instruct-2501" \
	"$PWD/runs/mistral-small-24b-tokenizer"
qsub -A "$ALLOCATION" \
	-v "PROJECT_ROOT=$PWD,MODEL_ROOT=$MODEL_ROOT,E1_MANIFEST=$E1_MANIFEST,MATSIM_RESOURCES=$PWD/deployments/aurora/resources/mistral-small-24b.yaml,MISTRAL_TOKENIZER_DIR=$PWD/runs/mistral-small-24b-tokenizer" \
deployments/aurora/jobs/job-e1-development-aurora.sh
```

For Qwen3-235B Instruct on Aurora, the graph-enabled resource plan omits
`--enforce-eager` and keeps tensor parallelism at 8. Pass the XPU Graph setting
to the server through the PBS job environment:

```bash
qsub -A "$ALLOCATION" \
	-v "PROJECT_ROOT=$PWD,MODEL_ROOT=$MODEL_ROOT,E1_MANIFEST=$E1_MANIFEST,MATSIM_RESOURCES=$PWD/deployments/aurora/resources/qwen3-235b-instruct-xpu-graph.yaml,MATSIM_VLLM_TIMEOUT_SECONDS=3600,MATSIM_VLLM_MAX_RETRIES=0,VLLM_XPU_ENABLE_XPU_GRAPH=1" \
deployments/aurora/jobs/job-e1-development-aurora.sh
```

Use this profile for a bounded canary first; graph initialization can increase
server startup time. The 2,048-token E1 canary completed with valid structured
output and substantially higher observed decode throughput than the eager
baseline on the tested Aurora software stack.

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
