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

E4-B selects a specific model-code checkout rather than the default upstream
source. Follow [the frozen HydraGNN source requirement](../docs/hydragnn-frozen-source.md)
to request repository `allaffa/HydraGNN` at commit
`81d5b88304f20bb45c840571e2e26c490d1b19da` through the Aurora installer's
`HYDRAGNN_REPO`/`HYDRAGNN_REF` settings. Clone-only preparation does not require
running an installer or modifying a working environment.

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

For Qwen3-235B E1 collections on Aurora, use the dedicated runner. It selects
the graph-enabled resource plan from `E1_MODEL_KEY`, rejects eager-mode plans,
and sets `VLLM_XPU_ENABLE_XPU_GRAPH=1`. Both Instruct and Thinking use TP=8 and
`--max-num-seqs 1` with PIECEWISE-only graph capture. An earlier Thinking run
using FULL_AND_PIECEWISE captured graphs but native-aborted during full decode
graph replay; the current plans avoid that path while retaining piecewise XPU
graphs. Old plans containing `--enforce-eager` silently disabled graph use, so
the dedicated runner now rejects them.

```bash
qsub -A "$ALLOCATION" \
	-v "PROJECT_ROOT=$PWD,MODEL_ROOT=$MODEL_ROOT,E1_MODEL_KEY=qwen3-235b-instruct-2507,E1_MANIFEST=$E1_MANIFEST" \
	deployments/aurora/jobs/job-e1-qwen3-235b-aurora.sh
```

Point `E1_MANIFEST` at the one-question, 2,048-token Instruct canary manifest
first; promote to the full manifest only after checking the run artifacts and
PBS status. Instruct PIECEWISE canary job `9036264` completed on Nov 3, 2026
with one response, zero failures, and PBS exit status 0. At the recorded
checkpoint, Thinking PIECEWISE canary `9036266` and the 18-question Instruct
campaign `9036469` were still running, so they are not yet qualified results.
See [`docs/vllm-aurora.md`](../docs/vllm-aurora.md#qwen3-235b-e1-with-xpu-graphs)
for the tested module stack, reproducible submission commands, log checks, and
incident history.

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
