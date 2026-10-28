# Aurora Setup Scripts for matsim-agents

This directory provides an Aurora-focused setup flow that mirrors the phased
install style used by the Frontier and Perlmutter scripts.

The single full-install entry point is:

```bash
bash deployments/aurora/setup/install.sh
```

It preserves an existing HydraGNN checkout unless `HYDRAGNN_REF` is explicitly
set, runs HydraGNN's current Aurora installer, then installs and verifies
matsim-agents wheels in that environment.

The phases follow the Perlmutter installer: HydraGNN dependencies, project
wheels in `.venv`, then optional isolated UMA and MACE environments. Aurora
loads `FRAMEWORKS_MODULE` (default `frameworks`) in the parent shell first so
PyTorch and every child installer inherit the Intel runtime libraries.

The HydraGNN extra accepts PyTorch 2.13 and 2.14, including Aurora's 2.13 XPU
pre-release, with the corresponding torchvision 0.28/0.29 series. A historical
exact pin in a feature branch's requirements is not evidence that its model
code cannot run on those versions. The wrapper stages the selected code with
requirements from the latest ORNL/HydraGNN main, under `.hpc-build/aurora`,
without modifying the selected checkout. The staged Torch requirement is pinned
to the framework-provided build, including its XPU version suffix. NumPy, SciPy,
PyG and the remaining requirements come from upstream, not the feature branch.
Missing package initializers are added only to the staged copy so setuptools
includes the selected branch's utility modules in the wheel.

The full `pip check` is reported but is not fatal, matching Perlmutter: Aurora's
shared framework can contain unrelated dependency conflicts. A separate fatal
audit checks the complete dependency tree of the selected Matsim extras and
HydraGNN, followed by fatal import checks. UMA and MACE preserve the framework
Torch/torchvision builds with exact constraints and verify Torch's source path.

After a completed HydraGNN dependency build, set
`REUSE_HYDRAGNN_DEPENDENCIES=1` to resume without rebuilding its native libraries.
This requires an existing environment, `RECREATE_ENV=0`, working compiled PyG
imports, and the same subsequent dependency and import checks.

```bash
RECREATE_ENV=1 RUN_MODEL_TESTS=1 bash deployments/aurora/setup/install.sh
```

`RUN_MODEL_TESTS=1` enables both compatibility environments, checks their model
imports, and runs the Matsim-Agents suite in the base environment from the
repository root, making source-only benchmark and diagnostic modules importable. It does not
download model weights or qualify XPU inference; gated-model and hardware
tests still require their own prerequisites.

To keep installation and tests running after logout:

```bash
RUN_DIR="$PWD/.hpc-build/aurora/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
nohup setsid env RUN_DIR="$RUN_DIR" \
  bash deployments/aurora/setup/run-install-tests.sh \
  < /dev/null > "$RUN_DIR/install.log" 2>&1 &
disown
```

The runner enables model tests and writes `status.txt` with `RUNNING`,
`SUCCEEDED`, or `FAILED` and the final exit code. `install.log` captures all
installation and test output; `tests.xml` is generated when pytest runs.
Inspect progress with `tail -f "$RUN_DIR/install.log"` and
`cat "$RUN_DIR/status.txt"`. A detached login-node process survives shell logout,
but not a node reboot or a site process limit.

`FRAMEWORKS_MODULE` is passed to the staged canonical HydraGNN installer as
well as loaded by the parent. Before installing project wheels, the resulting
environment must match the captured framework Python, Torch version, and Torch
package path. The upstream dependency checkout is not edited.

`RECREATE_MACE_ENV` and `RECREATE_UMA_ENV` independently control compatibility
environment recreation; both default to `0` (reuse), following Perlmutter's
opt-in compatibility-environment lifecycle. `RECREATE_ENV=1` rebuilds only the
base environment; it does not request deletion of UMA or MACE. This allows
rebuilding one model environment without replacing the base environment.
`INSTALL_VLLM=1` verifies Aurora's framework-provided vLLM-XPU stack rather than
installing Perlmutter's CUDA vLLM environment.

The canonical installer and upstream requirements are fetched into a separate
checkout at `.hpc-build/aurora/installer-source`. Its default source is
`https://github.com/ORNL/HydraGNN.git`, branch `main`. The wrapper logs the exact
dependency revision and refreshes that checkout on subsequent installs.
`HYDRAGNN_INSTALLER_REPO`, `HYDRAGNN_INSTALLER_REF`, and `HYDRAGNN_INSTALLER_DIR`
can pin an audited dependency revision or change its location. An explicit
`HYDRAGNN_INSTALLER` overrides the installer script, not the dependency profile.
The selected feature branch is never switched implicitly.

## Module Stacks and Job Catalog

Aurora jobs source `aurora-module-stack.sh` instead of loading modules inline:
`load_aurora_frameworks_modules` (the ALCF `frameworks` stack the matsim venvs
build on), `load_aurora_xpu_modules` (oneAPI + GPU UMD + PTI for the qualified
MLIP XPU path), and `load_aurora_oneapi_modules` (native builds such as VASP).
Jobs that run vLLM source `launchers/aurora-vllm.sh`, which launches the server
through `mpiexec` with the same environment as the vLLM smoke test.

The following entry points match their Frontier and Perlmutter counterparts.
Submit with your allocation (`qsub -A <project> ...`); none embed an account.

| Script | Purpose |
| --- | --- |
| `launchers/_hydragnn-train-step-aurora.sh` | AL retraining step: one built-in HydraGNN fine-tuner on one XPU tile (`--device xpu`) |
| `jobs/job-active-learning-paper-cases-aurora.sh` | Full AL loop for one paper case (`CASE=hea_bcc\|lifepo4\|phosphorene\|phosphorene_qe`) using `examples/paper_cases/al_<case>_aurora.yaml` |
| `jobs/submit-active-learning-paper-cases.sh` | Submit the paper-case job for every case |
| `jobs/job-al-dft-scaling-aurora.sh` | One DFT strong-scaling point; node count from `PBS_NODEFILE` |
| `jobs/submit-al-dft-scaling-sweep.sh` | Submit the scaling sweep over `NODES` x `REPS` |
| `jobs/job-llm-check-aurora.sh` | Live vLLM server + six-stage `matsim-agents llm-check` qualification |
| `jobs/job-sequential-benchmark-aurora.sh` | Per-model vLLM benchmark with merged leaderboard and plot |
| `launchers/launch-test-all-models-aurora.sh` | Submit the single-node vLLM smoke test once per local model |
| `launchers/launch-test-singlenode-resume-aurora.sh` | Resume the single-node sweep for a subset (`RESUME_MODELS`) |
| `launchers/launch-test-multinode-aurora.sh` | Submit `job-serve-multinode-vllm-aurora.sh` with `SERVE_SMOKE=1` for multi-node models |
| `smoke-tests/smoke-transformers-aurora.sh` | HuggingFace provider smoke test across the 12 PVC tiles |

These scripts are syntax-checked and their argument wiring is tested off-system;
they have not yet been run on Aurora.

## Single-Node XPU Qualification

To check the framework Torch build and PyG operations without model checkpoints,
submit the focused smoke test:

```bash
qsub -A <active-allocation> -v PROJECT_ROOT="$PWD" \
  deployments/aurora/jobs/job-qualify-torch-pyg-xpu-aurora.sh
```

It runs the shared `deployments/common/qualify_torch_pyg_device.py` suite, which
checks XPU arithmetic, PyG scatter and graph pooling, gradients, and the
`torch_scatter` add/sum/mean APIs used by HydraGNN's MACE path. Run it on a
compute node; login-node XPU visibility is not a valid qualification. The same
suite is available on Frontier and Perlmutter through their matching
`job-qualify-torch-pyg-*.sh` launchers. It does not qualify optional
CPU-compiled custom `torch_scatter` max/min/mul kernels. The JSON report is
saved to `.hpc-build/aurora/qualification/<job-id>/torch-pyg.json` by default;
set `QUALIFICATION_OUTPUT` to choose another location.

From the repository root, submit a bounded checkpoint-inference job:

```bash
qsub -A <active-allocation> -v PROJECT_ROOT="$PWD" \
  deployments/aurora/jobs/job-qualify-mlip-xpu-aurora.sh
```

The job uses one node and one XPU tile, launching each model through `mpiexec`
in its own environment. It loads the framework runtime modules directly rather
than activating conda. Each backend has an eight-minute limit and independent
logs/results under `.hpc-build/aurora/qualification/<job-id>/`. A pass requires
working XPU arithmetic, model parameters resident on XPU, and finite energy and
forces for perturbed two-atom silicon. CPU fallback is a failure. This smoke
test does not establish scientific accuracy or production scaling.

Prepare checkpoint caches on a login node before submission; compute nodes run
offline. The defaults are the local HydraGNN BEST6 logdir/BranchWeightMLP,
UMA `uma-s-1p1`, and MACE-MP `small`. `MATSIM_HYDRAGNN_LOGDIR`,
`HYDRAGNN_BRANCH_MLP_CHECKPOINT`, `MATSIM_UMA_MODEL_NAME`, `MATSIM_MACE_MODEL`,
and `QUALIFICATION_RUN_DIR` override inputs/output. The model configuration
accepts explicit `xpu` devices; actual support is determined by this test, not
by configuration acceptance alone.

Each model writes a timed `.progress.json` before imports, XPU setup, checkpoint
loading, energy/force prediction, synchronization, and residency checks. Logs
include Python stack dumps every 90 seconds and on termination. Launcher
timeouts also produce result JSON, including the last recorded stage; the
eight-minute limit has a 30-second forced-cleanup grace period. UMA residency
is checked after prediction because FairChem initializes device placement lazily.

## Model Download Safety Policy

See the canonical cross-platform policy in `docs/model-download-safety.md`.

Download entry points:

- `deployments/aurora/download/download-models-aurora.sh` (full set, gated models optional via token)
- `deployments/aurora/download/download-open-models-aurora.sh` (open-access set)

## vLLM-XPU

### install-vllm-xpu-aurora.sh

Verifies that the `frameworks` module ships a working vLLM-XPU stack (no
source build required as of `frameworks/2025.3.1`).  Run on a login node; XPU
will report unavailable there — that is expected.

```bash
bash deployments/aurora/setup/install-vllm-xpu-aurora.sh
```

After verification, submit the smoke test:

```bash
qsub deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh
```

**Important Aurora-specific pitfalls** — see the full guide at
`docs/vllm-aurora.md` for details on every challenge and fix, including:

- SIGSEGV in vLLM's model-registry subprocess (node-specific, fixed via
  `aurora_vllm_entrypoint.py`)
- `ONEAPI_DEVICE_SELECTOR` syntax differences between node driver versions
  (`opencl:cpu` is the canonical cross-node form)
- Must launch the server via `mpiexec -n 1 --ppn 1` for PALS permissions
- numpy must **not** be pinned to 1.x on Aurora (frameworks ships 2.2.6, 2.x ABI)

---


Runs a two-stage install:

1. Run HydraGNN's Aurora installer from
   `scripts/hpc/alcf/aurora/installation/install.sh` in the HydraGNN checkout to
	create/configure the environment and install HydraGNN dependencies.
2. Install matsim-agents and additional runtime dependencies.

Usage:

```bash
bash deployments/aurora/setup/install.sh
```

Common overrides:

```bash
# Choose env location
VENV_PATH=/path/to/aurora_venv bash deployments/aurora/setup/install.sh

# Choose Python interpreter
PYTHON_BIN=python3.11 bash deployments/aurora/setup/install.sh

# Install matsim-agents with extra backends
MATSIM_EXTRAS="huggingface,dev" bash deployments/aurora/setup/install.sh

# Request a clean rebuild
RECREATE_ENV=1 bash deployments/aurora/setup/install.sh
```

> **UMA / FairChem:** use `INSTALL_UMA=1 bash deployments/aurora/setup/install.sh`.
> This installs `fairchem-core` in the matsim-owned `.venv-uma` environment and
> verifies its calculator API. FairChem does not currently document Intel XPU support, so a
> successful import is not evidence of accelerated UMA inference on Aurora.

### Experimental FairChem Compatibility

Aurora's Torch `2.13.0a0+gitcf30153` does not satisfy upstream FairChem's stable
`torch~=2.13.0` metadata. The UMA installer downloads FairChem 2.23.0 and creates
an Aurora-only wheel with an explicit compatibility build tag. Its Torch
requirement is changed to the exact framework version. The installer additionally enables
an experimental FairChem 2.23.0 XPU device-initialization adapter, identified by
the `2auroraxpu` wheel build tag. Other FairChem versions are rejected for this
adapter. All other dependencies are resolved normally; after resolution, only
the selected FairChem wheel is force-reinstalled to replace any same-version
older compatibility build.

For an explicit XPU request, the adapter requires a visible XPU, selects that
device instead of rejecting it, and uses general uncompiled inference with
TF32 and MOLE merging disabled. It does not modify model weights or tensor
operations, and it does not permit CPU fallback. CPU/CUDA setup is unchanged.
This is an XPU compatibility experiment, not evidence that checkpoint inference
has passed on Intel hardware.

The upstream wheel, patched wheel, and JSON provenance with SHA-256 hashes are
retained under `.hpc-build/aurora/fairchem/<version>/`. This override is
experimental, not an upstream claim of XPU support. The installer verifies the
runtime dependency tree, imports, and the framework Torch path.

On October 13, 2026, FairChem 2.23.0 passed those checks with the framework Torch
above. UMA `uma-s-1p1` also loaded and evaluated a two-atom diamond silicon cell
on CPU, producing finite energy and finite `(2, 3)` forces. This smoke test does
not establish scientific accuracy, training compatibility, or Intel XPU
execution. XPU qualification still requires a compute-node test. Checkpoint
access may require accepting the upstream model license and Hugging Face
authentication; never put tokens in source files or logs.

> **MACE:** use `INSTALL_MACE=1 bash deployments/aurora/setup/install.sh`.
> Upstream MACE requires `e3nn==0.4.4`, so the installer creates
> `$MATSIM_DIR/.venv-mace` rather than changing HydraGNN's `e3nn==0.5.1` in
> `.venv`. The compatibility environment inherits the Aurora PyTorch/XPU stack
> and applies the site-specific h5py repair. XPU execution still requires an
> on-node inference qualification.

The Aurora MACE installer adds a backend guard to the optional OpenEquivariance
imports in `.venv-mace` only. MACE 0.3.16 otherwise imports inherited
OpenEquivariance even when acceleration is disabled; that package asserts that
Torch has CUDA or HIP and crashes on the XPU build. The guard treats it as
unavailable when both `torch.version.cuda` and `torch.version.hip` are absent,
allowing MACE's existing e3nn fallback. CUDA/HIP builds retain the optional
import, and unrelated import/runtime failures are not suppressed. Neither the
base environment nor OpenEquivariance itself is modified.

MACE 0.3.16's calculator imports and a real e3nn tensor-product CPU smoke test
passed with this guard on October 13, 2026. This does not qualify checkpoint
inference, scientific accuracy, or Intel XPU execution.

Default environment path created by this flow:

```bash
$MATSIM_DIR/.venv
```

### setup_matsim_aurora.sh
Quick setup script for daily use after installation.

Usage:

```bash
source deployments/aurora/setup/setup_matsim_aurora.sh
```

Override environment path:

```bash
MATSIM_AURORA_VENV=/path/to/venv source deployments/aurora/setup/setup_matsim_aurora.sh
```

### build-qe-cpu-aurora.sh
Builds Quantum ESPRESSO on Aurora (CPU-focused), mirroring the workflow used by
the Frontier/Perlmutter QE build scripts.

Usage:

```bash
bash deployments/aurora/setup/build-qe-cpu-aurora.sh
```

Common overrides:

```bash
QE_VERSION=develop bash deployments/aurora/setup/build-qe-cpu-aurora.sh
QE_PREFIX=/path/to/quantum-espresso bash deployments/aurora/setup/build-qe-cpu-aurora.sh
NCORES=32 bash deployments/aurora/setup/build-qe-cpu-aurora.sh
```

### build-qe-gpu-aurora.sh
Builds Quantum ESPRESSO on Aurora with Intel GPU offload intent, using the same
clone/configure/build/install flow as the other machine scripts.

Usage:

```bash
bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

Common overrides:

```bash
QE_GPU_ARCHS=intel_gpu_pvc bash deployments/aurora/setup/build-qe-gpu-aurora.sh
QE_GPU="openmp;oneapi" bash deployments/aurora/setup/build-qe-gpu-aurora.sh
EXTRA_CMAKE_ARGS="-DVAR=VALUE" bash deployments/aurora/setup/build-qe-gpu-aurora.sh
```

Validated results from the current repository run:

- build/install completed successfully (exit code 0)
- install prefix populated under `external/quantum-espresso/install-gpu/`
- 106 executables installed in `install-gpu/bin/`
- core binaries present: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `epw.x`

See full Aurora QE documentation:

- `docs/quantum-espresso-aurora.md`

Run `pw.x` with the Aurora launcher after building:

```bash
bash deployments/aurora/launchers/run-pw-gpu-aurora.sh path/to/pw.in
```

### VASP 6.6.1 on Aurora

The repository does not commit VASP source because it is proprietary, but it
does track the build helper and makefile configuration for the Aurora GPU build.

- Provenance note: `docs/vasp-aurora.md`
- Build script: `deployments/aurora/setup/build-vasp-gpu-aurora.sh`
- Tracked recipe: `deployments/aurora/setup/makefile.include.aurora-gpu`
- Upstream template used: `arch/makefile.include.oneapi_omp_off`
- Local working file during the build: `external/vasp6/src/vasp.6.6.1/makefile.include`

This keeps the exact makefile lineage discoverable without checking any VASP
source or vendor archive into git. The helper installs the tracked recipe before
building; `MAKEFILE_INCLUDE` selects a custom recipe. It loads
`ONEAPI_MODULE` (default `oneapi/release/2026.1.0`) and `GPU_UMD_MODULE`
(default `intel_gpu_umd_aicoe`). The default target builds `vasp_std`, `vasp_gam`,
and `vasp_ncl` in one invocation.
