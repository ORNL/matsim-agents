# Perlmutter Setup Scripts for matsim-agents

This directory contains setup scripts for running matsim-agents on NERSC's Perlmutter supercomputer.

This implementation mirrors the methodology used for Frontier, providing both quick-setup and full-installation options.

## Model Download Safety Policy

See the canonical cross-platform policy in `docs/model-download-safety.md`.

Perlmutter download entry points:

- `scripts/download/perlmutter/download-models-perlmutter.sh` (full set, gated models optional via token)
- `scripts/download/perlmutter/download-open-models-perlmutter.sh` (open-access set)

## Scripts

### Installation Approach

You have two options:

#### Option 1: Quick Setup (Activate Installer-Created Environment)
Use the **quick setup** scripts to activate the HydraGNN environment created by the installer.

**Best for:** Development, testing, quick runs when HydraGNN environment is stable

```bash
source setup_matsim_perlmutter.sh [--gpu]
```

#### Option 2: Fresh Installation (Always Recreates Environment)
Use the **full installation** script to recreate the HydraGNN-aligned conda environment.

**Best for:** Reproducible installs aligned with HydraGNN runtime expectations

```bash
bash install_matsim_perlmutter.sh [--gpu]
```

---

## Script Details

### `perlmutter-module-stack.sh`
Loads the necessary Perlmutter module stack for CPU and GPU computing.

**Functions:**
- `load_perlmutter_modules()` - Load standard Perlmutter CUDA stack (CPU)
- `load_perlmutter_modules_gpu()` - Load modules for GPU (A100) compute nodes

**Module Stack Includes:**
- Cray Programming Environment (cpe/24.07)
- GNU compiler suite (PrgEnv-gnu/8.5.0)
- Cray MPI (cray-mpich/8.1.30)
- CUDA toolkit (12.9)
- Modern GCC compiler (gcc-native/13.2)
- CMake, Conda, and build tools

### `setup_matsim_perlmutter.sh` (Quick Setup)
Quickly activates a pre-configured HydraGNN environment:
1. Loads Perlmutter modules
2. Activates the shared HydraGNN conda environment
3. Sets up PYTHONPATH for matsim-agents

**Usage:**
```bash
# CPU-only
source setup_matsim_perlmutter.sh

# With GPU support
source setup_matsim_perlmutter.sh --gpu
```

**Environment used:** 
```
$HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
```

### `install_matsim_perlmutter.sh` (Fresh Installation)
Frontier-style phased installation that always recreates the Perlmutter HydraGNN environment used at runtime:

**Phase 1:** Run HydraGNN Perlmutter installer (delegated build of CUDA/PyTorch/PyG stack)
**Phase 2:** Activate resulting env and install matsim-agents + runtime extras

**Usage:**

First-time full installation (creates new environment):
```bash
bash install_matsim_perlmutter.sh [--gpu]
```

**Flags:**
- `--gpu` - Compatibility flag (installer already targets A100/CUDA)

**Customization via environment variables:**
```bash
# Custom environment location
VENV_PATH=/custom/path bash install_matsim_perlmutter.sh --gpu

# Runtime setup override (quick setup script)
MATSIM_PERLMUTTER_VENV=/custom/path source setup_matsim_perlmutter.sh --gpu

# Python version
PYTHON_VERSION=3.12 bash install_matsim_perlmutter.sh --gpu

# Parallel build jobs
MAX_JOBS=32 bash install_matsim_perlmutter.sh --gpu

# Also install vLLM server package
INSTALL_VLLM_SERVER=1 bash install_matsim_perlmutter.sh --gpu
```

**Advanced module/path overrides (forwarded to the delegated HydraGNN installer):**
```bash
# Override where module-init scripts are sourced from
MODULES_SH_PATH=/etc/profile.d/modules.sh \
LMOD_INIT_BASH_PATH=/usr/share/lmod/lmod/init/bash \
MODULES_INIT_BASH_PATH=/usr/share/Modules/init/bash \
bash install_matsim_perlmutter.sh --gpu

# Override Perlmutter module names/versions used by HydraGNN installer
PERLMUTTER_CPE_MODULE=cpe/24.07 \
PERLMUTTER_PRGENV_MODULE=PrgEnv-gnu/8.5.0 \
PERLMUTTER_MPICH_MODULE=cray-mpich/8.1.30 \
PERLMUTTER_ACCEL_MODULE=craype-accel-nvidia80 \
PERLMUTTER_GCC_MODULE=gcc-native/13.2 \
PERLMUTTER_CONDA_PRIMARY_MODULE=conda/Miniforge3-24.11.3-0 \
bash install_matsim_perlmutter.sh --gpu
```

These overrides are optional and mainly useful when site module paths or
module names differ from the defaults.

**Packages installed by the script (high level):**
- Core HydraGNN + PyTorch/PyG stack (via delegated HydraGNN installer)
- matsim-agents (editable)
- Core runtime/test dependencies (`langchain-core`, `pytest`, `pytest-cov`)
- HydraGNN runtime dependencies (`scikit-learn==1.5.1`, `vesin==0.4.2`)
- `huggingface_hub` (includes `hf` CLI for resumable model downloads)
- `transformers` + `accelerate`
- Optional: `vllm` server package when `INSTALL_VLLM_SERVER=1`

**Install root + environment path (default):**
```
INSTALL_ROOT = $HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter
VENV_PATH    = $INSTALL_ROOT/hydragnn_venv
```
`INSTALL_ROOT` is the single source of truth: the conda env AND all HydraGNN
dependency build trees (ADIOS2, MPI4PY, DDStore, GPTL, DeepHyper, PyG) live
inside it. Override `INSTALL_ROOT=/custom/path` to relocate everything together.
`VENV_PATH` defaults to `$INSTALL_ROOT/hydragnn_venv`; override it alone only to
place the env elsewhere (build deps still stay under `INSTALL_ROOT`).

Quick setup (`setup_matsim_perlmutter.sh`) prefers this shared in-HydraGNN path,
falling back to the legacy local `matsim-agents/perlmutter_venv` only if the
shared env is absent.
### `job_perlmutter.sh`
Example SLURM job submission script with proper environment setup.

**Usage:**
```bash
# Edit to set your project allocation
vim job_perlmutter.sh

# Submit
sbatch job_perlmutter.sh
```

### `test_matsim_perlmutter.sh`
Comprehensive test suite validating the complete environment:
- Module loading
- Python environment verification
- PyTorch + CUDA availability
- Package imports (HydraGNN, matsim-agents)
- GPU information

**Usage:**
```bash
bash test_matsim_perlmutter.sh
```

---

## Runtime launchers and smoke tests

Once the environment is installed, the following script directories provide
ready-to-submit Slurm jobs that mirror the Frontier set:

### `scripts/launchers/perlmutter/`
| Script | Purpose |
|---|---|
| `run-pw-gpu-perlmutter.sh` | QE `pw.x` GPU launcher used by `MATSIM_QE_LAUNCHER`. Loads NVHPC 25.5 (CUDA 12.9, matches the PyTorch wheel) inside its own subshell, `srun`s with `--gpus-per-node=4 --gpu-bind=closest`, 4 ranks × 16 OMP threads. |
| `run-qe-warmstart-benchmark-perlmutter.sh` | SBATCH wrapper for `tests/integration/test_qe_warmstart.py` (HydraGNN warm-start vs `pw.x` cold-start). Exports the `MATSIM_QE_*` and `MATSIM_HYDRAGNN_*` env vars; restrict fixtures via `FIXTURES=Si_diamond,...`. |
| `launch-test-all-models-perlmutter.sh` | Sequentially submits one single-node smoke job per local HF model under `$PROJ/models/`. Skips models without a local directory; retries on QOS-limit errors. |
| `launch-test-multinode-perlmutter.sh` | 2-node TP smoke test for the largest models (Qwen2.5-72B, Mixtral-8x22B). |
| `launch-test-singlenode-resume-perlmutter.sh` | Resumes a partial single-node sweep. Optional `RESUME_AFTER_JOBID=<jid>` blocks until the in-flight job clears; `RESUME_MODELS="A,B"` whitelists. |
| `run-vasp-gpu-perlmutter.sh` | VASP GPU launcher used by `MATSIM_VASP_LAUNCHER`. Sources `perlmutter-module-stack.sh` (`load_perlmutter_modules_nvidia`), enables `MPICH_GPU_SUPPORT_ENABLED=1`, `srun`s with `--gpus-per-node=4 --gpu-bind=closest`, defaults to 4 ranks × 16 OMP threads. Reads `VASP_VARIANT=std\|gam\|ncl`, `VASP_BIN`, `NRANKS`, `OMP_NUM_THREADS`, `GPUS_PER_NODE`. Multi-node-ready (works in any `salloc`/`sbatch` allocation, adds `-N1` only when run outside Slurm). |

### `scripts/smoke-tests/perlmutter/`
| Script | Purpose |
|---|---|
| `smoke-transformers-perlmutter.sh` | Single-node HuggingFace smoke (uses `matsim_agents.llm.get_chat_model`, provider `huggingface`, `device_map="auto"` over the 4 A100s). |
| `smoke-transformers-multinode-perlmutter.sh` | Multi-node HuggingFace smoke. Pure `srun + torch.distributed` (no `torchrun` nesting, no DeepSpeed). Uses `transformers`' `tp_plan="auto"` tensor-parallel sharding over NCCL on Slingshot. |
| `_torchrun_smoke_loader.py` | Companion loader: reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, calls `dist.init_process_group("nccl")`, loads the model with `tp_plan="auto"` (≥ 2 ranks) or `device_map="auto"` (1 rank), runs a one-shot generate and prints from rank 0. |

### `scripts/advanced/perlmutter/`
| Script | Purpose |
|---|---|
| `job-discovery-chat-perlmutter.sh` | End-to-end discovery validation: **Phase A** runs `matsim-agents chat` with the HF provider against Qwen2.5-72B + HydraGNN MLFF (FIRE relaxation, 64+ atoms, 2 orderings). **Phase B** runs the QE warm-start `pytest` with the cu129-aligned `pw.x`. Toggle phases via `SKIP_LLM=1` / `SKIP_QE=1`. |
| `job-single-relaxation-perlmutter.sh` | Runs `examples/single_relaxation.py` (HydraGNN FIRE relaxation) under a single A100 node; exports `MATSIM_HYDRAGNN_*` and `MATSIM_LLM_*` for the vLLM/HF backend. |
| `job-active-learning-uq-perlmutter.sh` | Full active-learning loop (`matsim-agents al run`) on Perlmutter: MD sampling → ensemble uncertainty → DFT labelling (VASP or QE) → HydraGNN retraining. Multi-phase SBATCH with node-level srun steps. |
| `job-qe-warmstart-perlmutter.sh` | QE warm-start benchmark job: exercises the HydraGNN-preconditioned `pw.x` cold-vs-warm convergence test via `tests/integration/test_qe_warmstart.py`. |

### Submission examples
```bash
# Single-node HF smoke (Qwen2.5-72B by default; override via MATSIM_MODEL_DIR)
sbatch scripts/smoke-tests/perlmutter/smoke-transformers-perlmutter.sh

# Multi-node TP smoke (2 nodes × 4 A100s = 8 ranks, NCCL over Slingshot)
sbatch scripts/smoke-tests/perlmutter/smoke-transformers-multinode-perlmutter.sh

# Sweep all local models, one job at a time
nohup bash scripts/launchers/perlmutter/launch-test-all-models-perlmutter.sh \
  > $PROJ/runs/launch-test-all-pm.log 2>&1 &

# QE warmstart benchmark (defaults to FIXTURES=Si_diamond)
sbatch scripts/launchers/perlmutter/run-qe-warmstart-benchmark-perlmutter.sh

# Full discovery validation (LLM + HydraGNN + QE)
sbatch scripts/advanced/perlmutter/job-discovery-chat-perlmutter.sh
```

All these scripts source `perlmutter-module-stack.sh` (`load_perlmutter_modules_gpu`)
and activate the same `hydragnn_venv` produced by `install_matsim_perlmutter.sh`,
so they inherit the unified HydraGNN-aligned toolchain (`cudatoolkit/12.9`,
`gcc-native/13.2`, torch `2.11.0+cu129`).

---

## VASP 6.6.0 GPU build (NVHPC OpenACC, multi-node enabled)

Three scripts together produce a multi-node-capable VASP 6.6.0 GPU binary on
Perlmutter A100 nodes (sm_80) and a runtime launcher consumed by
`matsim_agents.tools.vasp_relax` via the `MATSIM_VASP_LAUNCHER` env var.

### Toolchain
- `PrgEnv-gnu/8.5.0` + `cpe/24.07` + `cray-mpich/8.1.30` (NVIDIA variant) +
  `cudatoolkit/12.9` + `cray-fftw/3.3.10.11` + `cray-libsci/25.09`
  (`libsci_nvidia_mp` for BLAS/LAPACK).
- Compilers: `nvfortran` / `nvc` / `nvc++` from NVHPC 25.5 SDK (PATH override;
  Cray `ftn`/`cc` wrappers are bypassed).
- VASP CPP defines: `-DACC_OFFLOAD -DNVCUDA -DUSENCCL` (NVHPC OpenACC offload).
- scaLAPACK: built locally from netlib 2.2.0 source against the same
  nvfortran + cray-mpich stack. The cray-libsci NVIDIA variant ships no
  scalapack, and the NVHPC-bundled `libscalapack.a` is linked against
  OpenMPI/HPC-X and is ABI-incompatible with cray-mpich.

### Critical compatibility notes
- For nvfortran the cray-mpich `mpi.mod` *must* come from
  `/opt/cray/pe/mpich/8.1.30/ofi/nvidia/23.3/include` — the `gnu/12.3` variant
  is GFortran-built and rejected by nvfortran (`Corrupt or Old Module file`).
  `build-vasp-gpu-perlmutter.sh` auto-detects and overrides `MPICH_DIR`.
- The scalapack build must define `CDEFS=-DAdd_` so the REDIST C wrappers emit
  Fortran-mangled symbols (`pzgemr2d_`, `pcgemr2d_`, `pdgemr2d_`,
  `psgemr2d_`, `pztrmr2d_`, `pdtrmr2d_`); without these VASP fails to link.
- The upstream scalapack Makefile races on `ar cr`, so the build must be
  serial (`make -j 1 lib`).

### Scripts

#### `build-scalapack-perlmutter.sh`
Builds netlib scaLAPACK 2.2.0 against the Perlmutter NVIDIA stack.

**What it does:**
1. Sources `perlmutter-module-stack.sh` and calls `load_perlmutter_modules_nvidia`.
2. Resolves `MPICH_DIR`, `LIBSCI_NVIDIA` (NVIDIA/.../x86_64), and `GTL_DIR`.
3. Writes a netlib-style `SLmake.inc` with `FC=nvfortran`, `CC=nvc`,
   `CDEFS=-DAdd_`, `BLASLIB=-L${LIBSCI_NVIDIA}/lib -lsci_nvidia_mp`.
4. Runs `make -j 1 lib` and installs `libscalapack.a` under `external/scalapack/install/lib/`.

**Usage:**
```bash
bash scripts/setup/perlmutter/build-scalapack-perlmutter.sh
# Output: external/scalapack/install/lib/libscalapack.a
```

#### `build-vasp-gpu-perlmutter.sh`
Builds VASP 6.6.0 (`vasp_std`, `vasp_gam`, `vasp_ncl`) with the NVHPC OpenACC
GPU port. Source layout convention:
`${REPO}/external/vasp6/src/vasp.6.6.0/...`.

**What it does:**
1. Loads the NVIDIA-variant module stack (matches the launcher).
2. Auto-builds scaLAPACK by invoking `build-scalapack-perlmutter.sh` if
   `${SCALAPACK_ROOT}/lib/libscalapack.a` is missing (set
   `SCALAPACK_AUTOBUILD=0` to skip; set `SCALAPACK_ROOT=""` to disable
   scalapack entirely and produce a single-node `-DnoSCALAPACK` build).
3. Generates `makefile.include` (controlled by `REGENERATE_MAKEFILE=1`) with
   `FC = nvfortran -mp -acc $(GPU)`, `LLIBS = $(MPI_LIB) -cudalib=cublas,cusolver,cufft,nccl -cuda`,
   `BLAS = -L${libsci_nvidia}/lib -lsci_nvidia_mp`, plus the resolved
   scalapack lib + `-DscaLAPACK` define.
4. Runs `make PREFIX=$PREFIX DEPS=1 MODS=1 -j$NCORES <target>` for each variant.

**Usage:**
```bash
# First-time / clean rebuild of all three variants
REGENERATE_MAKEFILE=1 CLEAN_BUILD=1 NCORES=16 \
  bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh

# Build only vasp_std
VASP_TARGET=std bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh

# Single-node build without scalapack (faster, no multi-node parallel diag)
SCALAPACK_ROOT="" REGENERATE_MAKEFILE=1 CLEAN_BUILD=1 \
  bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh
```

**Tunables:** `VASP_ROOT`, `PREFIX=build`, `NCORES`, `CLEAN_BUILD=0|1`,
`VASP_TARGET=all|std|gam|ncl`, `REGENERATE_MAKEFILE=0|1`, `GPU_ARCH=cc80`
(A100; use `cc90` for H100, `cc89` for L40), `CUDA_VER=12.9`,
`SCALAPACK_ROOT`, `SCALAPACK_AUTOBUILD=0|1`.

**Outputs:** `external/vasp6/src/vasp.6.6.0/bin/{vasp_std,vasp_gam,vasp_ncl}`.

**Long builds:** the full preprocess + compile + link pass takes ~25 min per
variant on a Perlmutter login node. Launch under `setsid nohup` so the build
survives login-node disconnects:
```bash
TS=$(date +%Y%m%d-%H%M%S)
LOG="external/vasp6/logs/build-${TS}.log"
REGENERATE_MAKEFILE=1 CLEAN_BUILD=1 NCORES=16 \
  setsid nohup bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh \
  > "$LOG" 2>&1 < /dev/null & disown
```

#### `run-vasp-gpu-perlmutter.sh`
Runtime `MATSIM_VASP_LAUNCHER`. Invoked from the cwd containing
`INCAR/POSCAR/KPOINTS/POTCAR`; no argv is passed. Sources the NVIDIA module
stack, exports `CUDA_HOME`/`LD_LIBRARY_PATH`/`MPICH_GPU_SUPPORT_ENABLED=1`,
and `srun`s the requested variant.

**Usage (in a Slurm allocation):**
```bash
export MATSIM_VASP_LAUNCHER=$PWD/scripts/launchers/perlmutter/run-vasp-gpu-perlmutter.sh
export VASP_VARIANT=std       # or gam / ncl
export NRANKS=4 OMP_NUM_THREADS=16 GPUS_PER_NODE=4
# ...then anything that calls matsim_agents.tools.vasp_relax.run_vasp(...)
```

For multi-node runs increase `NRANKS` to `4 * <num_nodes>` and keep
`GPUS_PER_NODE=4`; the launcher inherits the Slurm allocation node count.

### End-to-end build + sanity check
```bash
# 1. Place the source tree (one-time)
mkdir -p external/vasp6/src external/vasp6/logs
tar -xzf vasp.6.6.0.tar.gz -C external/vasp6/src/

# 2. Build (auto-builds scalapack first time)
REGENERATE_MAKEFILE=1 CLEAN_BUILD=1 NCORES=16 \
  bash scripts/setup/perlmutter/build-vasp-gpu-perlmutter.sh \
  2>&1 | tee external/vasp6/logs/build-$(date +%F).log

# 3. Confirm scalapack REDIST symbols are linked in
nm external/vasp6/src/vasp.6.6.0/bin/vasp_std | grep -E 'pzgemr2d_|pdgemr2d_'

# 4. Smoke test in a 1-node allocation
salloc -N 1 -C gpu -q interactive -t 30 -A <allocation>
export MATSIM_VASP_LAUNCHER=$PWD/scripts/launchers/perlmutter/run-vasp-gpu-perlmutter.sh
cd <work_dir_with_INCAR_POSCAR_KPOINTS_POTCAR>
"$MATSIM_VASP_LAUNCHER"
```

---

## Requirements

### Quick Setup (`setup_matsim_perlmutter.sh`)
- Running on a Perlmutter login or compute node
- Existing HydraGNN environment at:
  ```
  $HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
  ```

### Full Installation (`install_matsim_perlmutter.sh`)
- Running on a Perlmutter login node (compute nodes work but slower)
- Sufficient disk space for conda environment (~10-20 GB)
- Internet access (for PyPI package downloads)
- ~30-45 minutes for first-time build

---

## Comparison: Quick Setup vs Fresh Installation

| Aspect | Quick Setup | Fresh Installation |
|--------|------------|-------------------|
| **Speed** | Minutes (1-2 min) | 30-45 minutes |
| **Disk Usage** | None (shared env) | ~10-20 GB (at configured env path) |
| **Isolation** | Shared unless overridden | Shared by default, isolated if `VENV_PATH` is custom |
| **Reproducibility** | Fast reuse | Rebuildable from script |
| **Best for** | Development, testing | Aligned install/runtime and rebuilds |
| **Env location** | Global (shared default) | Shared default (`VENV_PATH` configurable) |

---

## Workflow Examples

### Development Workflow (Quick Setup)
```bash
# One-time setup
source scripts/setup/perlmutter/setup_matsim_perlmutter.sh --gpu

# Every session
cd matsim-agents
source scripts/setup/perlmutter/setup_matsim_perlmutter.sh --gpu
python -m matsim_agents ...
```

### Production Workflow (Fresh Installation)
```bash
# Initial setup (30-45 min, run once)
bash scripts/setup/perlmutter/install_matsim_perlmutter.sh --gpu

# Every session
source scripts/setup/perlmutter/setup_matsim_perlmutter.sh --gpu
python -m matsim_agents ...

# Rebuild and reinstall in a fresh environment
bash scripts/setup/perlmutter/install_matsim_perlmutter.sh --gpu
```

### SLURM Job Submission (Both Approaches)
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -A <allocation>

# Load environment (choose one)

# Option 1: Quick setup (if HydraGNN env exists)
source scripts/setup/perlmutter/setup_matsim_perlmutter.sh --gpu

# Option 2: Activate a custom env path used during install
# conda activate /path/used/as/VENV_PATH

python -m matsim_agents.run --config config.yaml
```

---

## Troubleshooting

### "module command not found"
You must be running on a Perlmutter login node. The module system is not available on external machines.

### "HydraGNN virtual environment not found" (Quick Setup)
Ensure the shared HydraGNN environment exists at:
```
$HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
```

Contact the project maintainers if it needs to be re-created.

### "conda env not found" (Fresh Installation)
The environment was not created. Re-run the installation:
```bash
bash install_matsim_perlmutter.sh --gpu
```
Default expected path:
```
$HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
```

### CUDA not available
- Ensure you're using GPU nodes (`-C gpu`)
- Use the `--gpu` flag when loading modules:
  ```bash
  source setup_matsim_perlmutter.sh --gpu
  # OR
  bash install_matsim_perlmutter.sh --gpu
  ```

### PyTorch import fails
Check CUDA compatibility:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__)"
```

### "hf command not found"
`install_matsim_perlmutter.sh` now installs `huggingface_hub`, which provides `hf`.
If your environment predates that update, re-run:
```bash
bash install_matsim_perlmutter.sh --gpu
```

### Out of memory during build
If the installation fails with OOM:
```bash
# Reduce parallel jobs
MAX_JOBS=4 bash install_matsim_perlmutter.sh --gpu
```

---

## Perlmutter Specifics

### Compute Hardware
- **A100 GPUs:** Compute Capability SM80 (Ampere)
- **Node Configuration:** NVIDIA A100 with CPU support
- **CUDA Version:** 12.9 (matches PyTorch cu129 wheels)

### Important Environment Variables
- `TORCH_CUDA_ARCH_LIST`: Set to `8.0` (A100 compute capability)
- `MAX_JOBS`: Parallel build jobs (default: 16)
- `TORCH_CUDA_TAG`: PyTorch CUDA version tag (default: cu129)

### Module Stack Details
- **Cray Programming Environment:** cpe/24.07
- **PrgEnv-gnu/8.5.0:** GNU compiler suite
- **gcc-native/13.2:** Modern GCC for C++ extensions
- **cudatoolkit/12.9:** NVIDIA CUDA (matches PyTorch wheels)
- **Miniforge3:** Conda package manager

---

## Additional Resources

- [HydraGNN Repository](https://github.com/ORNL/HydraGNN)
- [matsim-agents Repository](https://github.com/ORNL/matsim-agents)
- [NERSC Perlmutter Documentation](https://docs.nersc.gov/systems/perlmutter/)
- [PyTorch CUDA Wheels](https://download.pytorch.org/whl/)
