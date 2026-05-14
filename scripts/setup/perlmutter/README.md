# Perlmutter Setup Scripts for matsim-agents

This directory contains setup scripts for running matsim-agents on NERSC's Perlmutter supercomputer.

This implementation mirrors the methodology used for Frontier, providing both quick-setup and full-installation options.

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
- CUDA toolkit (12.4)
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
/global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
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

**Environment path (default):**
```
$HYDRAGNN_DIR/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
```
Override with `VENV_PATH=/custom/path` when needed.

Quick setup (`setup_matsim_perlmutter.sh`) prefers this shared path, but will
fall back to legacy local `matsim-agents/perlmutter_venv` if present.
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

## Requirements

### Quick Setup (`setup_matsim_perlmutter.sh`)
- Running on a Perlmutter login or compute node
- Existing HydraGNN environment at:
  ```
  /global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
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
/global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
```

Contact the project maintainers if it needs to be re-created.

### "conda env not found" (Fresh Installation)
The environment was not created. Re-run the installation:
```bash
bash install_matsim_perlmutter.sh --gpu
```
Default expected path:
```
/global/cfs/projectdirs/amsc001/cm2us/mlupopa/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
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
- **CUDA Version:** 12.4 (matches PyTorch cu124 wheels)

### Important Environment Variables
- `TORCH_CUDA_ARCH_LIST`: Set to `8.0` (A100 compute capability)
- `MAX_JOBS`: Parallel build jobs (default: 16)
- `TORCH_CUDA_TAG`: PyTorch CUDA version tag (default: cu124)

### Module Stack Details
- **Cray Programming Environment:** cpe/24.07
- **PrgEnv-gnu/8.5.0:** GNU compiler suite
- **gcc-native/13.2:** Modern GCC for C++ extensions
- **cudatoolkit/12.4:** NVIDIA CUDA (matches PyTorch wheels)
- **Miniforge3:** Conda package manager

---

## Additional Resources

- [HydraGNN Repository](https://github.com/ORNL/HydraGNN)
- [matsim-agents Repository](https://github.com/ORNL/matsim-agents)
- [NERSC Perlmutter Documentation](https://docs.nersc.gov/systems/perlmutter/)
- [PyTorch CUDA Wheels](https://download.pytorch.org/whl/)
