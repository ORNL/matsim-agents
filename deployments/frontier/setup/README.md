# Frontier Setup Scripts for matsim-agents

The single full-install entry point for Frontier is:

```bash
bash deployments/frontier/setup/install.sh
```

It clones/updates `ORNL/HydraGNN`, runs HydraGNN's current ROCm 7.2 installer,
then installs and verifies matsim-agents in the resulting environment. There is
no second general-purpose installer or legacy alias.

## GPU Primitive Qualification

Submit the shared Torch/PyG device smoke test from a Frontier allocation:

```bash
PROJECT_ROOT="$PWD" sbatch -A <allocation> \
  deployments/frontier/jobs/job-qualify-torch-pyg-frontier.sh
```

It exercises Torch arithmetic, PyG scatter/pooling and gradients, and
`torch_scatter` add/sum/mean on ROCm GPUs. This is a backend primitive check,
not checkpoint-backed model qualification; the compute-qualified portability
workflow provides a separate model-level tier. The report is saved to
`.hpc-build/frontier/qualification/<job-id>/torch-pyg.json` by default; set
`QUALIFICATION_OUTPUT` to choose another location.

## Quantum ESPRESSO GPU Smoke Test

Build and install the QE GPU executables, then submit a small Si SCF calculation
with OpenMP offload required (so it fails rather than silently running on CPU):

```bash
bash deployments/frontier/setup/build-qe-gpu-frontier.sh
sbatch deployments/frontier/jobs/job-qe-gpu-smoke-frontier.sh
```

The smoke job requires the installed `external/quantum-espresso/install-gpu/bin/pw.x`
and uses the bundled `Si_r.upf` pseudopotential. Results are written under
`.hpc-build/frontier/qe-smoke/<job-id>/`.

## Model Download Safety Policy

See the canonical cross-platform policy in `docs/model-download-safety.md`.

Download entry points:

- `deployments/frontier/download/download-models-frontier.sh` (full set, gated models optional via token)
- `deployments/frontier/download/download-open-models-frontier.sh` (open-access set)

## Scripts

### install.sh
Frontier phased install that builds the HydraGNN runtime environment first, then
installs matsim-agents and supporting dependencies.

Usage:

```bash
bash deployments/frontier/setup/install.sh
```

### install-vllm-rocm72.sh

Optional isolated vLLM deployment workflow. Run `install.sh` first; HydraGNN's
official installer creates the Conda environment at `.venv` (despite the path
name). This script creates a separate standard Python `venv` at `.venv-vllm`,
installs a matching ROCm 7.2 PyTorch runtime and the pinned vLLM/Ray/Triton
dependencies there, then submits `build-vllm-rocm72.sh` for the GPU build. It
leaves the HydraGNN Conda environment untouched. The server launcher uses
`.venv-vllm` by default; override it with `MATSIM_VLLM_VENV`. The build is
submitted with Slurm account `lrn110` by default; set `SLURM_ACCOUNT` to another
account associated with your user if needed.

```bash
bash deployments/frontier/setup/install.sh
bash deployments/frontier/setup/install-vllm-rocm72.sh
```

### UMA MLIP Backend (fairchem-core)

HydraGNN main and matsim-agents use NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.14,
torchvision 0.29, and e3nn 0.5.1. FairChem requires Torch 2.13, so the
matsim-owned `.venv-uma` installs the ROCm 7.2 Torch 2.13 wheel separately:

```bash
INSTALL_UMA=1 bash deployments/frontier/setup/install.sh
```

The installer verifies FairChem imports and rejects a CUDA or non-ROCm Torch
installation. UMA model inference on Frontier remains unqualified until it
passes a model inference smoke test on a Frontier GPU.

### MACE compatibility environment

Current `mace-torch==0.3.16` requires `e3nn==0.4.4`, which conflicts with
HydraGNN's `e3nn==0.5.1`. Install and verify the matsim-owned compatibility
environment from the same entry point:

```bash
INSTALL_MACE=1 bash deployments/frontier/setup/install.sh
source .venv-mace/bin/activate
```

`.venv-mace` inherits the ROCm PyTorch stack from `.venv` but has its own e3nn
and MACE packages. HydraGNN and MACE must run as separate Python processes.

### setup_matsim_frontier.sh
Quick setup script for daily use after installation.

Usage:

```bash
# Default ROCm 7.1 env
source deployments/frontier/setup/setup_matsim_frontier.sh

# ROCm 7.2 env
source deployments/frontier/setup/setup_matsim_frontier.sh --rocm72
```

Overrides:

```bash
MATSIM_FRONTIER_VENV=/path/to/env source deployments/frontier/setup/setup_matsim_frontier.sh
MATSIM_FRONTIER_ROCM_VERSION=7.2 source deployments/frontier/setup/setup_matsim_frontier.sh
```
