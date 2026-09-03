# Frontier Setup Scripts for matsim-agents

The single full-install entry point for Frontier is:

```bash
bash deployments/frontier/setup/install.sh
```

It clones/updates `ORNL/HydraGNN`, runs HydraGNN's current ROCm 7.2 installer,
then installs and verifies matsim-agents in the resulting environment.
`install_matsim_frontier.sh` is only a compatibility alias.

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

### UMA MLIP Backend (fairchem-core)

HydraGNN main and matsim-agents now use NumPy 2.4.6, SciPy 1.17.1,
PyTorch 2.14, torchvision 0.29, and e3nn 0.5.1. FairChem cannot yet share
that environment: FairChem 2.22 requires Torch 2.13. Request the matsim-owned
`.venv-uma` environment with:

```bash
INSTALL_UMA=1 bash deployments/frontier/setup/install.sh
```

The installer verifies FairChem imports. UMA on ROCm remains unqualified until
the resulting environment passes a model inference smoke test on Frontier.

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
