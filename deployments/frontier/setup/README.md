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

HydraGNN main and matsim-agents now share the NumPy 2.2.6, SciPy 1.17.1,
PyTorch 2.13, and e3nn 0.5.1 contract required by current FairChem. Request UMA
support in the same environment with:

```bash
INSTALL_UMA=1 bash deployments/frontier/setup/install.sh
```

The installer verifies FairChem imports. UMA on ROCm remains unqualified until
the resulting environment passes a model inference smoke test on Frontier.

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
