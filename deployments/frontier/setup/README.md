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

`fairchem-core >= 2.0` requires `numpy >= 2.0`, which conflicts with HydraGNN's
`numpy == 1.26.4` pin. **Do not install `fairchem-core` into `hydragnn_venv`.**
UMA jobs must use a separate environment. The canonical HydraGNN installer does
not silently create a second environment.

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
