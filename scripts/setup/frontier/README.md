# Frontier Setup Scripts for matsim-agents

This directory provides both full-install and quick-setup scripts for Frontier.

## Model Download Safety Policy

See the canonical cross-platform policy in `docs/model-download-safety.md`.

Download entry points:

- `scripts/download/frontier/download-models-frontier.sh` (full set, gated models optional via token)
- `scripts/download/frontier/download-open-models-frontier.sh` (open-access set)

## Scripts

### install_matsim_frontier.sh
Frontier phased install that builds the HydraGNN runtime environment first, then
installs matsim-agents and supporting dependencies.

Usage:

```bash
bash scripts/setup/frontier/install_matsim_frontier.sh
# ROCm 7.2 variant
bash scripts/setup/frontier/install_matsim_frontier.sh --rocm72
```

### UMA MLIP Backend (fairchem-core)

`fairchem-core >= 2.0` requires `numpy >= 2.0`, which conflicts with HydraGNN's
`numpy == 1.26.4` pin. **Do not install `fairchem-core` into `hydragnn_venv`.**
Use `INSTALL_UMA=1` to create a separate `fairchem_venv` alongside `hydragnn_venv`:

```bash
INSTALL_UMA=1 bash scripts/setup/frontier/install_matsim_frontier.sh
```

UMA jobs must activate `fairchem_venv` instead of `hydragnn_venv`. This is a
known incompatibility until HydraGNN relaxes its numpy pin.

### setup_matsim_frontier.sh
Quick setup script for daily use after installation.

Usage:

```bash
# Default ROCm 7.1 env
source scripts/setup/frontier/setup_matsim_frontier.sh

# ROCm 7.2 env
source scripts/setup/frontier/setup_matsim_frontier.sh --rocm72
```

Overrides:

```bash
MATSIM_FRONTIER_VENV=/path/to/env source scripts/setup/frontier/setup_matsim_frontier.sh
MATSIM_FRONTIER_ROCM_VERSION=7.2 source scripts/setup/frontier/setup_matsim_frontier.sh
```
