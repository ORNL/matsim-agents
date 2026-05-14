# Frontier Setup Scripts for matsim-agents

This directory provides both full-install and quick-setup scripts for Frontier.

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
