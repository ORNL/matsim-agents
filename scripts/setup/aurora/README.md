# Aurora Setup Scripts for matsim-agents

This directory provides an Aurora-focused setup flow that mirrors the phased
install style used by the Frontier and Perlmutter scripts.

## Script

### install_matsim_aurora.sh
Creates a Python virtual environment and performs a two-stage install:

1. Install HydraGNN dependencies first (base/torch/pyg + editable HydraGNN).
2. Install matsim-agents and additional runtime dependencies.

Usage:

```bash
bash scripts/setup/aurora/install_matsim_aurora.sh
```

Common overrides:

```bash
# Choose env location
VENV_PATH=/path/to/aurora_venv bash scripts/setup/aurora/install_matsim_aurora.sh

# Choose Python interpreter
PYTHON_BIN=python3.11 bash scripts/setup/aurora/install_matsim_aurora.sh

# Install matsim-agents with extra backends
LLM_BACKENDS="huggingface,dev" bash scripts/setup/aurora/install_matsim_aurora.sh

# Also install vLLM
INSTALL_VLLM_SERVER=1 bash scripts/setup/aurora/install_matsim_aurora.sh
```

### setup_matsim_aurora.sh
Quick setup script for daily use after installation.

Usage:

```bash
source scripts/setup/aurora/setup_matsim_aurora.sh
```

Override environment path:

```bash
MATSIM_AURORA_VENV=/path/to/venv source scripts/setup/aurora/setup_matsim_aurora.sh
```
