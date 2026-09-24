# Model Download Safety Policy

This policy applies to model download workflows on Perlmutter, Frontier, and Aurora.

## Rules

- Use only HydraGNN-managed runtime environments for model downloads.
- Do not run ad-hoc pip installs in those runtime environments.
- Download scripts must not mutate package versions at runtime.
- Download scripts must fail fast if they detect an unsupported huggingface_hub major version that can destabilize dependency constraints.
- If tooling is missing, refresh/rebuild the platform environment with the platform installer instead of patching packages in place.

## Why This Exists

HydraGNN runtime environments carry pinned dependency constraints that are needed for stable HPC operation. Ad-hoc package upgrades can silently overwrite sensitive packages and break unrelated workflows.

## Supported Download Entry Points

### Perlmutter

- deployments/perlmutter/download/download-models-perlmutter.sh
- deployments/perlmutter/download/download-open-models-perlmutter.sh

### Frontier

- deployments/frontier/download/download-models-frontier.sh
- deployments/frontier/download/download-open-models-frontier.sh

### Aurora

- deployments/aurora/download/download-models-aurora.sh
- deployments/aurora/download/download-open-models-aurora.sh

## Recovery Guidance

If a platform download script reports missing or incompatible tooling, rebuild with the platform installer flow:

- Perlmutter: `deployments/perlmutter/setup/install.sh`
- Frontier: `deployments/frontier/setup/install.sh`
- Aurora: `deployments/aurora/setup/install.sh`
