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

- scripts/download/perlmutter/download-models-perlmutter.sh
- scripts/download/perlmutter/download-open-models-perlmutter.sh

### Frontier

- scripts/download/frontier/download-models-frontier.sh
- scripts/download/frontier/download-open-models-frontier.sh

### Aurora

- scripts/download/aurora/download-models-aurora.sh
- scripts/download/aurora/download-open-models-aurora.sh

## Recovery Guidance

If a platform download script reports missing or incompatible tooling, rebuild with the platform installer flow:

- Perlmutter: scripts/setup/perlmutter/install_matsim_perlmutter.sh
- Frontier: scripts/setup/frontier/install_matsim_frontier.sh
- Aurora: scripts/setup/aurora/install_matsim_aurora.sh
