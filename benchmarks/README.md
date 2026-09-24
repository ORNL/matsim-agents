# Benchmarks

Reusable performance and backend-comparison drivers belong here.  Diagnostic
and paper-specific post-processing remains under `scripts/diagnostics/` until
it is promoted into a stable benchmark with documented inputs and outputs.

The [`portability/`](portability/) benchmark is the canonical, fixed scientific
gate shared by Frontier, Aurora, and Perlmutter. It deliberately separates one
science configuration from small facility overlays and complements—rather than
replaces—the paper, scaling, and model-comparison benchmarks.

Validate the tracked Codabench assets with:

```bash
python benchmarks/codabench/validate_bundle.py
```

The public repository deliberately omits test structures and protected DFT
labels. Build the actual upload only after supplying those reviewed trees:

```bash
python benchmarks/codabench/build_bundle.py \
  --public-data /secure/export/public_data \
  --reference-data /secure/export/reference_data \
  --output dist/matsim-codabench.zip
```

The builder validates required assets and copies the baseline and submission
packaging utilities into the participant starting kit.

For facility baseline runs, install optional model stacks through the machine's
canonical installer. For example, on Aurora:

```bash
INSTALL_UMA=1 INSTALL_MACE=1 bash deployments/aurora/setup/install.sh
```

UMA remains in `.venv`. MACE runs from `.venv-mace` because the current
upstream package pins `e3nn==0.4.4`, conflicting with HydraGNN's exact
`e3nn==0.5.1` pin. The older Codabench MACE helper remains only as a forwarding
compatibility entry point and no longer modifies the active HydraGNN environment.

Codabench dependencies follow the same boundary: `requirements.txt` is
backend-neutral, `requirements-mace.txt` belongs in `.venv-mace`, and
`requirements-fairchem.txt` belongs in `.venv`. `run_baselines.py --model all`
dispatches each backend through the appropriate interpreter. Override the
defaults with `MATSIM_BASE_PYTHON` and `MATSIM_MACE_PYTHON`.
