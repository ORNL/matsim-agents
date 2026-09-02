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
