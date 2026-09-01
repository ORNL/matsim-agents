# Benchmarks

Reusable performance and backend-comparison drivers belong here.  Diagnostic
and paper-specific post-processing remains under `scripts/diagnostics/` until
it is promoted into a stable benchmark with documented inputs and outputs.

The [`portability/`](portability/) benchmark is the canonical, fixed scientific
gate shared by Frontier, Aurora, and Perlmutter. It deliberately separates one
science configuration from small facility overlays and complements—rather than
replaces—the paper, scaling, and model-comparison benchmarks.
