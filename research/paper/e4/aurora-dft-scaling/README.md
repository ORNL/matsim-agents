# E4 Aurora DFT scaling campaign

This planning package is separate from the existing E4 Si/MLIP/QE portability
draft. It covers the proposed refractory BCC, Cantor-family FCC, and LiFePO4
workloads only. It does not change E2 or authorize scheduler or DFT execution.

## Contents

- `campaign-plan.md`: scope, protocols, scaling schedule, pilots, cost model, and readiness.
- `vasp-ab-pilot-proposal.md`: planning-only, resource-capped comparison of the custom and ALCF VASP 6.6.1 binaries; currently NO-GO pending input recovery and explicit execution approval.
- `existing-data-audit.md`: public-dataset and local E2 overlap audit, including unresolved gaps.
- `allocation-status.json`: required project IDs and unavailable live account fields.
- `manifests/`: deterministic candidate IDs, exact composition candidates, and disjoint run assignments.
- `../../../../benchmarks/e4/aurora_dft/`: planner and local SQLite registry implementation.
- `../../../../deployments/aurora/e4-templates/`: non-executing PBS templates with unverified queue/walltime placeholders.

Generate the planning manifests with:

```bash
.venv/bin/python -m benchmarks.e4.aurora_dft.prepare
```

Run the focused checks with:

```bash
.venv/bin/pytest -q tests/test_e4_aurora_dft.py
```

The generated rows are candidates, not verified unique structures. Structure
hashes and canonical identifiers remain null until source archives, parent
geometries, and the live duplicate audit are available. Queue placeholders are
intentionally invalid; the PBS templates only print their identity and exit.