# E4 Aurora DFT scaling campaign

This planning package is separate from the existing E4 Si/MLIP/QE portability
draft. It covers the proposed refractory BCC, Cantor-family FCC, and LiFePO4
workloads only. It does not change E2. The planning material itself does not
authorize execution; the separately approved 20-calculation first production
batch is documented in
[`deployments/aurora/jobs/e4-first-batch/`](../../../../deployments/aurora/jobs/e4-first-batch/README.md).

## Contents

- `campaign-plan.md`: scope, protocols, scaling schedule, pilots, cost model, and readiness.
- `stage1-blocker-resolution-report.md`: refreshed Stage 1 inventory, PAW, duplicate, restart, PBS, and allocation findings.
- `bcc-restart-qualification-proposal.md`: focused two-job, 0.5-node-hour BCC restart test; executed on Aurora (restart mechanics GO, numerical equivalence not established).
- `lifepo4-inventory-v1/`: frozen 510-structure LiFePO4 antisite inventory used to select the first-batch LiFePO4 cases.
- `stage1-readiness-report.md`: superseded historical preparation report; do not use for current readiness decisions.
- `vasp-ab-pilot-proposal.md`: planning-only, resource-capped comparison of the custom and ALCF VASP 6.6.1 binaries; currently NO-GO pending input recovery and explicit execution approval.
- `existing-data-audit.md`: public-dataset and local E2 overlap audit, including unresolved gaps.
- `allocation-status.json`: refreshed balances and queue snapshot; project-specific queue acceptance remains unverified.
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

The planning manifests are not the frozen input bundle. The materialized
`stage1-inputs-v2/` bundle records POSCAR hashes, canonical identities, input
checksums, and secure PAW references; it contains no POTCAR bytes and is not
authorized for execution. Duplicate screening against inaccessible public and
private archives remains incomplete. The PBS templates are still preflight-only
and do not invoke VASP.