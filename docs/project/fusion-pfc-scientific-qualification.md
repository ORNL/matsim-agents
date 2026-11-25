# Fusion PFC: scientific qualification status

This page separates four states: implemented, verified, validated and qualified.

| State | Meaning | Fusion PFC status |
|---|---|---|
| Implemented | Code exists | Policy, objectives, scenarios, nuclear-evidence import, scoring, revision, acquisition, transmutation, CLI |
| Verified | Tests show the code does what it states | Unit and CLI tests with SYNTHETIC data; see below |
| Validated | Outputs agree with independent reference data | **Not done.** No benchmark anchor or evidence comes from measurements or qualified calculations. |
| Qualified | Validated within a declared scope, with a basis on record | **Not done.** No candidate, objective or score is qualified. |

## What the tests verify

- **Determinism**: scores and digests are reproducible, and the observable order is
  canonical.
- **Incomplete candidates**: they are never scored, and missing objectives are never
  zero-filled.
- **Condition matching**: evidence is used only when its conditions match the scenario,
  including ion incidence angle and dose geometry.
- **Nuclear evidence**: the family contracts are enforced.
- **Revision**: LLM drafts are accepted only if every number and id appears in the
  evidence packet. Revisions are append-only.
- **Acquisition**: plans stay within the real budget, label their priors, refuse
  proxies, and dispatch nothing.

## What is required before any materials claim

1. Benchmark anchors with cited sources, replacing `synthetic-demo-v1`.
2. Evidence for each objective from converged calculations or measurements, recorded
   in the ledger.
3. Nuclear evidence (dpa, transmutation, activation) from external codes run on the
   real alloy and spectrum.
4. A validation study comparing predictions with independent data, and a documented
   qualification basis.

Production QE, transport, neutron and activation runs have **not** been launched.

Tests: [tests/test_fusion_revision.py](../../tests/test_fusion_revision.py),
[tests/test_fusion_scenario.py](../../tests/test_fusion_scenario.py),
[tests/test_fusion_objectives.py](../../tests/test_fusion_objectives.py),
[tests/test_fusion_example.py](../../tests/test_fusion_example.py).
