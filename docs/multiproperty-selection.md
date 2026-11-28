# Multi-property characterization, selection, and hypothesis revision

One workflow connects property evidence to material selection and to the next
calculation to run. Thermal, electrical, mechanical, and radiation requirements
decide which candidates are promising and which calculations are worth their cost.

```text
characterize (MP2-4) -> evidence (MP1) -> select (MP6) -> revise hypothesis (MP7)
        ^                                      |                  |
        |                                      v                  v
   acquisition plan (MP8) <-------- confidence gain per node-hour; approval required
```

## Four separate statuses

Do not merge these. A row in any table below belongs to exactly one of them.

| Status | Meaning |
| --- | --- |
| Software implemented | Code exists with a typed interface. |
| Software verified | Unit tests with synthetic or mock inputs pass. |
| Numerically validated | A real calculation was run, converged, and checked against a reference. |
| Scientifically qualified | Measured or converged-calculated evidence meets the stated requirement. |

**Current state: everything is software implemented and verified. Nothing is
numerically validated or scientifically qualified.** No real QE SCF, EPW,
ShengBTE, or SPECTRA-PKA run for this campaign exists.

## Evidence contract (`science/properties/calculators.py`)

Every property value is a `PropertyEvidence` record with a kind:
`MEASURED`, `CALCULATED`, `SURROGATE_PREDICTED`, `APPROXIMATED`, `MISSING`, or
`UNSUPPORTED`.

* `MEASURED` needs an experimental level, a value, an uncertainty, and evidence ids.
* `CALCULATED` needs DFT-level evidence and `converged=True`.
* `SURROGATE_PREDICTED` must be below low-fidelity DFT.
* `APPROXIMATED` must name its approximation.
* `MISSING` and `UNSUPPORTED` carry a reason and never a value. There is no zero-fill.
* Thermal, electrical, and structural-performance properties need
  `conditions.temperature_K > 0`.
* Thermal-conductivity degradation needs an irradiation or defect state.
* Only `MEASURED` and `CALCULATED` records can be qualified.

DOS, band structure, and sigma/tau are not conductivity. Elastic stiffness is not
yield strength, toughness, creep, or ductility. Each recipe lists its forbidden
fallbacks (`matsim-agents` `list_recipes()`).

## Property capability matrix (`campaign/property_recipes.py`)

| Method | Kind | Backend state | Properties produced |
| --- | --- | --- | --- |
| `existing_data` | data | Available | Supplied records only |
| `stress_strain_mlip` | internal | Available if the MLIP provides stress | Elastic stiffness, bulk, shear, stability |
| `stress_strain_dft` | internal | Needs a configured DFT backend | Same, at DFT level |
| `wiedemann_franz` | internal | Available, approximation | `kappa_e` from sigma only |
| `epw` | external | `epw.x` is built in the QE install but is not on `PATH`, has no workflow inputs here, and has never been run | Electrical conductivity, thermal pieces |
| `kubo_greenwood` | external | No executable | Electrical conductivity |
| `shengbte` | external | No executable | `kappa_ph` |
| `almabte` | external | No executable and no measured cost | `kappa_ph` |

Wiedemann-Franz gives `kappa_e` as `APPROXIMATED`; it is not `kappa_ph`. A total
thermal conductivity needs both parts (`integrations/external/transport.py`).
Unsupported methods return `UNSUPPORTED` with a reason; they never return a number.

Radiation swelling, DBTT, and a conductivity-degradation model do not exist.
Degradation can only be derived approximately from supplied pristine and
irradiated conductivity (`campaign/radiation_properties.py`).

## Surrogates (`campaign/property_surrogates.py`)

A surrogate is accepted only after a compatibility check, an independent test, and
an out-of-distribution gate. Its output is `SURROGATE_PREDICTED`, so it can rank a
candidate as `INSUFFICIENT_EVIDENCE` at best.

## Scoring (`campaign/selection.py`, `selection-v1`)

Requirements come only from the caller's `SelectionSpec`: objectives (direction and
weight) and hard constraints. No threshold or weight is invented. Status, first
match wins:

| Status | Rule |
| --- | --- |
| `INFEASIBLE` | A scoreable value violates a hard constraint. |
| `INCOMPLETE` | A required property has no scoreable value. |
| `INSUFFICIENT_EVIDENCE` | All values exist, but one is surrogate, approximated, below the required qualification, or a constraint is met with too little confidence. |
| `QUALIFIED` | Every value is measured or calculated at the required qualification and every constraint holds. |

Candidates are compared by Pareto dominance. A weighted desirability score is an
optional view with explicit parameters. Thermal and electrical conductivity are
coupled, so correlated objectives share one weight. Uncertainty is propagated by
seeded Monte Carlo into score spread, constraint probability, rank distribution, and
Pareto robustness. `render_report(result)` lists per-objective values, weights, and
the text of how they combine. The result carries a `score_digest`.

## LLM hypothesis revision (`campaign/selection_revision.py`)

The status of a revision is decided by rules, not by a model:

* `SUPPORTED`: the candidate is `QUALIFIED` and nondominated on measured or
  calculated evidence.
* `CONTRADICTED`: a measured or calculated value violates a constraint.
* `UNRESOLVED`: anything else.

A model may draft rationale text only, from an evidence packet. `validate_draft`
rejects a draft containing a number not in the packet or an evidence id not supplied;
the deterministic text is used instead and the rejection is recorded. Each revision
records traceability (score digest, evidence ids, model id) in the ledger. `SUPPORTED`
is computational support for stated requirements, not scientific qualification.

## Property acquisition (`campaign/property_acquisition.py`)

`plan_property_acquisition` ranks proposed calculations by expected gain in selection
confidence per node-hour, with a predicted cost for each. The plan
`requires_approval` and is never dispatched. `reserve_approved(plan, ledger,
approved, authorized_by=...)` only reserves budget, for explicitly named ids and a
named person. The `property` budget category defaults to 0 node-hours. Costs for
backends without a benchmark are unmeasured placeholders.

## Radiation properties (`campaign/radiation_properties.py`)

For the W-Ta-Cr-V-Hf-Mo reference alloy (W640 Ta128 Cr64 V96 Hf32 Mo64, 1,024 sites),
a radiation property is a supplied matching record, an approximate derivation, or
`MISSING` with blockers. A record at another dose or structure is never borrowed.
Blockers include unconverged cutoffs for the new Re/Os/H/He pseudopotentials, no validated
SPECTRA-PKA result (installed, one smoke run), and no cascade, swelling, or DBTT model.

## Example

[examples/characterization/fusion_alloy_selection.yaml](../examples/characterization/fusion_alloy_selection.yaml)
is **synthetic**: candidates A, B, and C, invented values, and a selection spec at
900 K. It is exercised by `tests/test_fusion_alloy_synthetic_campaign.py`, which checks
that missing and approximated values are not scored, that B and C need acquisition, that
reservation requires a person and explicit ids, and that the LLM cannot change a score.
It demonstrates software behaviour only and says nothing about any real alloy.

## Remaining work before any qualification claim

1. Resolve the QE segmentation fault (see [radiation-damage-status.md](radiation-damage-status.md)).
2. Converge k-points and cutoffs for the 1,024-site alloy.
3. Convergence-test cutoffs with the Re/Os/H/He pseudopotentials (added); validate SPECTRA-PKA
   against an independent reference case (installed and smoke-run).
4. Configure and benchmark EPW or Kubo-Greenwood, and ShengBTE or almaBTE, to replace
   placeholder costs.
5. Compare calculated values with measurements before using `QUALIFIED` for a decision.
