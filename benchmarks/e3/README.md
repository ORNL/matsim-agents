# E3 campaign protocols

## E3-MLIP benchmark

The current benchmark protocol is documented in
[`e3-mlip-exploration-plan.md`](../../research/paper/e3/e3-mlip-exploration-plan.md)
and encoded in
[`scientific-objectives.yaml`](scientific-objectives.yaml). It covers six
material classes, one LLM-guided strategy, seed `20261113`, and three adaptive
rounds of at most 64 candidate relaxations per class (1,152 maximum total).
Each class is limited to 16 Perlmutter GPU nodes and two hours, with 30-minute
round deadlines and parent-enforced five-minute task cutoffs. The physical
allocation cap is 768 GPU-hours on the premium QOS.

New E3-MLIP manifests require scientific hypothesis tracking. Before each
round the acquisition prompt requests explicit material claims, falsifiable
predictions, stable hypothesis IDs, status, reasoning, and evidence citations.
The complete prompt and raw response, including these hypotheses, are saved in
`round-NN/acquisition/proposal-001.json`. After each completed geometry round,
a separate LLM call saves the updated claims and its full prompt and response in
`round-NN/hypothesis-assessment.json`. The next round receives that assessment
as its previous hypothesis state. All existing IDs must be retained, including
retired claims; invalid or missing citations and omitted IDs reject the response.
The final completed round receives the same assessment even if there is no
next acquisition.

The protocol therefore authorizes at most **two LLM calls per completed round**:
one acquisition call and one additional assessment call. This changes the
previously frozen call budget and requires a newly generated manifest and
explicit authorization before a new campaign; old runs and manifests must not
be resumed under this protocol. A request, raw response, or rejection is
persisted before further processing; a failed or interrupted assessment is not
silently retried and requires operator reconciliation. If generation stops
before optimization, the acquisition proposal remains available but no
post-round assessment is invented. MACE-only evidence cannot establish
thermodynamic stability or DFT-level accuracy.
In the E3-QE variant the assessment runs after the current QE audit, training
decision, and promotion comparison, before committing the iteration. This also
applies to the final iteration and to deferred or untriggered training. Its
exact numerical input is retained in each iteration's
`hypothesis-evidence.json`; all earlier audit inputs remain available to later
acquisition and assessment calls. A missing or rejected assessment blocks
completion and resume. The same two-call budget applies: no separate
pre-audit geometry assessment is made.

The generated manifest and execution policy are validated by
[`production.py`](production.py); the Perlmutter array launcher is
[`job-e3-mlip-production-perlmutter.sh`](../../deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh).
Campaign execution uses MACE evidence only and does not read or submit E3-DFT/
VASP calculations. One strategy and seed support descriptive reporting only,
not a comparison or inferential claims. Resource authorization is still
required before submission.

Use the [CPU-generation submission helper](../../deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh)
to execute and validate the actual first-round structures before requesting GPUs.
The helper reserves the entire 460,800 GPU-second allocation **before `sbatch`**,
then binds that reservation to its scheduler job. Direct `sbatch` invocation of
the GPU launcher is unsupported: it requires the helper's reservation ID.
Startup failures, cleanup, and scheduler teardown remain fully charged.
The launcher does not release time from an EXIT trap because Slurm has not
yet reported the final allocation lifetime.

After the job is terminal, reconcile it from a CPU host using the reservation
ID printed by the helper:

```bash
.venv/bin/python -m benchmarks.e3.production budget-reconcile \
  --state runs/e3-mlip-production/resource-budget.json \
  --reservation-id submission-REPLACE_WITH_PRINTED_ID
```

Only verified terminal `sacct` allocation records release the unused balance.
Unavailable, ambiguous, or nonterminal scheduler records retain the full charge.
An ambiguous `sbatch` failure also retains its reservation; an operator must
establish whether an allocation was created before releasing any time.
The [shared executable catalog](feasibility.py) filters unavailable prototypes
and exposes joint capacities to both policy implementations. The production
design remains LLM-only. See the [generation and recovery contract](../../docs/workflows/run-campaigns.md)
for raw proposal retention, generation shortfalls, and the frozen one-call
limit for acquisition decisions that blocks unapproved replacement decisions.

Every round outcome records one LLM hypothesis assessment in the round
directory: `hypothesis-assessment.json` for a completed round, and
`hypothesis-assessment-<outcome>.json` for `stopped_generation_exhausted`,
`round_time_limit`, or `round_failed`. Each record states its `round_outcome`,
the hypotheses it assessed, and the evidence it cited; the campaign summary
and report reference the assessment of the stopping round.

To audit hypothesis records after a campaign (read-only on the run data):

```bash
.venv/bin/python -m benchmarks.e3.hypothesis_audit runs/<campaign> \
  --output runs/<campaign>/hypothesis-audit
```

The command writes `interaction-index.csv`, `hypothesis-evolution.csv`,
`review-flags.csv` and a SHA-256 `manifest.json`. The revision types are
mechanical field diffs. The flags mark rows that need human review:

- hypotheses resolved on introduction;
- status changes without new converged data;
- resolved statuses that cite several compositions;
- changes to claim or prediction text;
- rounds with no converged structures;
- records that were not accepted.

The tool does not judge whether a change is a scientific revision.

Generator revision 3 retains the 0.08 Angstrom phosphorene reconstruction bound
but samples collective in-plane displacements with zero net translation.
Formate proton variants rotate all C-H vectors independently by at most
15 degrees, keeping bond lengths and checking framework connectivity. Both
changes add genuine degrees of freedom without relaxing deduplication.
Seeded PyXtal uses NumPy's `Generator`, as required by the installed version.
Deduplication performs the same Niggli reduction directly before matching,
avoiding the matcher's composition-keyed cache equality bottleneck for large
frameworks. Matching tolerances, primitive-cell handling, and equivalence
decisions are unchanged.

`generation-time.json` preserves every generation attempt, including failed
attempts and failures before publishing a receipt. Cached generated structures
do not reset this cumulative time. `execution-time.json` similarly tracks each
supervised execution attempt; both the supervisor and worker-step deadline use
the same remaining balance after generation and prior execution.
Reservations are persisted before work. An attempt killed before its finish
record retains its entire remaining allowance, not a fresh 30 minutes on retry.
Interrupted execution evidence without a time ledger is refused rather than
assigned an invented zero duration. Completed rounds are not rerun or recharged.

Valid generation exhaustion is a scientific stop, not an infrastructure failure.
The round remains incomplete with status `stopped_generation_exhausted`.
`acquisition/outcome.json` retains the requested/accepted counts, rejection
reasons, partial structures, and remaining capacity groups for LLM ingestion.
It explicitly records that no new optimization evidence was collected. The
round's one hypothesis assessment then evaluates the prior hypotheses against
this outcome and is saved as `hypothesis-assessment-stopped_generation_exhausted.json`.
The campaign writes this stop to its summary/report and exits normally without
starting later rounds. The CPU `prepare` command instead exits with documented
code `3` and JSON diagnostics, preventing the submission helper from requesting
GPUs for an incomplete acquisition. Invalid actions, infrastructure errors, and
timeouts remain explicit errors; no structures are silently substituted.

A separately authorized full campaign rerun must use a separate output root
to preserve prior evidence. Set `MATSIM_E3_MLIP_RESOURCE_STATE` to the original
campaign's absolute `resource-budget.json` path so the helper reserves and the
terminal scheduler reconciliation charges all attempts against the same physical GPU cap; do not create a
fresh resource budget for the rerun.

The discovery API now accepts a `SeedGenerationAction` for explicit
formula-unit tiers, AFLOW candidate caps, PyXtal counts, space-group/prototype
filters, and reproducible seeds. Pass it to `explore_composition` to relax the
generated batch with the selected MLIP; the action and geometry metadata are
recorded separately per action. This is the shared generation/relaxation
interface. `benchmarks.e3.production` executes the registered three-round protocol.
Actions are chosen from the material-specific allow-list; structures pass
composition, periodicity, geometry, exact-hash, and symmetry-deduplication
checks. The LLM sees only its material stream's prior round summaries and
allowed actions. It must cite design or stream-local evidence IDs.
Cell-relaxation settings and class-specific generators are fixed in the
manifest. Generic legacy `generate_seeds` defaults are not used.

## Legacy E3-DFT paired adaptive acquisition across six materials

E3 compares two adaptation strategies, using the terminology of the revised
manuscript:

- **LLM-guided adaptation** selects follow-up DFT calculations through
  evidence-grounded scientific reasoning. Every selection must cite evidence
  already in that strategy's own ledger. Selections outside the candidate pool
  are rejected.
- **Rule-based adaptation** (v2) uses a frozen, deterministic procedure:
  - Uncertainty is the existing `score_ensemble` force disagreement of a
    MACE-MP/UMA committee.
  - Only the top 25% of candidates by uncertainty are eligible.
  - Diversity uses the existing `soap_descriptor` and `soap_novelty`.
  - Near-duplicates are excluded.
  - Each candidate is scored as u/u_max + 0.5 × novelty/novelty_max.

  v1 (composition fingerprints) is superseded.

The frozen design is [`protocol.yaml`](protocol.yaml); its SHA-256 is recorded
in every ledger and accounting file.

## What both strategies share

- Initial model, candidate pool, reference data, and VASP method signature.
- Validation requirements and conditional fine-tuning criteria.
- Acquisition ceiling: **8 logical DFT attempts per strategy per activated
  material**, in **4 rounds of at most 2**. A failed calculation consumes its
  attempt. Restarting the same calculation uses no new slot, but its node-hours
  are charged.

Each strategy has its own append-only `EvidenceLedger`. Both ledgers start from
the same content-addressed seed record, so their initial evidence ID matches.
Evidence acquired by one strategy is never visible to the other.

## Leakage control

Each material's pool goes through the existing `make_split` (`generic_group`,
70/15/15 by scientific group). Only the training role is visible to either
strategy. Validation and test groups never appear in prompts, rule inputs, or
the shared cache. The dry run aborts if a strategy selects a withheld structure
or if any split leakage check fails. The 20 E4 first-batch calculation IDs are
excluded from all E3 pools, keeping E3 independent of the running E4 campaign.

## Accounting

| Quantity | Definition |
|---|---|
| Logical attempts | Every DFT request by a strategy, including failures and retries |
| Unique physical calculations | Distinct (structure hash, method signature); identical requests run once through a shared cache |
| Total expenditure | Measured node-/device-hours of unique calculations plus LLM tokens/time, MLIP inference, and fine-tuning |

Each strategy is still charged the counterfactual cost of its own requests when
the physical calculation is shared.

## Commands

Load the Aurora XPU module stack first
(`source deployments/aurora/setup/aurora-module-stack.sh; load_aurora_xpu_modules`).

```bash
# Six-material inventory, readiness and blockers.
.venv/bin/python -m benchmarks.e3.run inventory --output research/paper/e3/inventory.json

# Legacy reference-backed E3-DFT preparation.
.venv/bin/python -m benchmarks.e3.run prepare --output runs/e3/<run>

# Candidate-only E3-MLIP preparation; no DFT data are read.
.venv/bin/python -m benchmarks.e3.run prepare-candidates \
  --output runs/e3-mlip/<run> \
  --materials lifepo4 bcc_refractory fcc_cantor phosphorene zn_formate_mof nb_ta_o

# Zero-shot predictions on all available E3-MLIP candidate geometries.
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
for m in lifepo4 bcc_refractory fcc_cantor phosphorene zn_formate_mof nb_ta_o; do
  f=runs/e3-mlip/<run>/$m/candidates.extxyz
  .venv-mace/bin/python -m benchmarks.e3.mlip_predict --model mace-omat-0-medium \
    --input $f --output ${f%.extxyz}.mace-omat-0-medium.json --device cpu
done

# Legacy DFT-enabled adaptive decision (not part of E3-MLIP).
.venv/bin/python -m benchmarks.e3.run round1 --source runs/e3/<run> \
  --output runs/e3/<round1-run> --material bcc_refractory
```

`round1` writes `round1-manifest.json` (hashed), per-strategy ledgers,
`rejected-proposals.json`, `rule-scores.json`, and the LLM prompt and reply.
LLM selections must also pass a numerical-claim check: every decimal value
they state must appear in the prompt.

`references.py` admits only frames whose INCAR, VASP version, and SCF
convergence match the frozen signature. It records every rejected source and
the reason. `replay` uses `select_candidates(strategy="ensemble")` on stored
MACE/UMA committee forces for Rule-based adaptation. For LLM-guided adaptation
it calls the protocol's MAG route; each selection must be in the pool, state a
scientific purpose, and cite evidence in that strategy's own ledger. An
unavailable LLM is recorded as blocked and never replaced. `dry-run` remains
for offline plumbing checks only.

## Status

Every material is currently **blocked** for live acquisition; see
[`research/paper/e3/README.md`](../../research/paper/e3/README.md). LiFePO4
and BCC are the intended first paired comparison once their reference and
method requirements are met.
