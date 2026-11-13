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

The generated manifest and execution policy are validated by
[`production.py`](production.py); the Perlmutter array launcher is
[`job-e3-mlip-production-perlmutter.sh`](../../deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh).
Campaign execution uses MACE evidence only and does not read or submit E3-DFT/
VASP calculations. One strategy and seed support descriptive reporting only,
not a comparison or inferential claims. Resource authorization is still
required before submission.

Use the [CPU-generation submission helper](../../deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh)
to execute and validate the actual first-round structures before requesting GPUs.
The [shared executable catalog](feasibility.py) filters unavailable prototypes
and exposes joint capacities to both policy implementations. The production
design remains LLM-only. See the [generation and recovery contract](../../docs/workflows/run-campaigns.md)
for raw proposal retention, generation shortfalls, and the frozen one-call
limit that blocks unapproved replacement decisions.

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
