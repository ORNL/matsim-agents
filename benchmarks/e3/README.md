# E3 unified six-material adaptive-acquisition protocol.

Scientific reporting and launch gates are specified in
[`scientific-objectives-v1.yaml`](scientific-objectives-v1.yaml). That spec
defines exploration, learning, and independent discovery as separate outcomes.
It does not authorize campaign execution or DFT; the E3-MLIP campaign remains
blocked pending model-set, endpoint, reference-coverage, protocol, and budget
approval.

E3-MLIP scope is all six classes with 5,000 unique valid configurations planned
per class (30,000 total); per-class model and reference coverage is tracked in
[`mlip-suitability-v1.yaml`](../../research/paper/e3/mlip-suitability-v1.yaml).
The legacy E3-DFT protocol and acquisition budgets remain unchanged.
# E3: paired adaptive acquisition across six materials

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

# 1. Freeze pools, splits, qualified references and overlap checks.
.venv/bin/python -m benchmarks.e3.run prepare --output runs/e3/<run>

# 2. Zero-shot predictions, each backend in its isolated venv (CPU, offline).
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
for m in lifepo4 bcc_refractory; do for s in acquisition references; do
  f=runs/e3/<run>/$m/$s.extxyz; [ -f $f ] || continue
  .venv-mace/bin/python -m benchmarks.e3.mlip_predict --model mace-mp-0-small \
    --input $f --output ${f%.extxyz}.mace-mp-0-small.json
  .venv-uma/bin/python -m benchmarks.e3.mlip_predict --model uma-s-1p1-omat \
    --input $f --output ${f%.extxyz}.uma-s-1p1-omat.json
done; done

# 3. Frozen Round 1 paired decision for one material (needs AMSC_TOKEN for MAG).
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
