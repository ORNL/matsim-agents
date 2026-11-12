# E3 campaign protocols

## E3-MLIP crystallographic exploration

The revised design is documented in
[`e3-mlip-exploration-plan-v1.md`](../../research/paper/e3/e3-mlip-exploration-plan-v1.md).
It makes adaptive structure generation and mandatory MACE geometry optimization
the workload: 5,000 unique valid configurations per class **per strategy**,
with matched Rule-based and LLM-guided budgets and no DFT labels, references,
or VASP.

The current Perlmutter preparation job remains a CPU-only single-point
diagnostic; it does not implement the revised campaign. The feasibility pilot
and production campaign are not authorized or launched. Per-class checkpoint
limitations and preflight checks are tracked in
[`mlip-suitability-v1.yaml`](../../research/paper/e3/mlip-suitability-v1.yaml).

The discovery API now accepts a `SeedGenerationAction` for explicit
formula-unit tiers, AFLOW candidate caps, PyXtal counts, space-group/prototype
filters, and reproducible seeds. Pass it to `explore_composition` to relax the
generated batch with the selected MLIP; the action and geometry metadata are
recorded separately per action. This is the shared generation/relaxation
interface. `benchmarks.e3.mlip_exploration.run_generation_round` executes a
per-strategy action list with MACE-OMAT-0 medium and no relaxation cap, or
selects from an allow-listed action space. The deterministic selector balances
composition, size, generator, prototype and space-group coverage, with a
smoothed convergence-rate term from that strategy's prior same-class MLIP
results. The LLM selector receives only that strategy's round summaries and
allowed actions; it must cite initial design or strategy-local result IDs, and
unsupported or repeated actions are rejected. Cell relaxation is off by
default and should be enabled only for a preflight-verified class/model
combination. These generic selectors do not replace the required
class-specific generators for alloys, phosphorene, defects and the MOF. The
E3-MLIP campaign must use explicit actions rather than the backward-compatible
`generate_seeds` defaults.

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
