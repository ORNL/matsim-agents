# E2 publication readiness: NO-GO (one blocker)

Checked 2026-11-06T02:49Z.

## Blocker

| Item | Why it blocks | Smallest action |
|---|---|---|
| Cantor FCC has no E2 case | Six-material coverage is the stated E2 scope. No Cantor relaxation has met EDIFFG (Fmax 0.055-0.314 vs 0.02 eV/A), several ionic steps hit NELM=200, random3 was killed (exit 143), and no static exists. Partial energies are inadmissible. | Leave PBS 9040206 untouched. Two cases are staged unfrozen in `research/paper/e2/drafts/cantor-fcc-v1.draft.yaml` (no reference answers, not runnable). After `ideal` has verified relax and static records: fill evidence, set and hash reference answers, run the 4 arms, obtain 2 blinded scores per response, rerun `scripts/e2/consolidate_e2.py`. Nothing else. |

## Scope after Cantor

No further DFT, seeds, or materials are scientifically justified. Cantor completion resolves six-material coverage only; it does not restore historical reproducibility.

## Passed

| Requirement | Status |
|---|---|
| LiFePO4, phosphorene, Nb-Ta-O (v1+v2), Zn-formate, BCC: cases, 4 arms | PASS (20 cases, 80 responses) |
| Two-rater blinded scoring for every response | PASS (160/160) |
| Rater metadata (not vendor-independent) | PASS |
| no_llm reproduced by pinned evaluator | PASS (20/20) |
| Inference costs consolidated | PASS |
| Aggregates by independent evidence groups (8 clusters) | PASS |

## Disclosed limitations (not blockers; persist after Cantor)

Publication readiness is not full historical reproducibility. These remain explicit limitations:

- Historical evaluator revision for LLM arms unrecoverable; stored outputs used as-is (no_llm reproduced 20/20 under the pinned evaluator).
- lfp-phosphorene-v1 and nbtao-v1 lack independent pre-run answer-key freeze evidence.
- LiFePO4/phosphorene raw OUTCARs are in a private archive (4e18697) not verified here.
- Single seed; 8 evidence clusters; LLM-arm differences are within one case. No strategy superiority claim.
- Raters are LLMs from the same vendors as the assessors; anonymized style can reveal arm family.
- BCC ranking and Zn polymorph ordering are narrow by design (no k-point/spin sensitivity; 0.037 meV/atom gap).

Decision: **GO** for the five completed materials as scoped, disclosed results. **NO-GO** for "six-material E2 complete" until Cantor is qualified.
