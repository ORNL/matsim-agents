# E2 publication readiness: NO-GO (one blocker)

Checked 2026-11-06T02:49Z.

## Blocker

| Item | Why it blocks | Smallest action |
|---|---|---|
| Cantor FCC has no E2 case | Six-material coverage is the stated E2 scope. No Cantor relaxation has met EDIFFG (Fmax 0.055-0.314 vs 0.02 eV/A), several ionic steps hit NELM=200, random3 was killed (exit 143), and no static exists. Partial energies are inadmissible. | Wait for PBS 9040206 (running; no resubmission needed yet). When `ideal` qualifies with its static: freeze 2 cases (force criterion; ranking-adequacy = unresolved), run 4 arms, score with both raters, rerun `scripts/e2/consolidate_e2.py`. If ideal does not finish by walltime, a restart of that one case needs your authorization. |

## Passed

| Requirement | Status |
|---|---|
| LiFePO4, phosphorene, Nb-Ta-O (v1+v2), Zn-formate, BCC: cases, 4 arms | PASS (20 cases, 80 responses) |
| Two-rater blinded scoring for every response | PASS (160/160) |
| Rater metadata (not vendor-independent) | PASS |
| no_llm reproduced by pinned evaluator | PASS (20/20) |
| Inference costs consolidated | PASS |
| Aggregates by independent evidence groups (8 clusters) | PASS |

## Disclosed limitations (not blockers)

- Single seed; 8 evidence clusters; LLM-arm differences are within one case. No strategy superiority claim.
- Historical evaluator revision for LLM arms unrecoverable; stored outputs used as-is.
- lfp-phosphorene-v1 and nbtao-v1 lack independent pre-run answer-key freeze evidence.
- LiFePO4/phosphorene raw OUTCARs are in a private archive not mounted here.
- Raters are LLMs from the same vendors as the assessors; anonymized style can reveal arm family.
- BCC ranking and Zn polymorph ordering are narrow by design (no k-point/spin sensitivity; 0.037 meV/atom gap).

Decision: **GO** for the five completed materials as scoped, disclosed results. **NO-GO** for "six-material E2 complete" until Cantor is qualified.
