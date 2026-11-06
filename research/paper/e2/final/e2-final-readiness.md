# E2 final readiness: NOT COMPLETE

| Requirement | Status | Smallest action to resolve |
|---|---|---|
| LiFePO4: 4 cases qualified, reproducible | PASS with limitation | Mount private archive 4e18697 to re-verify OUTCAR hashes |
| Phosphorene: 4 cases, failure documented | PASS with limitation | Same archive check |
| Nb-Ta-O: v1/v2 reconciled, 6 cases scored | PASS | — |
| Zn-formate: 4 cases scored, qualified | PASS | — |
| BCC: qualified case(s), 4 arms, scored | PASS (2 cases) | Optional: 3x3x3 static k-point and ISPIN=2 checks to decide the ranking case |
| **Cantor FCC: case(s), 4 arms, scored** | **FAIL** | Finish the 3 running relaxations; qualify one with verify_vasp_run.py; freeze 1-2 cases; run 4 arms; score |
| Rater metadata corrected (same vendors) | PASS | — |
| Scoring complete for every response | PASS (80/80, two raters) | — |
| Evaluator provenance | PARTIAL | Historical revision unrecoverable; disclosed. no_llm reproduced 20/20 under pinned evaluator |
| Answer-key freeze chronology | PARTIAL | lfp-phosphorene-v1, nbtao-v1 lack pre-run freeze evidence; disclosed |
| Inference costs consolidated | PASS | — |
| Single-seed and dependency limits stated | PASS | — |

**Blocker:** Cantor FCC coverage. Nothing else blocks; the remaining items are disclosed limitations.

No HPC jobs were submitted. Cantor completion requires your authorization only if the running relaxations fail or a k-point/magnetic check is needed.
