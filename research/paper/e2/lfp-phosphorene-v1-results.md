# E2 results: LiFePO4 and phosphorene (suite `e2-lfp-phosphorene-v1`)

- Run: `raw/lfp-phosphorene-v1-hvk5dot2/`. `answers.jsonl` sha256 `4c410d2e…`; `summary.json` sha256 `142695fe…`; protocol digest `974d24de…`.
- Config: `configs/lgea-dea-mag-v1.yaml`. Panel: claude-sonnet-5-5 and gemini-2.5-flash through MAG; temperature 0; seed 0; 2 critique rounds.
- Evidence: frozen packets built from private archive `matsim-agents-dft-reference@4e186975` by `scripts/e2/extract_vasp_evidence.py`; values were checked against the OUTCARs (`evidence/`).
- All arms received identical gold-free packets and question text, and were graded by the same validator (`grade_answer`).
- Answer keys were frozen in `benchmarks/agent_eval_e2/lfp-phosphorene-v1/cases.yaml` before any assessor ran.
- No new DFT was run for this suite.

## Qualified cases: 8 (LiFePO4 4, phosphorene 4); 32 assessor records

| Case | Key | DEA | single | multi | cross |
|---|---|---|---|---|---|
| lfp-ideal-force-criterion | supported | supported | supported | supported | supported |
| lfp-ideal-afm-ground-state | contradicted | contradicted | contradicted | contradicted | contradicted |
| lfp-antisite-energy-ranking | unresolved | unresolved (no DFT flag) | unresolved | unresolved | unresolved |
| lfp-on-convex-hull | unresolved | unresolved | unresolved | unresolved | unresolved |
| phos-uc-force-criterion | supported | supported | supported | supported | supported |
| phos-uc-threefold-puckered | contradicted | contradicted | contradicted | contradicted | contradicted |
| phos-supercell-consistency | contradicted | contradicted | contradicted | contradicted | **unresolved** |
| phos-vacancy-formation-below-2eV | unresolved | unresolved (no DFT flag) | unresolved | unresolved | unresolved |

## Validator scores

| Arm | correct | attribution | inconclusive | DFT request | calls | completion tokens | latency |
|---|---|---|---|---|---|---|---|
| DEA (`no_llm`) | 8/8 | 8/8 | 4/4 | 6/8 | 0 | 0 | 0 s |
| single_llm | 8/8 | 8/8 | 4/4 | 8/8 | 8 | 7,584 | 118 s |
| multi_llm_no_critique | 8/8 | 8/8 | 4/4 | 8/8 | 16 | 24,529 | 163 s |
| multi_llm_cross_critique | 7/8 | 8/8 | 4/4 | 8/8 | 48 | 97,729 | 646 s |

Paired differences (n = 8) are not significant for any comparison. The suite saturates on verdict accuracy, so it does not discriminate between arms.

## Independent scientific assessment (separate from the validator)

Provisional single-rater, unblinded scoring by the analysis agent. A blinded human or second rater is still required before the paper uses it.

Scores use a 0–2 scale. C = scientific correctness, A = attribution, U = uncertainty handling, F = usefulness of the follow-up.

| Arm | C | A | U | F | Notes |
|---|---|---|---|---|---|
| DEA | 16/16 | 16/16 | 0/16 | 9/16 | Verdicts and proposed test IDs are right. Rationales are fixed templates with no stated caveat. On antisite and vacancy it proposes a DFT test but sets `request_dft=false`, which is internally inconsistent. |
| single_llm | 16/16 | 16/16 | 15/16 | 15/16 | Identifies the 16 μB moment as exactly 4 × 4 μB from an all-parallel start. Calls the ~50 eV antisite gap a geometry artifact and recomputes E_f = −36.7 / −1.10 eV, matching our independent −1.096 eV. Flags the one-neighbour phosphorene geometry and proposes rebuilding from a reference structure. |
| multi_llm_no_critique | 16/16 | 16/16 | 15/16 | 14/16 | Same conclusions as single_llm. |
| multi_llm_cross_critique | 14/16 | 16/16 | 13/16 | 15/16 | Critique corrected a first-round per-atom error (28 atoms) on the hull case, and it added the antisite input-geometry check. On supercell consistency it found the 1.0175 eV/atom anomaly but turned "the archive is inconsistent" into "unresolved". Because the claim is about the archived records themselves, this is an error. |

Findings:

1. DEA alone reaches every verdict on this suite.
   - The LLM arms add evidence-grounded diagnosis: the FM origin of the moment, unphysical defect energies, and the non-puckered geometry. They also add a consistent DFT-request decision.
   - The LLM arms do not change any verdict.
2. Cross-critique costs 4× the tokens and 4× the latency of independent multi-LLM. It produced one wrong verdict, caused by over-hedging, and fixed one intermediate arithmetic slip.
3. The scientific follow-ups the LLM arms agreed on:
   - a completed same-signature FeLi antisite relaxation, after checking the input geometry (81 eV/Å forces);
   - a rebuilt puckered phosphorene P4/P36/P35 set;
   - consistent Li–Fe–P–O hull references.

   These are recommendations; none has been run.
