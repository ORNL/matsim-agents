# E1 Readiness and Evidence Review

Date: 2026-10-30
Status: audit note; not a protocol amendment, scientific approval, candidate authorization, or job authorization.

This note records the status of the existing GPT-6 Luna E1-R review set, the reviewer packet audit, the held-out (H) exposure finding, and the relationship between E1-E4 and the manuscript's four contributions. It does not change or reproduce any question, response, score, crosswalk, key, rubric, or protocol. No new inference was run for this review.

## Current Status Update (2026-10-30 15:15 UTC)

The status statements below are the audit snapshot recorded for this review and are not the current execution state. Existing E1 campaign execution and resource approvals remain in force; this note does not add or revoke authorization. Collect answers from all eligible remote and local candidates before D scoring/qualification or debate-panel selection. Reviewer assignment and conflict checks may proceed in parallel, but do not begin D scoring until collection is complete.

Aurora job `8990495.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`, a post-patch validation run using the existing pinned R1 manifest and resource plan with startup tracing enabled, was running at the time of this update. Its model log records the TorchComms patch installed and model-server initialization underway; no candidate response was present yet. This is startup-path evidence only, not a successful E1-L response or scientific result. Refresh job state and outputs before relying on it. The original status and approval-gate tables below are retained as historical findings and do not supersede this update.

## Summary

- The GPT-6 Luna E1-R collection has 18 complete responses available for review, but scientific review and candidate qualification are pending. The original 18-request attempt produced 13 complete responses and five truncations; a separate five-request recovery produced the remaining five. Report this as 13/18 first-attempt completion and 18/18 recovery-assisted coverage across 23 requests, not as 18/18 first-attempt success.
- Both blinded reviewer packets contain the same 18 response IDs in different orders. Score fields and reviewer-level qualification outcomes are blank. The packet structure has no explicit model/provider/endpoint/candidate/condition identity fields. The protected crosswalk and reviewer key were not accessed.
- The reviewer packets are stored in a mode `2700` directory and are each mode `0600`; ACL checks found no group/other grants. They remain ignored under `runs/` and are not tracked or present in reachable Git history. Use restricted, separate delivery and storage, and re-check destination access controls before distribution. Reviewer identities have not been assigned or conflict-checked.
- All 18 current H prompt texts were found verbatim in tracked history of `research/paper/e1-preparation.md`. The earliest matching commit identified in the available history is `33707934a0c21865920670c953ca3d18c74d9c71` (2026-10-28 17:37 UTC); matching later revisions include `e6809f0cadd973ce766975f2f3506f63de47296b`, `d91424da402d5c98f4842ba59a3e06fe78753c46`, and `257f9ca6f10a7049f6d6c52d4ba95437ea6cb001`. The current H file remains outside the repository, mode `0600`, with SHA-256 `ddbd2c6e65c133e0db7ea09932f5e4d24b83571348556e7e4f5f5969ac1f6638`. The E1 runner has not served H. Repository-history exposure nevertheless disqualifies this set as an uncontaminated holdout; replace and independently review it before any held-out evaluation.
- No new E1 inference is authorized by this note. For any further D inference, scientific protocol review, candidate/route approval, resource approval, and approval of the exact candidate-bound manifest remain distinct gates.

## Reviewer Preparation

The two packets are the assignments `reviewer-1` and `reviewer-2` in `runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/`. An offline structural check found 18 unique IDs per packet, identical ID sets, differing order, and matching protocol/schema/rubric/threshold definitions. All criterion scores are null; rationales, reviewability decisions, reviewability rationales, notes, integrity flags, and reviewer qualification outcomes are empty. The rubric weights are correctness 30%, physical reasoning 20%, falsifiability/tests 20%, evidence/uncertainty 20%, and feasibility 10%.

Before scoring begins:

1. Assign two independent materials-science reviewers, verify conflicts and prior exposure to the response set, and distribute each packet separately over an access-controlled channel. Do not send either reviewer the other packet or any admin crosswalk/key.
2. Confirm that the reviewers have the approved scoring instructions and understand that the model/provider identity must remain blinded. The structural check cannot rule out identity clues volunteered within a response; reviewers should report any such disclosure without attempting to identify the source.
3. Have each reviewer complete and lock all 18 forms independently. Preserve the original forms. Only after both are locked, inspect disagreements and generate the adjudication form. The protocol requires an independent third adjudicator for any criterion disagreement of at least two points or differing qualification outcomes.
4. Import the two forms and any required adjudication, then issue the qualification report. No score, qualification, or adjudication result exists yet.

The local packet directory and files are owner-only (`2700` directory, `0600` files); ACL checks found no group/other grants. The packet contents were not changed. Re-check every delivery and return location before use. The crosswalk and reviewer key remain protected admin files and must stay unavailable to reviewers until the independent scores and required adjudication are locked.

## Approval Gates

| Gate | Current state | Required before proceeding |
|---|---|---|
| Scientific approval of D protocol | `development-protocol.v1.md` is marked proposed for scientific and operational approval. Its hashes match the recorded D packet (`10bd35944511e49259cd920e46a32c96a8bf8116dbaaf2b120962d5ba9eeadce`), response schema (`9eab3b993777b864af11e0f0a9919227e7b8fccb5ae5f0ceffe3eb33adc11129`), and protocol (`a50ef7528acd1296e89864841f1aeac72570af415380aca90e4273feb925740d`). | Two domain reviewers must approve the D prompts, evidence/provenance and hypothetical B framing, rubric anchors, thresholds, veto definition, and D/H scientific independence. Freeze any approval record before using it to authorize a future run; do not edit the frozen artifacts in this audit. |
| Review assignments | Packets exist and are unscored; reviewer identities are not recorded here. | Select, conflict-check, and securely brief two independent reviewers; preserve separation and lock their forms before unblinding or adjudication. |
| GPT-6 Luna scientific qualification | Response collection complete with recovery; human scoring pending. | Complete independent blinded scoring and any rule-triggered adjudication. A collection or parse-completion result is not scientific qualification. |
| Future candidate and route | Candidate-specific readiness is not established by catalog presence or this review. | Pin immutable model/checkpoint identity and license/terms; approve provider/serving route, data handling, operational feasibility, and candidate-specific request/resource ceilings. |
| Facility or provider resources | No resource authorization is granted here. | Obtain the relevant allocation-owner/facility or API account approvals, including exact queue/route, walltime or token/spend ceiling, and stop conditions. |
| Exact D run manifest | No new manifest is approved by this note. | Build and validate a real candidate-bound manifest from the approved D packet, freeze its hashes and output path, and obtain an authorized operator's approval for that exact manifest before invoking the runner. |
| Held-out comparison | Current H is contaminated by tracked-history exposure; the full held-out comparison remains a later gate. | Replace H, independently review and freeze the replacement and full held-out protocol, freeze cohort/panel selection and comparison conditions, then verify the replacement access policy before any held-out access. Do not use the current H set for evaluation. |

These gates are separate. Approval to score the existing GPT-6 Luna responses does not authorize another inference run. Approval of a D protocol does not authorize a candidate, route, facility allocation, spending, or exact manifest. The execution-configuration amendment permits settings only within separately approved candidate-specific runs; it is not a resource authorization.

## H Exposure Finding

The history audit compared the protected current H packet against tracked revisions without printing question wording. Exact text matches were found for all 18 prompts in the tracked preparation document, spanning the six material/task rows. The first matching revision identified in available history is the October 28, 2026 commit listed above; subsequent revisions also retained the prompt text. The test fixture in the E1 runner file contained H identifiers but no exact H prompt text.

The current owner-only file and its digest provide a controlled copy, not a clean holdout. The runner's rejection of H IDs and its recorded non-use show that the current runner did not serve H; they do not remove the earlier repository exposure. The documented consequence is loss of uncontaminated held-out status, not evidence that any particular model or reviewer saw the prompts. Do not publish or quote H content while replacing and reviewing the set.

## E1-E4 and Manuscript Contributions

The table separates source-code or deployment support from scientific evaluation. The live manuscript is `research/paper/manuscript/main.tex`; planned scopes and evidence restrictions are in `docs/numerical-results-run-plan.md`.

| Contribution | Evaluation mapping | Evidence status |
|---|---|---|
| Scalability | E1 local-serving concurrency/throughput and approved workload scaling; supporting DFT throughput/scaling only if retained as a separately designed study. | The manuscript reports Perlmutter VASP strong-scaling values across 1, 2, 4, and 8 nodes and five workload draws (18.1, 28.0, 53.6, and 113.5 frames/h; reported efficiency 74-79%). These are manuscript-reported, not independently reconciled here. The run plan explicitly does not assume historical scaling outputs, scheduler records, or campaign timings are reusable. Local-LLM throughput and resource accounting remain unmeasured. The manuscript's five-campaign AL summary is explicitly provisional pending raw-output and scheduler-accounting reconciliation. |
| Portability across Frontier, Aurora, and Perlmutter | E4: one frozen representative scientific contract executed on all three facilities. | Deployment recipes and engineering qualification are not a scientific portability demonstration. E4 is planned; fresh matched end-to-end scientific execution at each facility, with method/configuration provenance, remains pending. Prior Si/SiC artifacts are engineering-only. |
| Backend flexibility across LLM, MLIP, and DFT | E1 compares eligible LLM candidates/routes; E3 evaluates supported HydraGNN/UMA/MACE combinations against fresh, method-consistent VASP references; matched QE/VASP complete-loop substitution is a separate supporting experiment. | Configurable interfaces and workflow support exist, but interface selection does not demonstrate scientific equivalence or complete-loop substitution. E1 candidate qualification is pending; E3's six-system reference/evaluation work is pending; complete matched QE/VASP substitution remains proposed. Manuscript-reported warm-start cases are bounded evidence only and are not treated as a complete-loop result; underlying records require audit before reuse. |
| Local LLM inference | E1-L, followed by the predeclared local-versus-remote comparison, with local model identity, serving configuration, concurrency, latency/throughput, and physical resource accounting. | No completed local E1 scientific inference is identified. The recorded Aurora GPT-OSS 20B download submission was queued, with no weights or inference results recorded at the last operational snapshot; refresh scheduler state before relying on it. Local candidate/checkpoint and serving feasibility, candidate authorization, and measured throughput remain unknown. |

### E1-E4 Status Ledger

- **E1, hypothesis-generation strategies:** the existing GPT-6 Luna E1-R D-response collection is complete after separate recovery; its scientific review is pending. The 13/18 original first-attempt completion and five recovery requests are separate operational outcomes. E1-L, candidate qualification/panel selection, and the four-condition held-out comparison are not complete. E1b remains a separate proposal.
- **E2, fixed-evidence assessment:** no completed six-system matched C-versus-deterministic assessment is identified in the run plan. Independent evidence packets, answer criteria, and evaluation remain pending; engineering fixtures are not scientific results.
- **E3, pretrained MLIP assessment plus conditional acquisition/adaptation:** fresh, method-consistent VASP references, independent splits, supported-cell baselines, and any validation-triggered adaptive comparisons remain pending. Historical QE labels are excluded from E3 VASP references. E3 is not authorized by the planning document.
- **E4, portability:** a fresh representative contract on Perlmutter, Frontier, and Aurora remains planned, not completed. A single successful contract per facility would establish a bounded portability demonstration, not comparative performance.
- **Supporting/proposed tests:** fresh scaling if that claim is retained; matched complete-loop DFT/MLIP substitution; failure-injection recovery; and local inference throughput/cost characterization require their own bounded designs and approvals. Do not substitute a proposal, interface test, mock, or deployment recipe for a measured scientific result.

## Source and Scope Notes

- Frozen artifact hashes and the current H digest are recorded in `research/paper/e1/packet-hashes.v1.json`; the current H digest and mode were independently checked for this note.
- Reviewer packet scores and leaves were inspected only to verify emptiness and packet structure. Response text and the protected crosswalk/key were not inspected.
- Historical manuscript numbers above are described only as manuscript-reported. Their underlying raw outputs, method provenance, failures, and scheduler/accounting records must be audited before classifying them as independently verified results or using them to support stronger claims.
- No protocol, D/H packet, response schema, scoring rubric, reviewer packet, or admin file was changed. The only artifact created for this review is this status note.