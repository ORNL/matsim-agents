# E1 Campaign Operations Addendum

Date: 2026-10-30
Status: operational guidance only. This is not scientific approval, candidate or account authorization, a protocol amendment, an inference authorization, or a scheduler-submission authorization. The current readiness baseline is [e1-readiness-review-2026-10-30.md](e1-readiness-review-2026-10-30.md).

No inference, model download, or scheduler job was run for this addendum. The frozen protocol, D packet, response schema, rubric, reviewer packets, and manuscript were not changed. No response text, crosswalk, or provenance key is reproduced here.

## Current Status Update (2026-10-30 15:15 UTC)

This update supersedes the operational sequencing and scheduler snapshot below where they differ. Existing E1 campaign execution and resource approvals remain in force; this note does not add or revoke authorization. Collect answers from all eligible remote and local candidates before applying D scoring/qualification or selecting the debate panel. Reviewer assignment and conflict checks may proceed in parallel, but do not begin D scoring until collection is complete.

Aurora validation job `8990495.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov` was running at the time of this update against the existing pinned R1 manifest and resource plan, with startup tracing enabled. Its model log records `[torchcomms-patch] installed`; model-server initialization was still underway, and no candidate response was present yet. This validates patch loading only, not successful serving or E1-L completion. Refresh scheduler state and outputs before treating it as a result. The queue and candidate observations below remain historical snapshots.

## Immediate Decision (Original Snapshot)

The next action in this original snapshot was to assign and conflict-check two independent scientific reviewers, then distribute the two existing score packets separately through an approved restricted channel. Under the current sequencing update above, reviewer preparation may proceed in parallel, but D scoring and qualification wait for collection from all eligible remote and local candidates.

The first attempt produced 13 complete responses from 18 requests. Five separate recovery requests supplied the remaining responses. Describe this as 13/18 first-attempt completion and 18/18 recovery-assisted coverage across 23 requests, not 18/18 first-attempt completion.

## Scientific Scoring Checklist

Use the existing rubric and [reviewer-scoring-instructions.md](reviewer-scoring-instructions.md) unchanged. Reviewers should:

1. Score all 18 responses independently on correctness, physical reasoning, falsifiability/tests, evidence and uncertainty, and computational feasibility; provide a response citation and rationale for each score.
2. Judge only against the supplied question and its stated assumptions. Do not reward invented measurements, sources, or results; accept multiple defensible hypotheses for open-ended questions.
3. Record reviewability separately from scientific quality, with a reason. Flag only the defined serious integrity cases and cite the supporting text.
4. Give a provisional candidate-level outcome and rationale only after completing item scoring. Keep candidate/provider identity blinded.
5. Lock the completed form, retain its original bytes and hash, and do not compare judgments with the other reviewer before both forms are locked.

The qualification thresholds remain frozen: weighted overall, correctness, and evidence/uncertainty means each at least 3.0/4; no material-level correctness mean below 2.5/4; at least 16/18 reviewable; and no upheld integrity veto. These thresholds are not replaced by reviewer consensus or response completeness.

## Secure Reviewer Distribution and Import

Reviewer identities, conflicts, prior exposure, and distribution approval are human decisions and remain unassigned here. Before distribution, verify that both reviewers are independent, qualified for the materials, have no relevant conflict, and have not seen the responses or provenance key. Use a facility-approved restricted channel with separate reviewer-specific delivery and return locations. Do not send the crosswalk, reviewer key, the other reviewer's packet, or the other reviewer's scores. Avoid email attachments or shared folders unless their access, retention, and forwarding controls are approved.

The existing packets are owner-only locally: the directory is mode `2700`; each packet is `0600`; no group/other ACL grants were found. Their SHA-256 values are:

| Assignment | Packet | SHA-256 |
|---|---|---|
| reviewer-1 | `reviewer-1-packet.json` | `1a2a23c681dcacdb7778edaac82b2ad9efe122212d2856df02881934c2060c1b` |
| reviewer-2 | `reviewer-2-packet.json` | `64038f07bde66de0dd7974fb093bf91d7d557e7a446d922e00034452feedf934` |

Packets are ignored under `runs/` and were not found tracked or in reachable repository history. This is not a substitute for controlled delivery or a review of response content for identity clues. Do not place any packet copy in a public or group-readable location.

For each approved recipient, make a separate copy into that recipient's restricted drop; never put both packets in a recipient-visible directory. For example, after the recipient-specific destination has been approved and created:

```bash
umask 077
install -m 600 \
  runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/reviewer-1-packet.json \
  "$REVIEWER_1_RESTRICTED_DROP/packet.json"
sha256sum "$REVIEWER_1_RESTRICTED_DROP/packet.json"
```

Use the corresponding reviewer-2 source and separate destination for the second delivery. Verify destination ownership, mode, ACL, and returned-file hash. Each reviewer returns one completed form only to the restricted operator intake. Preserve an untouched copy of each returned form; do not normalize, reformat, or repair a reviewer's source file in place.

After both forms are received, use a private output directory and restrictive umask. The command below is the initial validation/import only; it does not qualify the candidate. Point `FORM1` and `FORM2` to the completed returned forms, not to the blank packet templates:

```bash
umask 077
PYTHONPATH=src .venv/bin/python scripts/e1/import_review_scores.py \
  "$FORM1" "$FORM2" \
  --output "$PRIVATE_IMPORT_DIR"
```

The importer checks distinct reviewer IDs, all 18 unique response IDs, complete criteria and rationales, valid scores, matching frozen hashes, reviewability, and integrity flags. It creates an adjudication template when required. An independent blinded third reviewer must adjudicate any criterion difference of at least two points, reviewability disagreement, integrity flag, or differing qualification outcome. Do not expose the crosswalk/key before both initial forms and any required adjudication are locked.

Only after the required third-review form is complete, run the second stage with the same two original forms and the completed adjudication form:

```bash
umask 077
PYTHONPATH=src .venv/bin/python scripts/e1/import_review_scores.py \
  "$FORM1" "$FORM2" \
  --output "$PRIVATE_IMPORT_DIR" \
  --adjudication "$ADJUDICATION_FORM"
```

Confirm the original forms remain unchanged, inspect the final `qualification-report.json`, and retain all source forms, adjudication, import, and report under owner-only access. The importer test suite passed: `4 passed`. No human forms are present, and no import or qualification report has been produced.

## Replace Contaminated H Before Any Held-Out Access

The current H set is ineligible as an uncontaminated holdout: all 18 prompts appeared in tracked repository history. Runner non-use does not reverse that exposure. Keep the current owner-only copy and its digest for audit only; do not use, quote, paraphrase, or score it as held out.

Replacement sequence:

1. Appoint question authors who have not accessed the exposed H prompt set or its crosswalk. Keep the replacement content out of Git, issue trackers, shared logs, and reviewer packets from the moment of authorship.
2. Author a genuinely new 18-question set matching the frozen six-material by three-task design. Do not use paraphrases or merely change names/numbers in exposed items. Maintain a restricted provenance log identifying authors, source evidence, and revisions.
3. Have two independent domain reviewers approve scientific validity, supplied evidence, assumptions, answerability, scope limits, task-category fit, and scoring anchors without seeing model outputs. Check for overlap/leakage against D, all exposed H material, examples, and public/tracked study text using controlled access; investigate semantic near-duplicates, not only exact string matches.
4. Obtain explicit author approval for any H-protocol amendment required by replacement. Freeze the approved prompt packet, execution prompts, schema, rubric, panel-selection rules, and comparison analysis before model selection or held-out inference. Record hashes in a new versioned register; do not overwrite v1 artifacts.
5. Store the replacement outside tracked history with owner-only files and directory access, verify ACLs and backup/sync behavior, and restrict access to the designated data custodian and authorized evaluators. Keep D-only tooling from reading H. Release H only at the predeclared held-out evaluation stage after panel/protocol freeze.

No replacement prompts are authored in this addendum. A new H set, its independent review, protocol approval, and a full held-out freeze remain prerequisites to any held-out call.

## Candidate Readiness Inventory

`Not assessed` means the repository/catalog evidence does not establish the property. A model directory or complete shard index is only filesystem presence, not proof of checkpoint provenance, licensing, usable memory/parallelism, serving compatibility, authorization, or scientific ability. No candidate below is dispatchable on this inventory alone.

### Open-Weight Candidates

The catalog contains nine debate-enabled entries; Mixtral is a supplemental candidate. Local inspection checked expected checkpoint filenames and shard counts only, not weight contents or model behavior.

| Candidate | Local checkpoint evidence | Immutable revision / license / architecture details | Serving, resources, authorization, competence | Status |
|---|---|---|---|---|
| `moonshotai/Kimi-K2.5` | No matching local directory observed | Not assessed | Not assessed | Blocked; pin and verify all gates |
| `zai-org/GLM-4.7` | No matching local directory observed | Not assessed | Not assessed | Blocked; pin and verify all gates |
| `zai-org/GLM-4.7-Flash` | Directory with config, index, and 48 safetensors shards observed | Commit/hash and license terms not verified | vLLM compatibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `deepseek-ai/DeepSeek-V3.2` | Directory with config, index, and 163 shards observed | Commit/hash and license terms not verified | Serving feasibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Directory with config, index, and 118 shards observed | Commit/hash and license terms not verified | Serving feasibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `Qwen/Qwen3-235B-A22B-Thinking-2507` | Directory with config, index, and 118 shards observed | Commit/hash and license terms not verified | Serving feasibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `google/gemma-4-31B-it` | Directory with config, index, and 2 shards observed | Commit/hash and license terms not verified | Serving feasibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `google/gemma-4-26B-A4B-it` | Directory with config, index, and 2 shards observed | Commit/hash and license terms not verified | Serving feasibility, memory/parallelism, account/resource approval, scientific performance not assessed | Blocked; local files are not qualification |
| `mistralai/Devstral-2-123B-Instruct-2512` | No matching local directory observed | Not assessed | Not assessed | Blocked; pin and verify all gates |
| Supplemental `mistralai/Mixtral-8x22B-Instruct-v0.1` | No matching local directory observed | No immutable revision or local checkpoint confirmed | Serving, resource, license, authorization, and competence not assessed | Blocked; optional, not required to start an independently approved cohort |

Before any local candidate manifest, record the exact source revision and local snapshot/weight hashes, terms, model architecture and precision, tokenizer/template, serving stack/runtime versions, parser/schema behavior, device map and parallelism, measured startup/throughput/peak-memory profile, output cap, retry/attempt limit, and a candidate-bound project/resource approval. The Aurora downloader supports a full 40-character Hub commit through `MODEL_REVISION` for exactly one model per setting; an omitted revision downloads a moving default. No download was initiated.

### MAG Candidates

The 2026-10-30 catalog snapshot verifies account access to `/models` and the following catalog entries/advertised limits. It does not establish immutable serving revisions, effective account limits, route behavior, project approval, data-use/retention acceptance, or spending authority. No inference request was sent.

| MAG catalog ID | Upstream route | Catalog evidence | Missing before dispatch | Status |
|---|---|---|---|---|
| `claude-sonnet-5-5@amazon` | Amazon | Chat; advertises 1,000,000 input / 128,000 output tokens | Immutable-version semantics, route/model-bound allocation approval, effective token behavior, data handling, spend ceiling | Blocked |
| `gemini-4-argon@google` | Google | Chat; advertises 1,048,576 input / 1,048,576 output tokens | Same gates; no project/spending authorization or model-version semantics recorded | Blocked; catalog snapshot names it as a possible next candidate, not an authorized run |
| `gemini-3.8-flash@google` | Google | Chat; advertises 1,048,576 input / 65,536 output tokens | Same gates | Blocked |
| `gpt-6-astra@openai` | OpenAI | Chat; advertises 922,000 input / 128,000 output tokens | Same gates | Blocked |

MAG approval must bind the exact model ID and accepted revision semantics, upstream provider, endpoint, project authorization, data-use and retention terms, input/output/attempt caps, and numerical aggregate spend ceiling. Catalog visibility or prior account authentication is not inference authorization. The existing GPT-6 Luna response set remains awaiting human scoring; its results cannot be inferred from the model name, API success, or response completeness.

## Aurora Resource Route

The current E1 development job template requests one node for 90 minutes. The `debug` queue is limited to 60 minutes and is incompatible with that template. The enabled `capacity` queue advertises a 168-hour maximum, 16-node maximum, five queued jobs and two running jobs per user; the queue snapshot showed substantial overall activity. `qstat -u "$USER"` returned no listed jobs during this check. Queue configuration does not establish this project’s allocation balance, user/group eligibility, chargeability, approval, or availability at submission time. Scheduler defaults explicitly permit account-check failure and negative allocation, so scheduler acceptance is not evidence of charge authorization.

The `lustre_scaling` queue remains disabled and ACL-restricted. Do not route through it. The safe next Aurora step is an allocation-owner/facility confirmation for this project, queue, 90-minute one-node request, account/balance, current eligibility, and exact permitted workload. Only after that confirmation, and separate scientific/candidate approval, may an exact manifest and resource plan be authorized. This addendum does not change the PBS script, submit a job, or authorize an inference.

## Outcomes: Correctness, Usefulness, and Debate Benefit

The frozen five-criterion rubric assesses scientific correctness and several contributors to practical usefulness, but it does not directly score novelty, relevance to the scientific objective, decision value, or the expected value of the proposed next calculation. Do not add a sixth criterion, reinterpret existing scores as a direct usefulness score, or retroactively rescore the current 18 responses. If a direct usefulness endpoint is needed, obtain prospective approval for a separate, explicitly versioned measure before the next candidate responses are seen. It should distinguish objective relevance, decision/actionability, information gained by the next test, feasibility/cost, and whether the recommendation changes a scientific decision; report it separately from correctness.

Debate benefit is a paired causal comparison, not a property of a persuasive transcript. On uncontaminated held-out questions, compare the same qualified panel under the frozen strongest-single, deterministic independent aggregation, private self-revision, and structured-debate conditions. Use paired question-level correctness and usefulness outcomes, and separately count errors corrected, valid mechanisms/tests added, errors introduced, useful disagreements resolved, and unsupported consensus. Measure actual calls, tokens, latency, and allocated resources; disclose any remaining mismatch between self-revision and debate. Agreement, novelty, longer answers, or more calls alone do not demonstrate debate benefit.

The manuscript’s reduced-human-supervision statement is currently an aim, not a measured E1 result. To support it, prospectively define the human intervention unit and compare a fixed-quality baseline with the agent workflow on matched scientific tasks. Record active expert minutes, number/type of interventions and overrides, review/adjudication time, time to a justified decision, and scientific decision error or unresolved rate. Predefine an acceptable quality/non-inferiority boundary; lower intervention counts without that boundary could simply mean less oversight. No such baseline or time-and-intervention data are present in this campaign review, so no quantified supervision reduction is supported.

## E1-E4 and Manuscript Contribution Fit

| Manuscript contribution | Relevant experiment(s) | Current evidence boundary / next evidence needed |
|---|---|---|
| Scalability | E1 local-serving throughput/concurrency is a supporting measure; separate DFT/MLIP scaling studies support numerical scalability claims | No new local-LLM scaling data are available here. Reconcile any historical manuscript VASP/active-learning figures against raw outputs, scheduler records, methods, and allocation accounting before reusing them. A single successful workload is not a scaling curve. |
| Portability across Frontier, Aurora, and Perlmutter | E4 | Deployment recipes and engineering smoke tests establish software-path evidence only. E4 still requires the same frozen, bounded scientific contract and verified outputs at all three facilities; report a bounded portability demonstration, not comparative performance. |
| Backend flexibility across LLM, MLIP, and DFT | E1 candidate qualification; E3 zero-shot and conditional adaptive MLIP assessment; separate matched QE/VASP substitution if retained | Interfaces/configurations alone do not establish scientific interchangeability. E3 needs fresh method-consistent VASP evidence and independent splits for supported cells; DFT backend substitution needs matched full-loop science, settings, and provenance. |
| Local LLM inference | E1-L compared with approved E1-R under the frozen scientific protocol | No local E1 scientific responses or measured serving profile are established by the observed model directories. Pin and approve a checkpoint, qualify runtime/output parsing, run the approved D cohort, and report physical resource use before claiming local inference capability or local/remote parity. |

E1 is not complete when the current review is finished: local qualification, approved candidate screening, panel freeze, an uncontaminated H replacement, and the held-out conditions remain separate gates. E2 fixed-evidence assessment, E3 fresh-reference MLIP evaluation/adaptation, and E4 cross-facility scientific execution remain planned, not completed. Do not promote a planned experiment, software capability, reviewer score, or catalog observation into a result.

## Action Order

1. User selects/conflict-checks two reviewers and approves restricted distribution; reviewers complete and lock independent forms.
2. Import the two forms privately; assign an independent third adjudicator if any specified trigger occurs; issue the qualification report only after resolution.
3. In parallel, request explicit Aurora project/queue/allocation confirmation and decide which candidate route can be authorized; do not download or dispatch on the basis of this inventory.
4. Assign unexposed H authors and two domain reviewers; produce and freeze a replacement under a new version without altering v1 artifacts.
5. Prospectively approve any usefulness/supervision endpoints and full E1 held-out analysis before candidate outputs or H are accessed.
6. Freeze the eligible panel and matched resource accounting; only then consider separately authorized held-out execution. Keep E2-E4 in their own evidence and approval gates.