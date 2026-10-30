# E1 Development Protocol v1

Status: proposed for scientific and operational approval. This artifact covers
individual development qualification only. It does not authorize model calls,
model downloads, or scheduler submissions, and it does not freeze held-out
procedures.

## Frozen Design Inputs

- Six materials; categories A (controlled comparison), B (evidence audit),
  and C (hypothesis generation); 18 development questions and 18 distinct
  held-out questions, with one question in each category per material and
  partition.
- The versioned D source is `d-questions.v1.json`. Each D question has exactly
  one response per candidate in this screen. Type B packets are explicitly
  hypothetical qualitative scenarios; they contain no fabricated numerical
  results. Type A and C prompts likewise receive no invented evidence.
- The shared response contract is `response-schema.v1.json`; rubric, provenance,
  reviewer blinding, and adjudication rules are in this protocol. Their hashes,
  along with the D and current H packet hashes, are recorded in
  `packet-hashes.v1.json`. Hashes must be recomputed and checked after any edit;
  this status is proposed, not a declaration of scientific approval.
- The current H source is stored outside the repository at
  `$HOME/.local/share/matsim-agents/e1/h-questions.v1.json`, with mode `0600`.
  Its SHA-256 is recorded in `packet-hashes.v1.json`. H prompt content was
  present in tracked history, so this copy is not eligible as an uncontaminated
  held-out set; replace and independently review it before any held-out use.
  The runner rejects H IDs and exposes no held-out input. Preserve owner-only
  access to the current copy and verify the replacement's access policy before
  any held-out workflow or transfer to another system.
- The four-condition held-out comparison design and intended local-versus-remote
  comparison remain as recorded in the study plan. Do not begin held-out A-D
  comparison until cohort selection and the complete held-out evaluation
  protocol are both frozen. This gate does not block approved individual D
  qualification.
- Candidate screening includes the nine debate-enabled catalog entries and
  Mixtral-8x22B-Instruct-v0.1 as a possible supplemental candidate at the user's
  request; screening all ten is not required to begin either cohort. Run only
  candidates that have an approved route, exact revision, and resource ceiling.
  Candidate identity, immutable revision, authorization, and serving route must
  be pinned separately for each run. A development panel of three is an output
  of qualification, not an input requirement.

## Response and Review

Use the same frozen D prompt, response schema, rubric, and provenance contract
for every candidate in both execution cohorts. E1-L (local) and E1-R (approved
remote API, including MAG only when selected and approved) are independent
queues: either can be executed without waiting for the other or for an Aurora
submission. Keep their manifests, output directories, execution ledgers, and
resource accounting separate. Retain the full raw provider message and
normalized/parsed response, question/candidate/cohort identity, manifest,
protocol and prompt hashes, finish status, token counts when supplied, latency,
and errors. The runner sends the shared JSON schema, uses temperature 0, and
allows at most 16,384 generated tokens. The base screen uses one attempt per
question. Retries are disabled by default; a remote manifest must explicitly
approve its maximum attempt count and include that count in its worst-case spend
ceiling before `--retry-errors` can use it. Record malformed or truncated
responses and score scientific quality separately from format compliance.

Two reviewers independently score each response. Reviewers receive randomized
response IDs and do not see model name, provider, endpoint, or serving condition.
They may see the question and its approved evidence packet. Reviewers must not
see H prompts or annotations. Preserve both raw score sheets. A third blinded
adjudicator resolves any criterion disagreement of at least two points or any
case where the two reviewers reach different qualification outcomes. The
adjudicator records the reason and final criterion scores; otherwise use the
mean of the two criterion scores. Report per-reviewer and adjudicated scores,
agreement, and all missing responses.

## Rubric and Qualification Rule

Retain the agreed criterion weights. Score each criterion from 0 through 4;
the weighted total is also on a 0-4 scale:

| Criterion | Weight | 0 | 1 | 2 | 3 | 4 |
|---|---:|---|---|---|---|---|
| Scientific correctness | 30% | Central claims contradict the prompt/evidence or basic constraints | Major unsupported or incorrect claims | Mixed correctness; a material error or scope overclaim remains | Claims are sound; only minor noncentral errors | All material claims are correct, scoped, and tied to the supplied evidence |
| Physical reasoning | 20% | No usable mechanism or causal logic | Mechanism labels without a coherent causal chain | Partly coherent chain with important assumptions omitted | Coherent mechanism and assumptions, with minor gaps | Distinguishes competing mechanisms and explains causal links and limits |
| Falsifiability and tests | 20% | No discriminating test | Test cannot separate the stated alternatives | Plausible test but outcomes or controls are incomplete | Feasible discriminating test and interpretable outcomes | Matched, feasible tests distinguish alternatives and state clear falsifiers |
| Evidence awareness and uncertainty | 20% | Treats absent evidence as fact or asserts unsupported certainty | Misses major gaps/conflicts or scope limits | Identifies some gaps but overstates at least one conclusion | Calibrated uncertainty and key evidence gaps; minor omission | Separates observation, inference, and hypothesis; all material gaps/scope limits are explicit |
| Computational feasibility | 10% | Proposed work cannot answer the question or is infeasible as stated | Major feasibility/control flaw | Potentially useful but underspecified or costly without justification | Feasible and adequately controlled, with minor omissions | Efficient, matched, reproducible plan with appropriate MLIP/DFT/experimental scope |

For Type B, score only against the hypothetical packet as written: do not reward
invented measurements and do not penalize a response for refusing to infer a
ranking from missing evidence. For Type C, accept multiple physically
defensible hypotheses; score their mechanism, predictions, tests, and
falsifiability rather than agreement with a single preferred mechanism.

An individual candidate qualifies only if all of the following hold:

- Mean adjudicated Scientific Correctness is at least 3.0/4 across D, and no
  material-level mean for it is below 2.5/4.
- Mean adjudicated Evidence Awareness and Uncertainty is at least 3.0/4 across
  D.
- Weighted overall mean is at least 3.0/4.
- No severe integrity failure occurs: a confidently asserted fabricated
  numerical result, fabricated source, or unsupported global-stability claim
  after the prompt explicitly limits the domain is a veto, regardless of the
  average. Reviewers must cite the text and rationale for any veto.
- At least 16 of 18 responses are reviewable. Missing, truncated, or malformed
  responses remain in the ledger; they are not silently replaced. A candidate
  below this completeness rule is operationally incomplete and does not qualify
  on the current screen.

These are proposed anchors and thresholds, not domain-approved gold answers.
Approve or revise them before any candidate response is inspected; do not change
them after scoring begins. If fewer than three candidates qualify, report that
outcome and stop panel selection. Do not lower thresholds to fill a panel.
Select and freeze any eventual panel only after individual qualifications using
D responses alone. Complementarity ranking is unnecessary for individual
qualification and may not use H content.

## Execution Routes and Budgets

The upper-bound base screen is 10 candidates x 18 D questions = 180 requests,
with 18 requests per candidate, temperature 0, no retries, and no replicates.
Ten is an upper bound, not a minimum; candidates may proceed independently when
their own route is approved. Run each candidate from a separate runner manifest
and output directory. Before
execution, validate that manifest with the installed checkout, freeze its hash,
record the exact model ID and immutable revision or checkpoint hashes, and
authorize the candidate-specific route. E1-R approval must bind provider,
upstream MAG provider when applicable, model ID, immutable revision, input/output
prices, input-token cap, attempt count, approver, approval reference, and a
numerical spending ceiling. Its conservative ceiling covers every question,
approved attempt, maximum input, and maximum output. E1-R writes a separate
`costs.jsonl` ledger with token-derived estimates and any provider-reported
charge; estimates are not represented as billed amounts. E1-L records no USD
charges and reports physical device-hours/node-hours separately. The fixture
`d-manifest.schema-fixture.v1.json` is for offline validation only and is not a
run manifest.

| Route | Development budget | Resource accounting and release gate |
|---|---|---|
| E1-L: local open-weight, vLLM on HPC | 18 requests per candidate; at most 16,384 generated tokens per response. Ten-candidate maximum is 180 requests and 2,949,120 generated tokens. No model downloads or queue submissions are included. | Record model/tokenizer snapshot or weight hashes, license/source, vLLM and runtime versions, physical devices, startup/load, serving and teardown walltime, token counts, throughput, failures, and scheduler accounting per candidate. Node-hours equal reserved nodes x scheduler walltime; GPU-hours use physical devices x allocated walltime. No defensible aggregate node-hour/GPU-hour cost exists until exact candidates and measured profiles are approved. The previously documented one-node/60-minute Mistral serving qualification is a separate operational qualification, not authorization for these 180 scientific responses. |
| E1-R: approved remote API, including MAG only if explicitly selected | 18 requests per candidate; at most 294,912 generated tokens per candidate under the runner cap. Ten-candidate maximum is 180 requests and 2,949,120 generated tokens before any separately approved retries. | Before any request, identify and approve the exact provider/model/revision/upstream route, confirm terms/data handling and credential source without recording secrets, verify account/balance, and authorize per-question input/attempt caps, current input/output rates, and an aggregate dollar ceiling. The runner rejects a worst-case estimate above that ceiling and records tokens, estimated and provider-reported charge, latency, errors, and attempt count per request. No endpoint probe or inference is included here. |

Do not treat the runner's 16,384-token maximum as a spending authorization. A
smaller max-token cap may be approved before the manifests are frozen if it
meets response-completeness needs; apply the same cap across candidates and
record it in each manifest. Any retries, model restarts requiring repeated
responses, or extra qualification questions require a separately approved
budget. Stop at the first candidate-specific resource ceiling or integrity
failure; do not automatically escalate nodes, walltime, or provider spend.

## Exact Readiness

Completed offline: D/H packet extraction and 18/18 count; six-material A/B/C
coverage in each partition; strict `E1Question` validation of all D questions;
H-ID rejection by the D question schema; owner-only storage and digest of the
current H copy (which is not eligible for held-out use due to tracked-history
exposure);
and six synthetic `tests/test_e1_development.py` tests for cohort routing,
manifest mismatch/partial-tail recovery, route-bound remote approval and spend
limits, raw/parsed response retention, separate cost records, and recorded
errors with opt-in retries. No model inference, endpoint probe, download, or E1
compute submission was performed.

Still required before D inference:

1. Two domain reviewers approve the exact D packet, evidence/provenance and
   hypothetical B framing, D/H scientific independence, rubric anchors, veto
   definition, and qualification thresholds.
2. For each candidate actually to be run, record exact immutable revision or
   checkpoint hashes, applicable license/terms, provider and serving route,
   operational qualification, data handling, and candidate-specific permission.
   Mixtral remains included in the ten-candidate screening scope but its
   user-reported operability does not establish any of these items.
3. Approve per-candidate resource ceilings. For HPC, obtain allocation-owner and
   facility confirmation of project, queue, balance, walltime, and chargeable
   resource plan, then separately authorize the exact job submission. For APIs,
   approve provider, token ceiling, dollar ceiling, and account/balance. No
   credentials belong in manifests or this repository.
4. Create each real candidate-bound manifest from the approved D packet, freeze
   hashes and output path, and have an authorized operator approve that exact
   manifest before invoking the runner.

The settled 36-question design, six materials, three categories, 18/18 split,
four reasoning conditions, and local-versus-remote comparison are not pending
design choices. Held-out panel-comparison procedures remain later gates; they do
not block D packet creation or offline validation.

## Cohort Independence and Held-Out Gate

E1-L and E1-R are independently executable development cohorts. Lack of an
Aurora allocation, queue slot, or local candidate readiness does not block an
otherwise approved E1-R run, and lack of an approved remote route does not
block an otherwise approved E1-L run. Both use identical frozen D questions,
response schema, rubric, and provenance rules, with distinct cohort-bound
manifests, ledgers, and output directories. Do not compare or select on H.

Do not begin any held-out A-D comparison until the local and remote development
cohort selection protocols and the complete held-out evaluation protocol are
frozen. The selected panel, comparison conditions, aggregation, call/token
accounting, response reuse and repeatability rules, and scoring/adjudication
procedure must all be fixed before H is made accessible. This restriction does
not prevent independently approved D-only cohort work.