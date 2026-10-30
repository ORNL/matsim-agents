# E1 Blinded Scientific Scoring Instructions

Each reviewer receives one packet containing all 18 complete responses, in an
independently randomized order. The stable anonymous response IDs are shared
between reviewer packets. Do not request or inspect the protected provenance
crosswalk. Do not discuss scores or candidate-level judgments with the other
reviewer until both score forms are complete and locked.

For every response, score all five criteria from 0 through 4 using the anchors
embedded in the packet. Enter a concise rationale with a quotation or precise
location in the response for each criterion. Mark each response reviewable or
not reviewable and explain that operational judgment; judge response format
separately from scientific quality. Judge against only the supplied
question and its stated assumptions/evidence. The question text is the complete
supplied evidence packet; no external evidence is attached. Do not reward
invented measurements, sources, or results. Multiple physically defensible
hypotheses are acceptable for open-ended prompts; assess mechanism, predictions,
tests, and falsifiability rather than agreement with a preferred hypothesis.

Record a serious-integrity flag only when warranted, with the response text
and reason: confidently fabricated numerical results, fabricated sources, or
an unsupported global-stability claim where the prompt explicitly limits the
domain. Do not treat missing data, cautious abstention, or technical history
as a scientific error.

After item scoring, record your provisional candidate-level qualification
outcome as `qualifies` or `does_not_qualify`, and include a concise rationale.
The existing thresholds are reproduced in each form. At least 16 of
18 responses must be reviewable; this set contains 18 complete responses.
Preserve your completed JSON score form without rewriting its original scores.

## Import and Adjudication

After both reviewers lock their forms, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/e1/import_review_scores.py \
  runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/reviewer-1-packet.json \
  runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/reviewer-2-packet.json \
  --output runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/import
```

The importer validates complete coverage, score ranges, rationales, and
reviewer independence; it preserves both submitted forms and creates an
adjudication queue. A third blinded adjudicator must resolve any criterion
score difference of at least two points, reviewability disagreement, integrity
flag, or different reviewer qualification outcomes. The initial import writes
`adjudication-form.template.json` when a third review is required. The
adjudicator fills every listed resolution with a score/disposition, response
citation where requested, and rationale, and records a candidate-level outcome
only when the reviewers disagreed. Then rerun the importer with `--adjudication`:

```bash
PYTHONPATH=src .venv/bin/python scripts/e1/import_review_scores.py \
  runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/reviewer-1-packet.json \
  runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/reviewer-2-packet.json \
  --output runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/import \
  --adjudication runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/adjudication-form.json
```

For non-disputed criteria use the mean of the two reviewer scores. The second
stage verifies the original score forms and adjudication template against the
locked copies, leaves `review-import.json` unchanged, and writes
`qualification-report.json`. Final adjudicated scores and the fixed
qualification-threshold checks appear there only after all required
adjudication is complete. Do not expose the provenance crosswalk to reviewers
or adjudicators until score sheets and adjudication are locked.

Retain reviewer forms and adjudication records as raw evaluation artifacts.
Use `research/paper/e1/qualification-report.template.v1.json` as the report
shape; the importer emits the populated final report as
`qualification-report.json`. Report individual and adjudicated scores,
reviewer agreement, qualification threshold checks, integrity flags, and
missing responses. Do not calculate or claim scientific qualification from
schema validity alone.