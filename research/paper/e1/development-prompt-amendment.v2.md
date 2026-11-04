# E1 Development Prompt Amendment v2

Status: authorized D-prompt wording change for all subsequent E1 development
requests. It changes no question, evidence packet, response schema, decoding
setting, scoring criterion, qualification threshold, or adjudication rule in
`development-protocol.v1.md` or `development-execution-amendment.v1.md`.

## Change

The v1 system prompt listed the ten response keys with JSON type labels, for
example `- conclusion: string` and `- hypotheses: array of string values`.
Under temperature-0 decoding, Qwen3-235B-A22B-Instruct-2507 copied those labels
verbatim as values: 18/18 responses in job `9036469` were schema-valid but
contained only placeholders. JSON-schema-constrained decoding enforces types,
not content, so these passed format validation.

The v2 prompt (`prompt_variant: "v2-field-guidance"`) keeps the same opening
instructions and the same ten keys, but describes what each key must contain
and states that type names, key names, and copied list text are not valid
values.

| Variant | Manifest field | System-prompt SHA-256 |
|---|---|---|
| v1 | absent (archived runs) | `21cb5454f4f0a9780519abc84eabad5dba9d8c4d317cd15540cac32af971b314` |
| v2 | `"prompt_variant": "v2-field-guidance"` | `ef7a981bdc2cf948c0c3251efbb87a539c8901f48cbe7a55d310157626c95038` |

## Validation

Every E1 record now reports four separate outcomes: `technical_completion`,
`schema_compliance`, `semantic_validity`, and `scientific_reviewability`. A
schema-valid response whose values are type names, key names, or prompt text
is recorded as `status: "error"` and `nonreviewable_semantic_output`. Review
packets are not prepared from runs containing such responses. This is a format
check only; it assigns no scientific-quality score.

## Comparability and Historical Runs

Archived v1 runs are preserved unmodified as the original first-attempt
record, with their original protocol and prompt hashes. They are not
reclassified in place; `scripts/e1/audit_semantic_validity.py` writes a
separate `semantic-audit.json` sidecar. v1 and v2 results are not pooled.

The nine non-Qwen candidates completed under v1 produced no placeholder-only
responses, so their v1 results are retained and they are not rerun. A v2
rerun of these candidates was started and stopped; only `gemma-4@cborg`
completed its 18 v2 responses, kept separately as supplementary data. Any
comparison between a v1 candidate and a v2 candidate must disclose the
prompt difference.

## Qwen3-235B Serving Fault

The placeholder responses were a symptom, not the root cause. Every
Qwen3-235B run on Aurora (Instruct and Thinking, eager and XPU-graph, v1 and
v2) reports that its input is garbled. A controlled test with
Mistral-Small-24B on the same vLLM stack gave correct answers on one tile but
garbled answers with 8-way tensor parallelism, also with
`CCL_ZE_IPC_EXCHANGE=sockets`. All Qwen3-235B E1 responses collected so far
are therefore invalid and are excluded from analysis, whatever their recorded
schema or semantic status. Qwen E1 collection resumes only after a serving
configuration passes the same fixed-prompt check.
