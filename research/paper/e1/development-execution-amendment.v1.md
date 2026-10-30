# E1 Development Execution-Configuration Amendment v1

Status: explicitly authorized execution configuration for subsequent E1
development requests. This amendment changes no scientific question, evidence
packet, response schema, scoring criterion, qualification threshold, or
adjudication rule in `development-protocol.v1.md`.

## Authorized Settings

- Maximum generated output: 32,768 tokens per request, subject to the selected
  model's advertised and operationally supported output limit. Record the
  model-specific capability in `candidate.max_output_tokens_supported` when
  known; lower the request limit only when that model requires it and record
  the limitation in its candidate manifest.
- Temperature: 0 where supported by the selected route. Do not silently
  substitute a different decoding setting.
- Retries: disabled. A retry or additional request requires separate approval
  and a manifest whose attempt and aggregate resource limits cover it.
- The MAG OpenAI-compatible adapter maps E1 `max_tokens` to request parameter
  `max_completion_tokens`. Other adapters use their provider-specific mapping;
  validate the effective request parameter before an authorized run.

Example decoding block for subsequent authorized frontier-remote and local
development manifests:

```json
{
  "temperature": 0.0,
  "max_tokens": 32768,
  "top_p": null,
  "seed": null
}
```

The 32,768 setting is an output allowance, not a requirement to consume that
many tokens, and is not itself resource authorization. Each remote campaign
still needs explicit route, data-use, retention, allocation or spending, and
request-limit authorization. Local campaigns still need candidate, facility,
queue, and compute authorization. When a model supports fewer than 32,768
output tokens, document its maximum and use only that supported cap under a
candidate-specific approved manifest.

## Historical Runs and Hashes

Do not edit or re-run completed campaigns to apply this amendment. The GPT-6
Luna original run remains an 18-request, 2,048-token first-attempt evaluation:
13 valid structured outputs and five truncated technical failures. Its
separately authorized recovery remains five one-attempt requests at 32,768
tokens, with five valid complete outputs. The recovery does not convert those
five questions into original first-attempt successes.

The completed run manifests retain their original protocol artifact hashes.
The original `development-protocol.v1.md` SHA-256 is
`a50ef7528acd1296e89864841f1aeac72570af415380aca90e4273feb925740d`.
The amendment has its own independently recorded SHA-256 in the E1 evaluation
record. It is execution-configuration provenance, not a new scientific
protocol hash.

For the two distinct technical-reliability summaries, report original
single-attempt completion as 13/18 and recovery-assisted coverage as 18/18
across 23 requests. Scientific competence is assessed only from blinded human
review of the 18 complete responses; schema validity and recovery completion
are not scientific scores.