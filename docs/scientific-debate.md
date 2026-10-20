# Scientific hypothesis debate

`matsim-agents debate` lets two or more LLMs iteratively examine one scientific
hypothesis. Debate behavior has two independent choices:

1. **Participation mode** controls whether models are equals or have roles.
2. **Conclusion method** controls whether every model gives a verdict or one
   model synthesizes the discussion.

This produces four supported modalities:

| Participation | Conclusion | Configuration | Outcome |
|---|---|---|---|
| Equal | Independent verdicts | `equal` + `independent_verdicts` | One equally weighted final verdict per model; recommended default |
| Equal | Designated synthesis | `equal` + `designated_model` | Neutral debate followed by one named model's synthesis |
| Role-based | Independent verdicts | `role_based` + `independent_verdicts` | Specialized perspectives plus one final verdict per specialist |
| Role-based | Designated synthesis | `role_based` + `designated_model` | Specialized panel followed by one named chair/synthesizer |

The debate is round-robin and sequential within a round. A participant sees all
completed earlier rounds and participants who have already spoken in its current
round. Speaking order rotates every round to distribute the first/last-speaker
advantage. `rounds` accepts 1–100; portability qualification requires at least
2. At least two participants with unique names are required.

## Common participant configuration

Every participant declares:

```yaml
- name: model_a                 # unique dialogue identity
  provider: vllm               # ollama | vllm | openai | anthropic | huggingface
  model: org/model-a            # provider-specific model ID
  base_url: http://node:8000/v1 # optional provider endpoint
  role: independent reviewer    # used only by role_based mode
```

Credentials and provider-specific environment variables follow the normal
[LLM provider configuration](../README.md#llm-provider-configuration). An
endpoint must be reachable from the process running the debate. Participant
names label dialogue contributions; they do not imply rank or authority.

Run any configuration with:

```bash
matsim-agents debate examples/debate/equal-independent.yaml
```

### Participants served from a resource plan

A participant may name a `server` from the `llm` pool of a `resources:` block
instead of giving `provider`, `model`, and `base_url`
([resource dispatch](resource-dispatch.md)). The debate then starts those
servers on the GPUs the plan assigns, waits until each lists its model, and
stops them afterwards; see
[`equal-independent-plan-servers.yaml`](../examples/debate/equal-independent-plan-servers.yaml).
Without a `resources:` block, server names refer to the plan served by
`matsim-agents resources serve PLAN -- matsim-agents debate CONFIG`, which
lets several commands share the same running servers.

The checked-in `org/model-*` identifiers and `node-*` URLs are placeholders.
Replace them with deployed model IDs and endpoints before execution. Different
participants may use different supported providers; they do not need to share
one server or model family.

## 1. Equal debate with independent verdicts

Use this when models should have identical authority and no model should frame
the panel's conclusion.

```yaml
debate_mode: equal
synthesis_method: independent_verdicts
```

In equal mode, `role` is deliberately ignored and every model receives the
same neutral system instruction. After the debate, every participant receives
the complete transcript and writes an independent verdict. The `synthesis`
field is a deterministic side-by-side presentation of those verdicts—not a new
LLM-generated consensus. This is the default and the modality used by the
all-model portability benchmark. Complete example:
[`equal-independent.yaml`](../examples/debate/equal-independent.yaml).

## 2. Equal debate with designated synthesis

Use this when the discussion must remain neutral but one model must prepare a
single readable conclusion:

```yaml
debate_mode: equal
synthesis_method: designated_model
synthesis_participant: model_a
```

All round prompts remain identical. Only after the rounds does `model_a`
receive the transcript and produce the sole final verdict. If
`synthesis_participant` is omitted, the first configured participant is used.
This modality introduces asymmetry in the conclusion and should not be called
an equal final decision. Complete example:
[`equal-designated.yaml`](../examples/debate/equal-designated.yaml).

## 3. Role-based debate with independent verdicts

Use this to deliberately cover complementary scientific perspectives without
giving any specialist control over the conclusion:

```yaml
debate_mode: role_based
synthesis_method: independent_verdicts
```

Each participant's `role` is inserted into its system prompt. For example, a
transport theorist can focus on electronic/phonon mechanisms while an
experimentalist focuses on synthesis and characterization. Every specialist
still writes an independent verdict. Complete example:
[`role-based-independent.yaml`](../examples/debate/role-based-independent.yaml).

## 4. Role-based debate with designated synthesis

Use this for a conventional panel with specialist reviewers and an explicitly
named chair:

```yaml
debate_mode: role_based
synthesis_method: designated_model
synthesis_participant: chair
```

The role-specific prompts affect debate turns, and the chair alone produces the
final verdict. The chair is not treated as scientifically more correct; it is
only responsible for presentation. Complete example:
[`role-based-designated.yaml`](../examples/debate/role-based-designated.yaml).

## Shared controls

```yaml
hypothesis: "What candidate material provides high thermoelectric ZT near 800 K?"
rounds: 2
output_root: ./runs
max_transcript_chars: 60000
max_completion_tokens: 8192
participants: [...] 
```

- `hypothesis` is the exact question or claim assigned to every model.
- `rounds` is the number of complete panel passes.
- `output_root` receives a uniquely named, provenance-tracked run directory.
- `max_transcript_chars` bounds the earlier answers shown to a model in one
  prompt. Contributions are kept whole, newest first; the oldest ones that do
  not fit are left out and named in the prompt and in the contribution's
  `omitted_context`. The dialogue on disk is never shortened.
- `max_completion_tokens` (default 8192) bounds one reply, including a
  reasoning model's reasoning; a participant's own `max_completion_tokens`
  overrides it, and `null` uses the provider's default.

Peers are shown each other's answers only, quoted between `<contribution>`
tags, never their reasoning. A reasoning model's reasoning is separated from
its answer whether the server returns it separately (vLLM with
`--reasoning-parser`) or before `</think>` in the reply.
- `synthesis_participant` is forbidden with `independent_verdicts`, must match
  a participant name with `designated_model`, and otherwise defaults to the
  first participant.

## Outcomes and artifacts

A run directory is named `debate-<UTC timestamp>_<suffix>` and contains:

- `transcript.md`: the debate for reading, one section per argument and
  verdict, with the reasoning folded away and incomplete replies marked;
- `contributions.jsonl`: one record per argument and verdict, appended as soon
  as it is made, so a crash or walltime cutoff loses at most the reply in
  progress. Each record has the contribution `id`, `kind` (`argument`,
  `verdict`, or `synthesis`), round, speaker, provider, model, `text` (the
  answer), `reasoning`, `finish_reason`, `truncated`, prompt and completion
  token counts, latency, and the earlier contributions the model was shown
  (`context`) or not shown (`omitted_context`);
- `prompts/<id>.json`: the exact messages sent for each contribution;
- `result.json`: status, contribution ids, and `incomplete_contributions`,
  the replies the token limit cut off or that contain no answer;
- `request.json`, `resolved_config.json`, `provenance.json`, and
  `events.jsonl`, managed by the scientific run directory.

Argument IDs encode their round and turn; verdict IDs identify the independent
or designated conclusion. A reply cut off at the token limit is logged as a
warning, marked in the transcript, and shown to later speakers with a note;
raise `max_completion_tokens` for that participant when it happens.

All debate output has `hypothesis` evidence level. Agreement between many LLMs
is not physical validation. Candidate materials, mechanisms, and property
claims must still be checked against databases, MLIP/DFT calculations, and
experiments.

## Using debate inside the agentic materials workflow

Debate can replace the single-LLM hypothesis-discussion stage without becoming
part of active learning. Configure `InvestigationConfig.hypothesis_discussion`:

```yaml
hypothesis_discussion:
  mode: multi_llm_debate       # or single_llm
  multi_llm_debate:
    rounds: 2
    debate_mode: equal
    synthesis_method: independent_verdicts
    participants:
      - name: model_a
        provider: vllm
        model: org/model-a
        base_url: http://node-a:8000/v1
      - name: model_b
        provider: vllm
        model: org/model-b
        base_url: http://node-b:8000/v1
```

The workflow requires each final verdict to return a structured, testable
hypothesis, proposed chemical compositions, rationale, and assumptions. With
independent verdicts, compositions are merged in participant order and
duplicates are removed; hypotheses, rationales, assumptions, contributor IDs,
and disagreements retain their model attribution. The resulting
`ScientificHypothesis` is then consumed by phase exploration exactly like a
single-LLM hypothesis.

Phase exploration may subsequently invoke active learning when its existing
policy and uncertainty criteria require it. Debate is not run inside the AL
loop, after AL iterations, or as an AL decision maker. Complete investigation
configurations are provided in
[`examples/investigation`](../examples/investigation/).

## All-model portability modality

Run `benchmarks/portability/all_model_scientific_debate.py` to require every
first-class catalog model whose checkpoint exists under `--models-root` (or
`MODEL_ROOT`) and is not temporarily disabled by catalog policy. It always uses
equal debate, independent verdicts, and at least two rounds. It fails for fewer
than two selected models, missing endpoints for selected models, missing or
empty turns/verdicts, incomplete selected-model coverage, or duplicate
contribution IDs. See the
[portability benchmark guide](../benchmarks/portability/README.md#all-model-scientific-debate-qualification).
