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
participants: [...] 
```

- `hypothesis` is the exact question or claim assigned to every model.
- `rounds` is the number of complete panel passes.
- `output_root` receives a uniquely named, provenance-tracked run directory.
- `max_transcript_chars` bounds context sent to a model. It does not truncate
  the dialogue persisted to disk.
- `synthesis_participant` is forbidden with `independent_verdicts`, must match
  a participant name with `designated_model`, and otherwise defaults to the
  first participant.

## Outcomes and artifacts

Every successful run contains:

- `dialogue.json`: chronological dialogue beginning with the assigned
  hypothesis, followed by every argument and final verdict;
- `debate_transcript.json`: structured turns, verdicts, and aggregate output;
- `results.json`: typed workflow result and artifact paths;
- resolved configuration and provenance files managed by the scientific run
  directory.

Every model argument and verdict has a unique contribution ID. Argument IDs
encode their round and turn; verdict IDs identify the independent or designated
conclusion. Records include participant, provider, model, and complete text.

All debate output has `hypothesis` evidence level. Agreement between many LLMs
is not physical validation. Candidate materials, mechanisms, and property
claims must still be checked against databases, MLIP/DFT calculations, and
experiments.

## All-model portability modality

Run `benchmarks/portability/all_model_scientific_debate.py` to require every first-class model
in `deployments/common/open-model-catalog.json`. It always uses equal debate,
independent verdicts, and at least two rounds. It fails for missing endpoints,
missing or empty turns/verdicts, incomplete model coverage, or duplicate
contribution IDs. See the
[portability benchmark guide](../benchmarks/portability/README.md#all-model-scientific-debate-qualification).
