# Genesis Mission Model Access Gateway (MAG)

The Genesis Mission Model Access Gateway (MAG) provides an OpenAI-compatible
API for approved projects to access hosted language and embedding models. This
page records the basic MAG workflow for `matsim-agents` users. MAG availability,
model catalog, prices, and policies can change; follow the official
[MAG documentation](https://amsc-docs-d762d2.gitlab.io/model-access-gateway/)
and [getting-started guide](https://amsc-docs-d762d2.gitlab.io/mag-get-started/)
for current instructions.

## Access and credentials

MAG is not available to every Genesis Mission account by default. The project
must be approved for MAG and have an allocation; confirm access with the
project PI or allocation owner before sending requests. MAG usage is charged to
the project's allocation.

Create a Personal Access Token (PAT) in the myAmSC dashboard according to the
[official onboarding instructions](https://amsc-docs-d762d2.gitlab.io/mag-get-started/).
The PAT is a bearer credential. Set it in the client environment without
printing or committing its value:

```bash
export AMSC_TOKEN="<your-MAG-PAT>"
```

Do not put a PAT in source code, manifests, logs, or command history. Treat it
as a secret, rotate it if it is exposed, and generate a replacement when it
expires. A `401 Unauthorized` response can indicate an expired or invalid PAT.

## Endpoint and model catalog

The API root is:

```text
https://mag.genesis.americansciencecloud.org/v1
```

List the models currently available to the authenticated account with a
read-only request:

```bash
curl --fail-with-body --silent --show-error \
  --url https://mag.genesis.americansciencecloud.org/v1/models \
  --header "Authorization: Bearer ${AMSC_TOKEN}"
```

Use the live catalog as the source of truth for available model IDs. The
[MAG dashboard](https://mag.genesis.americansciencecloud.org/dashboard/)
also provides model and usage information. Never infer account availability
from a model name shown in an example or an older catalog snapshot.

## Chat completion example

MAG accepts the OpenAI chat-completions request shape. Select a provider with
`X-AmSC-MAG-Provider` when required or when a specific provider is part of the
experiment; alternatively, use the `@provider` model suffix supported by MAG.
The following is an illustrative request; replace the example model with an ID
visible in your account's catalog:

```bash
curl --fail-with-body --silent --show-error \
  --url https://mag.genesis.americansciencecloud.org/v1/chat/completions \
  --header "Authorization: Bearer ${AMSC_TOKEN}" \
  --header "Content-Type: application/json" \
  --header "X-AmSC-MAG-Provider: openai" \
  --data '{
    "model": "<catalog-model-id>",
    "messages": [{"role": "user", "content": "Hello."}],
    "temperature": 0
  }'
```

MAG model names may include supported reasoning-effort, service-tier, or
compression modifiers. Valid modifiers depend on the model; an unsupported
modifier returns a client error. See the official
[model selection reference](https://amsc-docs-d762d2.gitlab.io/model-names/).
Responses include `X-AmSC-MAG-Applied`, which records the provider and other
MAG options applied to the request; preserve it with experiment provenance.

## Rate limits and responsible use

The documented per-user limits are 20 concurrent requests and 120 requests per
minute. Streaming requests count as concurrent until they finish. A `429 Too
Many Requests` response includes rate-limit information such as `retry-after`;
use bounded exponential backoff with jitter and stop after a fixed number of
attempts. Do not evade limits by distributing one user's workload across
multiple projects or credentials. See the official
[rate-limit guidance](https://amsc-docs-d762d2.gitlab.io/mag-rate-limits/).

Before running a scientific evaluation, separately approve its protocol,
candidate/model revisions, prompt partition, call/token budget, allocation
ceiling, retries, data handling, and result provenance. API reachability or a
successful model-catalog request does not authorize inference or establish
scientific readiness. Keep development and held-out data access separate as
specified by the study protocol.

## Facility network setup

On ALCF and OLCF systems, external API calls may require the facility proxy
environment configured in the client shell. See
[LLM readiness and portability qualification](llm-readiness.md#external-api-access-from-alcf-and-olcf)
for the recorded proxy profiles and the OLCF endpoint verification caveat.
Network access can differ between login and compute nodes; follow facility and
project policy.

## `matsim-agents` integration

Install the MAG client extra and select the dedicated provider:

```bash
pip install -e '.[mag]'
export AMSC_TOKEN="<your-rotated-MAG-PAT>"
matsim-agents llm-check <mag-check.yaml>
```

The provider is `mag`; it uses the MAG endpoint by default and requires an
explicit model ID from the authenticated `/v1/models` catalog. Set
`MATSIM_MAG_BASE_URL` to override the endpoint or `MATSIM_MAG_PROVIDER` to set
the upstream route. In YAML, `base_url` and `mag_provider` can override those
environment settings. Example:

```yaml
provider: mag
model: <catalog-model-id>
mag_provider: openai
output_root: ./runs/mag-qualification
temperature: 0.0
timeout_sec: 120
```

For an E1 development candidate, set `provider: mag`, use the catalog model ID
as `model_id`, and optionally set `mag_provider` in the candidate record. This
selection is included in the manifest hash and response provenance. `llm-check`
and the provider factory read `AMSC_TOKEN` at runtime; they do not put it in
resolved config, run metadata, or manifests. Responses retain MAG's
`X-AmSC-MAG-Applied` value when provided. The E1 runner remains restricted to
manifest-supplied development questions; this integration does not authorize
or launch the study.

## Official resources

- [MAG overview](https://amsc-docs-d762d2.gitlab.io/model-access-gateway/)
- [Get started with MAG](https://amsc-docs-d762d2.gitlab.io/mag-get-started/)
- [MAG API reference](https://mag.genesis.americansciencecloud.org/)
- [Model selection and options](https://amsc-docs-d762d2.gitlab.io/model-names/)
- [Rate limits](https://amsc-docs-d762d2.gitlab.io/mag-rate-limits/)
- [Troubleshooting](https://amsc-docs-d762d2.gitlab.io/mag-troubleshooting/)
- [MAG dashboard](https://mag.genesis.americansciencecloud.org/dashboard/)