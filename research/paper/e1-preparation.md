# E1 Preparation Draft

Status: design scaffold only. No candidate has been served or scientifically
qualified. Questions below are draft wording, not independently authored or
expert-reviewed final questions, and are not validated gold answers. No held-out
question may be used for panel selection. Do not start model inference until
the candidate access, serving, and resource authorization gates are approved.

Manuscript reconciliation source: `origin/research/numerical-results-refresh`
at commit `0aa40cfa3d887a4677d4be604fcd469bd02e462b`, archive
`docs/paper/new-manuscript.zip`, containing `MATSIM_AGENTS/main.tex` (archive
SHA-256 `c6c400ab8c2d1ad4baa8edf4828a9f8d471c90013ada4dee0c7d601910ee42a5`).
The uploaded LaTeX specifies four held-out conditions and treats the 24- and
36-question designs as exploratory feasibility studies, not evidence of a
generally winning strategy.

The governing design and gates are in
[`docs/numerical-results-run-plan.md`](../../docs/numerical-results-run-plan.md).

## Candidate Inventory

Catalog entries below are starting candidates, not eligibility findings.
Catalog metadata is taken from
[`deployments/common/open-model-catalog.json`](../../deployments/common/open-model-catalog.json).
Set every status from evidence on the intended facility; do not infer it from
catalog presence or another facility's run.

| Catalog name | Model ID | License / revision | Architecture / parameters | Precision / hardware / parallelism | Serving stack / version | Context / parser | Local complete checkpoint | Authorized | Served | Response parsed | Scientific competence | Facility feasible | Cost / status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kimi-k2.5` | `moonshotai/Kimi-K2.5` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `glm-4.7` | `zai-org/GLM-4.7` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `glm-4.7-flash` | `zai-org/GLM-4.7-Flash` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `deepseek-v3.2` | `deepseek-ai/DeepSeek-V3.2` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `qwen3-235b-a22b-instruct-2507` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `qwen3-235b-a22b-thinking-2507` | `Qwen/Qwen3-235B-A22B-Thinking-2507` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `gemma-4-31b-it` | `google/gemma-4-31B-it` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `gemma-4-26b-a4b-it` | `google/gemma-4-26B-A4B-it` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |
| `devstral-2` | `mistralai/Devstral-2-123B-Instruct-2512` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |

Record evidence links or artifact identifiers for every status transition.
Document exclusion reasons rather than deleting candidates. Store no tokens,
credentials, or private endpoints in this inventory.

## Feasibility Benchmark Design

This is a proposed feasibility design, not a power calculation or a frozen
benchmark. The preferred design has 36 questions: 18 development/selection and
18 held-out, with one question from each task category for each material in each
partition. A 24-question pilot has 12 questions per partition, two per material;
it cannot cover all three categories in every material/partition cell. Preserve
coverage by assigning two categories per material in each partition and rotating
the omitted category across materials so every category appears in both
partitions and across the six-material set. Record that incomplete
material-by-category coverage in every report. A six-question infrastructure
smoke test (one material/category-balanced prompt per material) tests serving,
parsing, transcript capture, and reviewer workflow only; it supports no
comparative scientific-performance claim.

Each question below is a design brief, not a final factual answer key. The final
packet must turn its brief into an exact model-facing prompt with verified
sources/evidence or explicitly hypothetical qualitative evidence. Do not invent
numerical reference values, assert a phase ranking without compatible evidence,
or use development annotations to cue held-out reasoning. The development and
held-out situations must differ scientifically, not by paraphrase. Held-out
prompts and annotations must be access-controlled from candidate selection; only
the approved prompt set is served after panel freeze.

### 36-question matrix (preferred)

IDs use `D` for development and `H` for held-out, followed by material and task
type (`A` comparison, `B` evidence audit, `C` open hypothesis generation).

| Material | A: controlled comparison (D / H) | B: evidence audit (D / H) | C: open hypothesis generation (D / H) |
|---|---|---|---|
| LiFePO4 | `D-LFP-A`: “Consider a Li-vacancy in olivine LiFePO4. Compare localized and delocalized Fe/electronic charge-compensation explanations. What mechanisms and magnetic/charge assumptions control this comparison, and what matched calculation and possible outcomes would distinguish the explanations?” `H-LFP-A`: “Compare olivine and maricite LiFePO4 as candidate structures under a specified Li chemical-potential window. What compatible evidence is required before preferring either, and which numerical outcomes would support or contradict that preference? Do not assume a ranking.” | `D-LFP-B`: “Audit this explicitly hypothetical report: two Li-vacancy configurations were compared, but their magnetic initializations differ and the report omits how charge compensation was checked. State what is supported, what is not, which missing evidence matters most, and a test that could change the interpretation.” `H-LFP-B`: “Audit this explicitly hypothetical olivine/maricite comparison: reported energies use different reference conventions and the Li chemical-potential limits are not supplied. What, if anything, can be concluded? Identify the evidence needed to make a valid comparison and how its outcomes would affect your conclusion.” | `D-LFP-C`: “For LiFePO4 under a Li-vacancy objective, propose at least two physically plausible vacancy-accommodation hypotheses. For each, give a mechanism, assumptions, an observable, a feasible MLIP/DFT investigation, and a result that would falsify it. Do not present a hypothesis as an established result.” `H-LFP-C`: “Suppose the objective is to understand how a specified change in Li content affects an electronic or structural response in LiFePO4. Develop at least two distinct hypotheses, state uncertainties, predict observables, and propose feasible discriminating calculations and falsifiers. Do not assume either hypothesis is correct.” |
| Cantor FCC alloy | `D-CANTOR-A`: “For equimolar CrMnFeCoNi FCC configurations, compare the explanation that chemical short-range order controls an energy difference with the explanation that magnetic initialization controls it. State the assumptions and propose a composition- and protocol-matched calculation; explain outcomes that distinguish the effects.” `H-CANTOR-A`: “Assess a proposed FCC-versus-HCP comparison for composition-matched CrMnFeCoNi configurations. What must be controlled in chemical arrangement, magnetic treatment, and relaxation before interpreting relative energies? Give a falsifying result for the proposed interpretation.” | `D-CANTOR-B`: “Audit this explicitly hypothetical packet: several FCC configurations have reported energies, but only a small and non-independent sample of chemical arrangements was examined; magnetic initializations are stated but no convergence comparison is provided. Identify supported conclusions, limitations, and the next discriminating test.” `H-CANTOR-B`: “Audit this explicitly hypothetical packet: one set of calculations favors one chemical ordering while a second magnetic initialization changes the ordering of two configurations. Explain what is and is not identifiable, which controls are missing, and what evidence would resolve the disagreement.” | `D-CANTOR-C`: “For an equimolar Cantor-alloy objective focused on local order and a nominated structural or energetic observable, formulate at least two plausible mechanisms. State assumptions, predicted observables, feasible MLIP/DFT tests, and outcomes that would falsify each mechanism.” `H-CANTOR-C`: “Suppose the objective is to determine whether a small composition change alters a local-environment response in the Cantor alloy. Propose at least two hypotheses that distinguish composition effects from configuration effects, with observable predictions, feasible tests, and falsifiers.” |
| NbTaVHfZrTi BCC alloy | `D-REFRACTORY-A`: “For one fixed Nb-Ta-V-Hf-Zr-Ti composition, compare BCC, FCC, and HCP structural prototypes while separating prototype effects from the particular chemical arrangement in each finite cell. Specify a matched comparison and explain which outcomes distinguish those factors.” `H-REFRACTORY-A`: “A finite-cell study compares BCC and competing prototypes at a stated composition but samples different chemical orderings and magnetic initializations. What conclusions are warranted, what confounding remains, and what cell/order control could reverse the apparent ranking?” | `D-REFRACTORY-B`: “Audit this explicitly hypothetical packet: one BCC cell and a small number of competing-prototype cells have relaxed energies, but independent chemical orderings are incomplete. State what the calculations establish, what they do not establish about a phase, and the most informative next comparison.” `H-REFRACTORY-B`: “Audit this explicitly hypothetical packet: prototype energies are reported, but some structures did not reach the same relaxation criteria and magnetic initialization was not matched. Identify whether any relative ranking is admissible, list key limitations, and propose a falsification test.” | `D-REFRACTORY-C`: “For a specified refractory-alloy composition and a local-environment objective, generate at least two hypotheses linking chemical neighborhoods to a structural or energetic response. Give mechanisms, uncertainties, observables, feasible MLIP/DFT tests, and falsifying outcomes.” `H-REFRACTORY-C`: “Suppose the objective is to explain a composition-dependent tendency toward chemical ordering in NbTaVHfZrTi. Propose competing hypotheses that distinguish local-order effects from structural-prototype effects, and specify feasible observables, tests, and falsifiers.” |
| Phosphorene | `D-PHOS-A`: “Compare black and blue phosphorus monolayers under the same explicitly defined in-plane strain convention. Which structural or energetic observable would discriminate a proposed ordering, and what matched calculation and outcomes would support or contradict it? Do not infer a bulk-phase ranking.” `H-PHOS-A`: “Assess a proposed strain-dependent stability comparison between two phosphorene monolayers. Explain how in-plane strain, vacuum separation, and numerical convergence must be controlled, and identify a result that would change the interpretation.” | `D-PHOS-B`: “Audit this explicitly hypothetical monolayer report: two phosphorene structures are compared at stated geometries, but a convergence check for the out-of-plane setup is missing. State the supported scope, what remains uncertain, and the most discriminating check.” `H-PHOS-B`: “Audit this explicitly hypothetical energy comparison: the structures have different strain conventions and k-point sampling, while vacuum dimensions are reported. Is the energy difference interpretable? Identify the minimum metadata/control needed and explain how the conclusion could change.” | `D-PHOS-C`: “For a phosphorene monolayer under a specified strain-response objective, propose at least two plausible mechanisms for a structural or electronic change. State assumptions, predicted observables, feasible MLIP/DFT investigations, and a result that would falsify each hypothesis.” `H-PHOS-C`: “Suppose the objective is to understand whether a dimensionality or strain constraint changes a nominated phosphorene response. Generate at least two distinct hypotheses, separate physical mechanisms from convergence artifacts, and give feasible tests and falsifiers.” |
| Zn(HCOO)2 MOF | `D-MOF-A`: “Compare two chemically complete Zn(HCOO)2 framework configurations with different linker or proton arrangements. How would you separate ordering effects from framework breathing in a controlled comparison? State assumptions, observables, and outcomes that distinguish the explanations.” `H-MOF-A`: “Compare two stated Zn-formate framework configurations that differ in guest or proton state under a common computational method. What controls are needed before interpreting an energy difference, and what result would weaken the proposed structural explanation?” | `D-MOF-B`: “Audit this explicitly hypothetical framework packet: stoichiometry is reported, but proton positions and guest occupancy are incomplete. State which structural conclusions are supportable, what cannot be compared, and what information or test would change that assessment.” `H-MOF-B`: “Audit these explicitly hypothetical Zn-formate structure/energy records: the framework labels match, but guest state, hydrogen placement, and method equivalence are not all documented. Identify admissible conclusions, essential missing evidence, and a falsification or validation test.” | `D-MOF-C`: “For a Zn(HCOO)2 framework with a specified proton- or linker-ordering constraint, formulate at least two plausible hypotheses for a framework observable. State mechanisms and uncertainties, then propose feasible MLIP/DFT investigations and outcomes that would falsify each.” `H-MOF-C`: “Suppose the objective is to determine whether guest occupancy or framework breathing drives a specified response in Zn(HCOO)2. Propose competing hypotheses that distinguish these effects from proton ordering, with predicted observables, feasible tests, and falsifiers.” |
| Nb-Ta-O | `D-NBTAO-A`: “Within an explicitly declared but incomplete Nb-Ta-O competitor domain, compare two possible explanations for composition-dependent oxide formation. State what the available domain can test, what assumptions affect the comparison, and which calculation or phase evidence could distinguish the explanations. Do not claim global stability.” `H-NBTAO-A`: “Given a frozen, declared-domain Nb-Ta-O phase packet, compare a target-phase interpretation with a decomposition interpretation. Identify the evidence required, explain which outcomes change the domain-relative conclusion, and distinguish that conclusion from global thermodynamic stability.” | `D-NBTAO-B`: “Audit this explicitly hypothetical coverage manifest: a candidate Nb-Ta-O hull includes computed entries, but one required formula or declared polymorph family is missing. Is the hull qualified under its declared contract? Explain the gap, the strongest allowed claim, and what must be added.” `H-NBTAO-B`: “Audit this explicitly hypothetical result: every formula and polymorph required by a declared Nb-Ta-O domain contract has converged compatible energies and a convex hull has been computed. Does this establish globally exhaustive thermodynamic stability? Explain the scope, remaining out-of-domain possibilities, and a test or evidence source that could revise the claim.” | `D-NBTAO-C`: “For a stated Nb-Ta-O composition objective, propose at least two distinct hypotheses for oxide formation, including ordered-phase and disordered/solid-solution possibilities where appropriate. Identify assumptions, required phase/ordering evidence, observables, feasible calculations, and falsifiers; do not assert an unsupported stable phase.” `H-NBTAO-C`: “Suppose the objective is to determine whether an Nb-Ta-O composition is better described by an ordered compound or a solid-solution-like arrangement within a bounded reference domain. Formulate competing hypotheses, predicted observables, feasible MLIP/DFT tests, domain limitations, and falsifying outcomes.” |

For Type B, the packet owner must supply provenance and the actual evidence or
mark the scenario explicitly hypothetical. A hypothesis is not a validated
physical conclusion. Type C accepts multiple defensible mechanisms; its packet
defines constraints and scoring criteria, not one uniquely correct hypothesis.
No E1 question substitutes for E2's controlled interpretation of complete,
fixed numerical evidence.

### 24-question pilot mapping

Use six materials × two questions per partition, with categories rotated so all
three types occur in each partition and overall. Suggested allocation: LiFePO4
A/B; Cantor B/C; NbTaVHfZrTi C/A; phosphorene A/C; Zn-formate B/A; Nb-Ta-O C/B.
Use distinct D and H situations within each selected category. This pilot does
not provide every task type for every material and cannot support category-level
claims for each material. If full taxonomy coverage is a priority, use 36.

### Common response contract

Require a structured response with: (1) conclusion or at least two proposed
hypotheses for Type C; (2) mechanisms/reasoning; (3) assumptions; (4) supporting
evidence; (5) missing or contradictory evidence; (6) falsifiable predictions;
(7) proposed discriminating calculations/observations; (8) interpretation of
possible outcomes; and (9) qualitative confidence/qualification without
unsupported numerical certainty. Use the same schema and prompt framing across
all candidates. Score scientific quality separately from schema compliance;
record malformed or incomplete output without automatically treating it as
physical misunderstanding.

### Frozen evaluation packet checklist

Every question packet must include material, type, exact prompt, supplied
evidence/assumptions, source/provenance for every factual or numerical item,
accepted defensible reasoning paths, meaningful alternatives, essential missing
evidence, discriminating tests and outcome interpretation, examples of
unsupported claims, scoring anchors, partition, version, and content hash.
Two domain reviewers should approve evidence and anchors before model responses
are inspected. Type C packets must explicitly allow multiple defensible answers.
Freeze the split and hashes before scoring; keep held-out annotations sealed from
panel selection and development feedback.

## Scoring, Qualification, and Complementarity

Provisional rubric weights (freeze with domain reviewers before inference):

| Criterion | Weight | Core grading question |
|---|---:|---|
| Scientific correctness | 30% | Are physical claims consistent with the supplied evidence and established constraints? |
| Physical reasoning | 20% | Are mechanisms and causal links coherent, not merely named? |
| Falsifiability and discriminating tests | 20% | Could the proposed tests distinguish alternatives, and are outcomes interpreted? |
| Evidence awareness and uncertainty | 20% | Are gaps, contradictions, scope limits, and confidence calibrated? |
| Computational feasibility | 10% | Are proposed MLIP/DFT/experimental investigations feasible and appropriately scoped? |

Use a common ordinal anchor per criterion (0 = materially wrong/absent, 1 =
major gaps, 2 = partly sound, 3 = strong with minor omissions, 4 = rigorous and
well-qualified), with task-specific examples in each packet. Score format
compliance separately. At least two blinded reviewers independently score
responses; adjudicate material disagreements under a written rule. Report
reviewer agreement and preserve individual and adjudicated scores. Report
per-material and per-task results, plus six-material macro-averages; repeated
responses are not independent scientific cases.

Before seeing candidate results, domain reviewers must approve minimum competence
criteria for overall scientific correctness and evidence awareness, including
handling of unsupported confident claims and Nb-Ta-O hull scope. Do not set or
change thresholds after scores are visible. If fewer than three candidates
qualify, report that and stop panel selection; do not lower thresholds.

For complementarity, reviewers blinded to model identity tag each candidate's
development response for mechanisms, assumptions, evidence gaps, observables,
tests, and error modes. For each candidate pair, score (0-2, anchored) distinct
defensible mechanisms, assumption coverage, gap/test complementarity,
observable complementarity, useful disagreements, and overlap in scientifically
material errors. Linguistic style, verbosity, disagreement count, and embedding
distance are not complementarity. A disagreement earns value only if both
interpretations are defensible and imply a meaningful discriminating test;
systematic shared errors count against the pair. Rank triples using a
predeclared balance of individual competence, pairwise scientific
complementarity, measured inference efficiency, and operational reproducibility.
Provide a ranked shortlist with sensitivity to the agreed weights, not a
quality-unsupported automatic winner. All selection uses development responses
only; freeze and approve the triple before held-out access.

## Held-Out Comparison Protocol

On the frozen held-out set compare four conditions:

| Condition | Protocol | Call accounting per question |
|---|---|---:|
| A: strongest qualified individual | One independent answer from the highest-ranked qualified individual. Reuse its matching initial answer from B when the model is on the selected panel and prompt/decoding match exactly. | 0 additional calls if reusable; otherwise 1 |
| B: independent three-model panel | Each selected model answers independently; combine through a frozen deterministic aggregation rule with no synthesis LLM. Preserve all three raw outputs. | 3 |
| C: additional-inference/self-revision control | Same three models, provisionally three calls each: initial independent answer, private critique/reasoning, revised answer; no peer access. Reuse B's initial answer for each model only when prompt, model revision and decoding are identical. | 9 condition calls; 6 additional unique calls when B initial answers are reused |
| D: structured debate | Existing workflow: three equal participants, two round-robin rounds with rotating first speaker, then independent final verdicts (`equal`, `independent_verdicts`). Preserve full transcript and provenance. | 9 |

The current debate implementation makes six discussion calls plus three
independent verdict calls. Verify this against the pinned implementation before
the run manifest. The three-call-per-model C schedule is provisionally
call-matched to D, not yet token-, latency-, or resource-matched: debate prompts
include growing peer context, while C includes private critique and revision.
After an explicitly authorized bounded serving pilot, measure per-model input,
output and reasoning tokens and serving costs, then freeze a token/resource
matching rule or report the residual imbalance. Do not describe C as fully
budget-matched unless the measured protocol supports that claim. Report both the
fixed-protocol comparison and inference-cost-adjusted quality, with calls,
tokens, latency, serving overhead, physical GPU-hours, node-hours, and verified
facility charges. Reuse the strongest individual from B only when model
revision, prompt, and decoding match exactly; do not count reused outputs as new
observations.

The call estimates below are single-generation baselines with no retries. The
manuscript also proposes repeated generations to quantify within-question
variation; determine their number after a bounded repeatability pilot and
account for them separately. Repeated generations are not additional
independent scientific questions.
Track how debate changes claims between rounds, including corrections, new
mechanisms, unresolved defensible disagreement, introduced errors, and
unsupported consensus. Grade correctness, reasoning, evidence awareness,
falsifiability, and completeness by material and task category. Treat numerical
differences as exploratory feasibility estimates, not statistical proof of
superiority.

## Call and Resource Cost Estimate

These are exact planned call-count estimates under one response per
candidate/question, no retries, and no repeatability replicates. They are not
GPU-hour, node-hour, or dollar estimates. Nine debate-enabled catalog entries
are currently listed, but authorization, complete checkpoints, serving
compatibility, and facility feasibility remain unassessed.

| Design | Candidate qualification (9 × development items) | Held-out B (3/item) | Held-out C (9 condition calls/item; 6 additional unique if B reused) | Held-out D (9/item) | Held-out A extra | Total unique calls with B-to-C reuse |
|---|---:|---:|---:|---:|---:|---:|
| 24 questions (12 D + 12 H) | 108 | 36 | 108 condition calls; 72 additional unique | 108 | 0-12 | 324-336 |
| 36 questions (18 D + 18 H) | 162 | 54 | 162 condition calls; 108 additional unique | 162 | 0-18 | 486-504 |

Totals count C's initial independent answers as reused from B when the complete
request is identical; otherwise add up to three C calls per held-out item.
Totals also count A's strongest-model answer as shared with B when byte-identical
in scientific prompt and inference configuration; the upper bound adds one A
call per held-out item if the strongest qualified candidate is not a panel member
or cannot reuse B's response. Totals exclude serving smoke checks, repeatability
pilots, reviewer adjudication retries, failed requests, and operational
restarts. The 24-question layout's incomplete material-by-type coverage is
described above.

| Phase | 24-question design | 36-question design |
|---|---|---|
| Development qualification | 12 questions × 9 candidates = 108 calls; 12 calls per candidate | 18 × 9 = 162; 18 per candidate |
| Held-out A, strongest single | 0 additional if reused from B; otherwise 12 | 0 additional if reused from B; otherwise 18 |
| Held-out B, independent panel | 12 × 3 = 36; 12 per selected model | 18 × 3 = 54; 18 per selected model |
| Held-out C, private independent reasoning | 12 × 9 = 108 condition calls; after reusing B's initial answer, 72 new calls (24 per selected model) | 18 × 9 = 162 condition calls; after reusing B's initial answer, 108 new calls (36 per selected model) |
| Held-out D, debate | 12 × 9 = 108; 36 per participant | 18 × 9 = 162; 54 per participant |
| Unique request total under exact B/C/A reuse | 324-336 | 486-504 |

These are request-count estimates only. A facility estimate should be generated
from authorized measured profiles for each selected model and serving stack:

| Resource/cost quantity | Estimate method | Required observation not yet available |
|---|---|---|
| Inference calls | Counts above, plus observed retry/failure policy | Retries and repeatability policy; model qualification may require extra calls |
| Prompt/completion/reasoning tokens | Sum measured tokens by condition/model; include debate peer context and private C critique prompts | Token distributions from actual pinned models and final prompts |
| Accelerator-hours | Sum, over serving allocations, physical GPUs assigned × allocation elapsed hours; also report useful request service time separately | GPU count per replica, measured session/load/serve/teardown duration, and facility physical-device accounting |
| Node-hours | Sum reserved nodes × scheduler allocation elapsed hours, including idle/load time | Intended nodes, allocation boundaries, queue/launch setup and sequential-vs-concurrent schedule |
| Monetary/accounting cost | Apply verified facility/project billing unit and current rate to the allocated-resource ledger | Facility/account, available balance, billing rules/rates, and approved ceiling |

If per-model qualified serving duration for the exact workload becomes available,
use the call counts by model above as a first-order estimate, add measured
model-load/session and scheduler overhead, and include a separately reported
retry/headroom scenario. Do not extrapolate from parameter count or a different
facility. The 36-question design is 1.5× the question-driven workload of the
24-question design under identical eligibility, serving, and retry assumptions;
fixed per-session load overhead may make total allocated cost less than or more
than a simple 1.5× depending on whether model sessions can be reused. The current
repository has no applicable per-candidate E1 throughput/token profile, so a
numeric GPU-hour/node-hour/dollar estimate is not yet defensible. Obtain those
measurements in an explicitly authorized, bounded serving qualification before
seeking a full-study resource ceiling.

For facility planning, measure per-model prompt/completion/reasoning token
distributions, sustained generation throughput, model load/teardown time,
physical GPUs per replica, queue/allocation overhead, and retry rate during an
explicitly authorized bounded qualification. Then estimate each model's
allocated GPU-hours as measured serving walltime multiplied by physical GPUs;
estimate node-hours from reserved nodes × reservation elapsed time; and apply
only verified facility billing units/rates. Sequential model loading is the
preferred cost-saving starting configuration if it meets throughput and
reproducibility needs. No valid numeric resource or dollar estimate is possible
yet: exact revisions/checkpoints, hardware placement, serving stack, throughput,
allocation balances, and billing rates have not been established. E1 ordinarily
requires no DFT allocation.

## Decisions and Authorization Gates

### Current decision record

- **Design scale selected:** 36 questions, with 18 development/selection and
	18 held-out questions. This selects the design size only; the draft questions,
	evidence packets, independence, and split still require domain review and
	approval before any inference.
- **Candidate screening scope selected:** all nine debate-enabled entries in
	the model catalog. Catalog membership is not authorization, serving
	feasibility, or scientific qualification; document evidence-based exclusions.
- **Intended facility supplied:** ALCF Aurora; project ID `HydraGNN` confirmed
	by the user. The user confirms approval to use this allocation for a bounded
	E1 serving qualification; facility-side authorization has not been
	independently verified.
- **Bounded serving qualification:** one Aurora compute node, one locally
	available Mistral Small 24B checkpoint, four PVC tiles with tensor parallel
	size 4, one request at a time, temperature 0, and at most 60 minutes (1
	node-hour by requested reservation). The planned vLLM `max_model_len` is
	32,768, matching the local checkpoint's `max_position_embeddings`; runtime
	compatibility and memory fit still require compute-node verification. The
	maximum generated output is 16,384 tokens for one separate long-context
	stress request only. The tokenizer targets a 16,384-token prompt; the six
	standard readiness/load/generation/structured/discussion/distributed checks
	remain short. `startup_timeout_s` is 2,700 seconds; the managed-server and
	qualification workflow has a 3,300-second deadline from job start, reserving
	up to five minutes for interruption, shutdown, and artifact collection. This
	leaves at most 10 minutes after a full-length startup for qualification work;
	the bounded run will establish whether that is sufficient before any request
	to extend the allocation.
- **Queue and allocation gate:** the inspected `debug` execution queue accepts
	1-2 nodes and up to 60 minutes, matching this first bounded run. Queue
	visibility is not proof that project `HydraGNN` is eligible. PBS reports
	`allow_account_check_failure=True` and `allow_negative_allocation=True`, so
	successful `qsub` validation cannot establish adequate balance. Facility or
	allocation-owner confirmation of project authorization, `debug` access,
	available balance, and actual chargeable reservation is still required.
- **Submission state:** configuration preview is valid for one node and
	resolves the local model path, four devices, TP=4, 32,768 context, and 2,700 s
	startup timeout. The job is marked non-rerunnable and has a 60-minute PBS
	walltime, a workflow deadline bounded to leave cleanup time, no automatic resubmission,
	and no node-count escalation. These configuration checks do not authorize
	submission. Do not submit until facility-side project/queue/balance checks
	are documented and the user separately approves the specific submission.
- **Measurements and stop rules:** save startup duration (including failed
	starts), request prompt/completion token counts, finish/truncation status,
	latency and output tokens/s, best-effort per-tile XPU memory samples, script
	elapsed time, scheduler-reported allocation walltime, and PBS final
	accounting output. Facility charge remains
	unreported until verified from facility accounting. Stop after the bounded
	qualification succeeds, on unrecoverable deployment failure, after 45
	minutes without model readiness, or at the 60-minute allocation cap; always
	stop the server and preserve artifacts. No DFT is part of this qualification.
- **Current shell readiness:** none of the nine catalog endpoint environment
	variables is set in the inspected shell. This is an environment observation,
	not evidence that endpoints are unavailable elsewhere.

### Facility Readiness Snapshot (2026-10-28)

- The nine endpoint variables for the debate-enabled catalog candidates were
	checked by name only; all are unset in this shell. No endpoint values or
	credentials were read or recorded. This does not establish whether a service
	is available through another environment.
- The local `Mistral-Small-24B-Instruct-2501` checkpoint occupies about 44 GB,
	and all ten weight shards referenced by its safetensors index are present.
	It is not one of the nine E1 screening candidates: the catalog marks
	`mistral-large-3` as debate-disabled due to the documented serving-stack
	incompatibility. The local checkpoint is not evidence of access to or
	readiness of any E1 candidate.
- `qstat -Qf debug` reports `resources_max.walltime = 01:00:00` and
	`resources_max.nodect = 2`. Thus `debug` cannot satisfy the proposed
	one-node, 90-minute E1 ceiling. The enabled `capacity` queue reports a
	1-node minimum, 16-node maximum, and 168-hour maximum walltime, with no queue
	ACL listed; its scheduler limits therefore fit the requested shape. This is
	only a queue-level candidate: HydraGNN project/account eligibility, available
	balance, and allocation-owner approval have not been verified. Other visible
	queues were either disabled, required at least 256 nodes, or showed ACL
	restrictions. No PBS job was submitted as part of this check.
- These observations update operational readiness only. They do not approve
	candidate inference, change competence thresholds, authorize model downloads,
	or open the held-out partition.

Before candidate inference, approve: (1) 36-question preferred design or
24-question pilot; (2) exact six-material prompts, evidence packets, sources,
question independence, and split/hash; (3) rubric anchors, reviewer assignments,
adjudication, competence thresholds, and complementarity weights; (4) candidate
eligibility and panel-shortlist rule; (5) deterministic B aggregation and C's
self-revision schedule; (6) model revisions, decoding, serving stack, response
schema, and provenance; (7) facility/account, candidate scope, resource ceiling,
data handling, budget, and stop condition; and (8) cost-accounting procedure.

The revised E1 manuscript is available in the uploaded package identified above;
the archive has not been extracted into or substituted for
`research/paper/manuscript/main.tex`. The planning documents are aligned to its
four-condition E1 design, while its proposed question bank still requires
independent domain authoring/review and its budget-matching claim requires
measured token/resource confirmation. Preserve all pending results as planned,
not completed. No scientific E1 inference, DFT, large HPC job, or held-out
inspection is authorized by this preparation draft. The user has approved only
the bounded serving qualification above; its job submission still requires
the separate facility-side verification and explicit submission approval
recorded above.

## Results Ledger

| Candidate | Question ID | Run / model revision | Prompt hash | Response artifact | Parse status | Reviewer scores | Tokens / latency | Serving resources | Failures / notes |
|---|---|---|---|---|---|---|---|---|---|
| Not run | Not run | Not run | Not run | Not run | Not assessed | Not assessed | Unmeasured | Unmeasured | No inference authorized |

## Development Runner and Readiness Gates

The repository now contains a development-only runner, exposed as
`matsim-agents e1-development MANIFEST.json --output RUN_DIR`. The strict JSON
manifest accepts only `partition: "development"` and question IDs beginning
with `D-`; it has no held-out input or path. It requires unique question IDs,
one candidate identity/revision, and deterministic temperature zero with a
maximum of 16,384 output tokens. A run executes its pending questions
sequentially through one model session. The checked-in synthetic tests cover
held-out rejection, duplicate IDs, response reuse after restart, incomplete
trailing JSONL recovery, and refusal to reuse an output directory with a
different manifest. These are software integrity checks, not evidence of model
readiness or scientific competence.

For an approved development packet, create a separate manifest containing only
the reviewed development prompts and invoke the command above inside the
approved serving environment. The output directory records the exact manifest,
run metadata, and append-only response ledger. A changed manifest requires a
new output directory. Do not put credentials or private endpoints in the
manifest. No production candidate manifest has been approved or run.

Candidate readiness statuses must be evidence-based and distinguish at least:
**not assessed** (no facility-specific check), **blocked** (a documented
missing authorization, artifact, dependency, or resource), **failed** (a
recorded qualification attempt did not pass), **qualified** (all predeclared
operational criteria passed for the exact revision and serving stack), and
**excluded** (a documented panel-rule or eligibility reason). Keep scientific
competence separate from operational qualification; neither catalog presence
nor a successful health check establishes competence. Current candidate rows
remain not assessed except where the inventory records a narrower observation.
At least three candidates must pass the approved competence threshold before
panel selection; do not lower the threshold if fewer qualify.

The Aurora job file is a non-submitting template only. It requests one node and
caps its workflow budget at 90 minutes (1.5 node-hours), but the inspected
`debug` queue allows only 60 minutes. The template therefore does not select a
queue; before any submission, independently verify a queue that permits the
full walltime, confirm the allocation/account and one-node resource plan, and
record explicit authorization for the exact manifest, candidate, and job. No
eligible 90-minute queue has been verified here. Do not submit the template or
start inference until the protocol, prompts/evidence, candidate/panel rule,
rubric thresholds, serving revision, resource ceiling, data handling, budget,
stop condition, and cost-accounting procedure are approved. Held-out A-D
execution remains unimplemented and unauthorized.