# E1 Preparation Draft

Status: design scaffold only. No candidate has been served or scientifically
qualified. Questions below are draft wording, not independently authored or
expert-reviewed final questions, and are not validated gold answers. No held-out
question may be used for panel selection. Do not start model inference until
the candidate access, serving, and resource authorization gates are approved.

Manuscript source of record: `research/paper/manuscript/main.tex`. The tracked
editable source postdates `origin/research/numerical-results-refresh` at commit
`0aa40cfa3d887a4677d4be604fcd469bd02e462b` and the earlier archive
`docs/paper/new-manuscript.zip` (historical archive SHA-256
`c6c400ab8c2d1ad4baa8edf4828a9f8d471c90013ada4dee0c7d601910ee42a5`). Do not
extract that archive over the live source. The manuscript specifies four E1a
held-out conditions and treats the 24- and 36-question designs as exploratory
feasibility studies, not evidence of a generally winning strategy. E1b is a
separate proposed supplementary assessment; see the E1b proposal files listed
in the run plan. Its questions and outcomes must not feed back into E1a panel,
prompt, or held-out-set revisions.

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
| `mixtral-8x22b-instruct-v0.1` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Not assessed | Unmeasured |

Claude Fable is retained only as a pending placeholder: N/A - not evaluated,
excluded from aggregates. This does not establish a candidate ID, serving
route, revision, authorization, inference, or scientific assessment.

Mixtral is an additional development-screening candidate requested by the user;
it is not in the debate-enabled catalog. The Aurora and Perlmutter open-model
download scripts list its model ID, but no complete local checkpoint or
immutable revision was found in the inspected project model roots. Its stated
operability is user-reported but is not yet tied to a facility, provider,
serving revision, license approval, or E1 authorization. Keep it in the
screening inventory as **not assessed**, not unavailable; establish an approved
local or remote route and immutable revision before creating its candidate-bound
run manifest.

### Platform-Blocked Models and Requalification Record

The models below are not E1 candidates on Aurora and have not been served or
scientifically assessed for this study. Their exclusion is a current-platform
readiness decision, not a claim that the weights are unusable on all hardware.
Keep the checkpoint identity, quantization, download route, and inference
instructions together so a future HOC platform with validated FP8/FP4 hardware
and kernels can requalify them without treating staging as approval.

| Model | Immutable source / license | Checkpoint format | Aurora study disposition | Staging and inference provenance |
|---|---|---|---|---|
| Mistral Large 3 675B Instruct | `mistralai/Mistral-Large-3-675B-Instruct-2512` at `383ffea2c7d60dfd44ca960e8e691709d4fdb9cd`; Apache-2.0 | `compressed-tensors` FP8_BLOCK, 128x128 weight blocks | Not qualified on Aurora's Intel PVC / vLLM-XPU 0.15.0 stack: no supported FP8 kernel or successful model-serving qualification is recorded. Separately, the project vLLM 0.29.0 + Transformers 5.17.0 stack has a known Pixtral import incompatibility; the catalog therefore disables it for debate on that stack. | Listed in the Aurora, Frontier, and Perlmutter model download scripts. Aurora can stage this exact revision with `MODEL_IDS="mistralai/Mistral-Large-3-675B-Instruct-2512"` and `MODEL_REVISION=383ffea2c7d60dfd44ca960e8e691709d4fdb9cd` using [`download-open-models-aurora.sh`](../../deployments/aurora/download/download-open-models-aurora.sh). The expected local leaf is `Mistral-Large-3-675B-Instruct-2512`; it was absent from the inspected `../models` root. For future GPU serving, Perlmutter already documents a 4-node native-format vLLM invocation in [`job-serve-multinode-perlmutter.sh`](../../deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh) and the isolated vLLM environment in [`setup/README.md`](../../deployments/perlmutter/setup/README.md); this is a route to re-test, not evidence of successful qualification. Aurora's stack setup/verification route is [`install-vllm-xpu-aurora.sh`](../../deployments/aurora/setup/install-vllm-xpu-aurora.sh) and [`vllm-aurora.md`](../../docs/vllm-aurora.md); neither establishes FP8 compatibility. |
| DeepSeek V4.1-Flash-1017 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`; MIT | FP8 dynamic quantization, 32x32 blocks, with FP4 experts | Not qualified on Aurora: the current XPU stack has no recorded FP8/FP4 kernel or model-serving validation. It also needs protocol-aware prompt encoding: the snapshot has no Jinja chat template and points to the DeepSeek encoding implementation / `deepseek-recipe`. | Not in current downloader defaults. Pin and stage with `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ../models/DeepSeek-V4.1-Flash-1017` (or add that exact ID and revision to a future platform-specific staging job). The inspected root contains a 48-shard, approximately 476 GiB snapshot; the final aggregate SHA-256 audit was not captured, so do not label it integrity-verified. The snapshot's `inference/README.md` is the current reference install/run path: install its `requirements.txt`, convert per tensor-parallel rank with `convert.py --expert-dtype fp4`, then run with `torchrun`. This is reference inference, not an Aurora vLLM integration. No platform-specific scheduler serving job or runtime qualification has been recorded. |

For either model, future-HOC requalification requires recording the actual GPU
generation, framework and serving versions, supported FP8/FP4 kernels and
weight-loading path, immutable checkpoint revision and completed integrity
audit, tokenizer/prompt-encoding behavior, and a bounded serving smoke test
before any E1 candidate approval. Preserve Mistral's known Pixtral/Transformers
issue as a separate software-stack blocker; resolving it would not establish
Aurora FP8 support. DeepSeek-R1-Distill-Qwen-32B is a separate BF16 checkpoint,
and DeepSeek-V3.2 is a separate FP8 checkpoint; neither inherits V4.1's status
or is qualified by this record.

The versioned development packet is
[`e1/d-questions.v1.json`](e1/d-questions.v1.json). The separate H packet is
stored outside this repository with owner-only permissions; its path and digest
are recorded in [`e1/packet-hashes.v1.json`](e1/packet-hashes.v1.json). H prompt
content appeared in tracked history, so the current copy is not eligible as an
uncontaminated held-out set and must be replaced and independently reviewed
before held-out use. The runner has not served H. The full-shape runner fixture at
[`e1/d-manifest.schema-fixture.v1.json`](e1/d-manifest.schema-fixture.v1.json)
uses a deliberately unsupported provider and is for offline schema validation
only; it is not a candidate manifest and must not be executed. Proposed
scoring, qualification, adjudication, execution, budget, and readiness details
are in [`e1/development-protocol.v1.md`](e1/development-protocol.v1.md).

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
held-out situations must differ scientifically, not by paraphrase. The H packet
is stored outside this repository with owner-only permissions; keep its content
inaccessible to candidate selection and serve it only after panel freeze.

### 36-question matrix (preferred; design summary)

The versioned D packet is canonical for exact model-facing text. The matrix
below is a planning summary retained for partition design; where its quoted
wording differs from `e1/d-questions.v1.json`, the versioned packet controls.
Scientific review annotations and prompt-specific anchors follow the matrix.

IDs use `D` for development and `H` for held-out, followed by material and task
type (`A` comparison, `B` evidence audit, `C` open hypothesis generation).

| Material | A: controlled comparison (D / H) | B: evidence audit (D / H) | C: open hypothesis generation (D / H) |
|---|---|---|---|
| LiFePO4 | `D-LFP-A`: “Consider a Li-vacancy in olivine LiFePO4. Compare localized and delocalized Fe/electronic charge-compensation explanations. What mechanisms and magnetic/charge assumptions control this comparison, and what matched calculation and possible outcomes would distinguish the explanations?” `H-LFP-A`: [sealed prompt; held-out packet stored outside repository] | `D-LFP-B`: “Audit this explicitly hypothetical report: two Li-vacancy configurations were compared, but their magnetic initializations differ and the report omits how charge compensation was checked. State what is supported, what is not, which missing evidence matters most, and a test that could change the interpretation.” `H-LFP-B`: [sealed prompt; held-out packet stored outside repository] | `D-LFP-C`: “For LiFePO4 under a Li-vacancy objective, propose at least two physically plausible vacancy-accommodation hypotheses. For each, give a mechanism, assumptions, an observable, a feasible MLIP/DFT investigation, and a result that would falsify it. Do not present a hypothesis as an established result.” `H-LFP-C`: [sealed prompt; held-out packet stored outside repository] |
| Cantor FCC alloy | `D-CANTOR-A`: “For equimolar CrMnFeCoNi FCC configurations, compare the explanation that chemical short-range order controls an energy difference with the explanation that magnetic initialization controls it. State the assumptions and propose a composition- and protocol-matched calculation; explain outcomes that distinguish the effects.” `H-CANTOR-A`: [sealed prompt; held-out packet stored outside repository] | `D-CANTOR-B`: “Audit this explicitly hypothetical packet: several FCC configurations have reported energies, but only a small and non-independent sample of chemical arrangements was examined; magnetic initializations are stated but no convergence comparison is provided. Identify supported conclusions, limitations, and the next discriminating test.” `H-CANTOR-B`: [sealed prompt; held-out packet stored outside repository] | `D-CANTOR-C`: “For an equimolar Cantor-alloy objective focused on local order and a nominated structural or energetic observable, formulate at least two plausible mechanisms. State assumptions, predicted observables, feasible MLIP/DFT tests, and outcomes that would falsify each mechanism.” `H-CANTOR-C`: [sealed prompt; held-out packet stored outside repository] |
| NbTaVHfZrTi BCC alloy | `D-REFRACTORY-A`: “For one fixed Nb-Ta-V-Hf-Zr-Ti composition, compare BCC, FCC, and HCP structural prototypes while separating prototype effects from the particular chemical arrangement in each finite cell. Specify a matched comparison and explain which outcomes distinguish those factors.” `H-REFRACTORY-A`: [sealed prompt; held-out packet stored outside repository] | `D-REFRACTORY-B`: “Audit this explicitly hypothetical packet: one BCC cell and a small number of competing-prototype cells have relaxed energies, but independent chemical orderings are incomplete. State what the calculations establish, what they do not establish about a phase, and the most informative next comparison.” `H-REFRACTORY-B`: [sealed prompt; held-out packet stored outside repository] | `D-REFRACTORY-C`: “For a specified refractory-alloy composition and a local-environment objective, generate at least two hypotheses linking chemical neighborhoods to a structural or energetic response. Give mechanisms, uncertainties, observables, feasible MLIP/DFT tests, and falsifying outcomes.” `H-REFRACTORY-C`: [sealed prompt; held-out packet stored outside repository] |
| Phosphorene | `D-PHOS-A`: “Compare black and blue phosphorus monolayers under the same explicitly defined in-plane strain convention. Which structural or energetic observable would discriminate a proposed ordering, and what matched calculation and outcomes would support or contradict it? Do not infer a bulk-phase ranking.” `H-PHOS-A`: [sealed prompt; held-out packet stored outside repository] | `D-PHOS-B`: “Audit this explicitly hypothetical monolayer report: two phosphorene structures are compared at stated geometries, but a convergence check for the out-of-plane setup is missing. State the supported scope, what remains uncertain, and the most discriminating check.” `H-PHOS-B`: [sealed prompt; held-out packet stored outside repository] | `D-PHOS-C`: “For a phosphorene monolayer under a specified strain-response objective, propose at least two plausible mechanisms for a structural or electronic change. State assumptions, predicted observables, feasible MLIP/DFT investigations, and a result that would falsify each hypothesis.” `H-PHOS-C`: [sealed prompt; held-out packet stored outside repository] |
| Zn(HCOO)2 MOF | `D-MOF-A`: “Compare two chemically complete Zn(HCOO)2 framework configurations with different linker or proton arrangements. How would you separate ordering effects from framework breathing in a controlled comparison? State assumptions, observables, and outcomes that distinguish the explanations.” `H-MOF-A`: [sealed prompt; held-out packet stored outside repository] | `D-MOF-B`: “Audit this explicitly hypothetical framework packet: stoichiometry is reported, but proton positions and guest occupancy are incomplete. State which structural conclusions are supportable, what cannot be compared, and what information or test would change that assessment.” `H-MOF-B`: [sealed prompt; held-out packet stored outside repository] | `D-MOF-C`: “For a Zn(HCOO)2 framework with a specified proton- or linker-ordering constraint, formulate at least two plausible hypotheses for a framework observable. State mechanisms and uncertainties, then propose feasible MLIP/DFT investigations and outcomes that would falsify each.” `H-MOF-C`: [sealed prompt; held-out packet stored outside repository] |
| Nb-Ta-O | `D-NBTAO-A`: “Within an explicitly declared but incomplete Nb-Ta-O competitor domain, compare two possible explanations for composition-dependent oxide formation. State what the available domain can test, what assumptions affect the comparison, and which calculation or phase evidence could distinguish the explanations. Do not claim global stability.” `H-NBTAO-A`: [sealed prompt; held-out packet stored outside repository] | `D-NBTAO-B`: “Audit this explicitly hypothetical coverage manifest: a candidate Nb-Ta-O hull includes computed entries, but one required formula or declared polymorph family is missing. Is the hull qualified under its declared contract? Explain the gap, the strongest allowed claim, and what must be added.” `H-NBTAO-B`: [sealed prompt; held-out packet stored outside repository] | `D-NBTAO-C`: “For a stated Nb-Ta-O composition objective, propose at least two distinct hypotheses for oxide formation, including ordered-phase and disordered/solid-solution possibilities where appropriate. Identify assumptions, required phase/ordering evidence, observables, feasible calculations, and falsifiers; do not assert an unsupported stable phase.” `H-NBTAO-C`: [sealed prompt; held-out packet stored outside repository] |

For Type B, the packet owner must supply provenance and the actual evidence or
mark the scenario explicitly hypothetical. A hypothesis is not a validated
physical conclusion. Type C accepts multiple defensible mechanisms; its packet
defines constraints and scoring criteria, not one uniquely correct hypothesis.
No E1 question substitutes for E2's controlled interpretation of complete,
fixed numerical evidence.

### E1a question review key (proposed)

This key is a reviewer aid, not a gold-answer set, verified evidence packet,
or scientific approval. Exact model-facing wording is owned by
[`e1/d-questions.v1.json`](e1/d-questions.v1.json); the file hash is recorded in
[`e1/packet-hashes.v1.json`](e1/packet-hashes.v1.json). No prompt includes
verified empirical or computed results. A and C ask qualitative comparisons or
hypothesis generation without supplied results; B presents explicitly
hypothetical audit evidence. Reviewers must not expect a preferred physical
result where none is supplied. For each item, approve scope and assumptions,
check that response paths are scientifically defensible, and record citations
or source identifiers before any factual evidence is added. The anchors identify
high-value reasoning and critical errors for blinded scoring; they do not
prescribe one answer.

| Question IDs | Task type | Supplied evidence and provenance |
|---|---|---|
| `D-LFP-A`, `D-CANTOR-A`, `D-REFRACTORY-A`, `D-PHOS-A`, `D-MOF-A`, `D-NBTAO-A` | A: qualitative controlled comparison | No empirical/computed results or external source cited; assumptions and comparison targets are stated in the canonical prompt. |
| `D-LFP-B`, `D-CANTOR-B`, `D-REFRACTORY-B`, `D-PHOS-B`, `D-MOF-B`, `D-NBTAO-B` | B: hypothetical evidence audit | Qualitative scenario facts are explicitly hypothetical; no numerical result, source, dataset, or real calculation is supplied. |
| `D-LFP-C`, `D-CANTOR-C`, `D-REFRACTORY-C`, `D-PHOS-C`, `D-MOF-C`, `D-NBTAO-C` | C: open hypothesis generation | No empirical/computed results or external source cited; prompts state a hypothetical objective/constraints and request falsifiable alternatives. |

The category and wording alignment below was checked against all 18 IDs in
the canonical JSON. The prompt text in that file remains authoritative if this
review aid is later edited. Any cited evidence or non-hypothetical result added
to a future packet requires independently traceable provenance and a new
versioned review; do not retrospectively treat these qualitative briefs as
validated evidence packets.

| Question | Objective, assumptions, evidence status, and defensible reasoning anchors | Critical errors to flag |
|---|---|---|
| `D-LFP-A` | Compare localized Fe-centered oxidation with a distributed hole in a neutral, spin-polarized Li-vacancy cell. Assume fixed composition and controlled magnetic order/starting moments. No results are supplied. Accept charge-density, local-moment, and matched electronic-structure routes that distinguish labels from observables. | Treating formal oxidation labels as proof; changing electron count or magnetic setup between alternatives; asserting a preferred state without evidence. |
| `D-LFP-B` | Audit two hypothetical vacancy configurations initialized with different magnetic states and no final moments/charge analysis. No energies are supplied. Strong responses limit conclusions and propose same-composition, same-protocol magnetic and charge-density comparisons. | Attributing an energy or localization difference to vacancy geometry while magnetic initialization is confounded; inventing moments or energies. |
| `D-LFP-C` | Generate multiple hypotheses at one fixed Li-deficient composition. State supercell, composition, and magnetic assumptions; separate vacancy ordering from size, finite-size interactions, and magnetic state. No reference calculations exist in the prompt. | Changing composition between candidates; treating finite-cell ordering as bulk ordering; presenting hypotheses as established behavior. |
| `D-CANTOR-A` | Distinguish chemical short-range-order effects from magnetic-initialization effects in equiatomic FCC CrMnFeCoNi. Assume matched composition, cell size/shape, spin treatment, method, and relaxation; sample independent arrangements and multiple magnetic initializations. Accept a local-order descriptor and within/across-arrangement analysis. | Inferring order effects from one configuration; conflating magnetic and chemical sampling; claiming a universal ranking from the proposed design. |
| `D-CANTOR-B` | Audit a small, non-independent arrangement sample with one magnetic initialization per configuration. No numerical energies are supplied. Accept a bounded statement about the described sample plus a follow-up that adds independent arrangements and magnetic-state checks. | Treating one initialization as a converged magnetic ground state; claiming configurational or phase statistics from a small biased sample. |
| `D-CANTOR-C` | Propose competing links between local-neighbor statistics and the relaxed-energy distribution at fixed equiatomic FCC composition. A valid plan defines a descriptor, energy statistic, independent arrangements, and consistent spin treatment. | Assuming short-range order lowers energy; failing to separate arrangement-level observations from thermodynamic claims. |
| `D-REFRACTORY-A` | Compare BCC/FCC/HCP prototypes for equiatomic Nb-Ta-V-Hf-Zr-Ti while separating prototype effects from finite-cell ordering. Require composition/atom-count matching, multiple arrangements represented across prototypes, and a common method/relaxation protocol. | Claiming bulk phase stability from one ordered cell or unmatched compositions/cell sizes; ignoring arrangement variance. |
| `D-REFRACTORY-B` | Audit one ordered cell per prototype with differing chemical arrangements and no independent sampling or finite-temperature data. The dossier supports only a comparison of those particular relaxed cells if its method is indeed matched. | Calling a prototype ranking a phase-stability result; fabricating energies, convergence, or temperature evidence. |
| `D-REFRACTORY-C` | Form competing local-neighborhood hypotheses for distortion or configuration energy within an explicitly declared prototype. Accept independent arrangement sampling and consistent relaxation, with configuration-level conclusions kept distinct from bulk claims. | Changing prototype while attributing the effect to local chemistry; inferring bulk stability from configuration-level evidence. |
| `D-PHOS-A` | Compare black and blue phosphorus monolayers at zero in-plane strain using like-for-like relaxed energy per atom, electronic-structure settings, vacuum, and in-plane sampling. Scope conclusions to the monolayers, not bulk phosphorus. | Assuming an energy order; extrapolating monolayer results to bulk phases; neglecting vacuum or convergence controls. |
| `D-PHOS-B` | Audit a nominally matched monolayer comparison with untested vacuum and out-of-plane electrostatic convergence; geometries and energies are absent. Accept a scoped uncertainty statement plus convergence tests that could alter the comparison. | Inferring a ranking from missing values; treating nominally identical methods as proof of convergence. |
| `D-PHOS-C` | Develop multiple hypotheses across the stated biaxial strain series relative to each structure's relaxed reference. Define structural/electronic observables and distinguish strain response from vacuum or k-point artifacts. | Calling a transition without evidence; using inconsistent strain references; confusing numerical artifacts with physical response. |
| `D-MOF-A` | Compare two explicitly hypothetical, guest-free, stoichiometric Zn(HCOO)2 configurations with fixed connectivity/protonation but differing formate orientation. Make hydrogen/proton policy, cell-shape policy, composition, and dispersion treatment comparable; separate orientation order from framework breathing. | Treating hypothetical structures as observed phases; changing guest/proton content or dispersion inconsistently; assigning a ranking without structures/results. |
| `D-MOF-B` | Audit missing hydrogen coordinates, guest occupancy, and dispersion provenance in a hypothetical framework comparison. The strongest answer identifies which structural/energy claims are underdetermined and requests minimum records or matched calculations. | Inventing framework details; claiming energetic comparability from stoichiometry/connectivity alone. |
| `D-MOF-C` | Form competing hypotheses for formate-orientation order versus framework breathing in the stated hypothetical guest-free framework. Define response, hydrogen treatment, cell constraints, and dispersion-aware matched tests. | Treating a hypothetical configuration as an experimentally realized phase; omitting proton/cell assumptions; claiming an unsupported dominant mechanism. |
| `D-NBTAO-A` | Compare a target compound with decomposition only within the declared finite Nb-Ta-O domain and compatible formation-energy references. Accept a domain-relative convex-hull test and an explicit account of missing phases that could change the result. | Claiming global stability from an incomplete competitor domain; using incompatible references or inventing energies. |
| `D-NBTAO-B` | Audit a hypothetical hull missing a polymorph family required by its predeclared domain contract. It fails that coverage contract; the strongest claim is limited to listed entries until the required phase and compatibility checks are added. | Calling the hull complete or the target stable/unstable despite the specified coverage gap. |
| `D-NBTAO-C` | Contrast ordered-compound and disordered/solid-solution-like hypotheses at zero temperature in a declared finite domain. Separate ordered-cell energies from finite-temperature configurational entropy; identify compatible formation energies, ordering evidence, tests, and falsifiers. | Inferring finite-temperature stability from zero-temperature ordered cells; asserting global stability without a complete reference domain. |

**Prompt change log.** The canonical D packet is version `e1-questions-v1`.
Five prompts were tightened to fix controls and scope: `D-LFP-A` states the
neutral spin-polarized cell and separates oxidation labels from charge/moment
evidence; `D-LFP-C` fixes composition and separates ordering, cell-size, and
magnetic effects; `D-CANTOR-A` requires replicated chemical arrangements and
magnetic initializations; `D-MOF-A` fixes the hypothetical guest-free framework,
connectivity, protonation, and cell policy; and `D-MOF-C` fixes stoichiometry,
protonation, connectivity, and hydrogen/cell assumptions. This change log does
not replace independent scientific review.

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
GPU-hour, node-hour, or dollar estimates. The screening set now contains the
nine debate-enabled catalog entries plus Mixtral-8x22B-Instruct-v0.1 as a
requested supplemental candidate. Authorization, complete checkpoints,
serving compatibility, and facility feasibility remain unassessed.

| Design | Candidate qualification (10 × development items) | Held-out B (3/item) | Held-out C (9 condition calls/item; 6 additional unique if B reused) | Held-out D (9/item) | Held-out A extra | Total unique calls with B-to-C reuse |
|---|---:|---:|---:|---:|---:|---:|
| 24 questions (12 D + 12 H) | 120 | 36 | 108 condition calls; 72 additional unique | 108 | 0-12 | 336-348 |
| 36 questions (18 D + 18 H) | 180 | 54 | 162 condition calls; 108 additional unique | 162 | 0-18 | 504-522 |

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
| Development qualification | 12 questions × 10 candidates = 120 calls; 12 calls per candidate | 18 × 10 = 180; 18 per candidate |
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
	the model catalog plus the user-requested Mixtral-8x22B-Instruct-v0.1
	supplemental candidate. Catalog/deployment-script presence is not
	authorization, serving feasibility, or scientific qualification; document
	evidence-based exclusions.
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

- The nine local endpoint variables for the debate-enabled catalog candidates were
	checked by name only; all are unset in this shell. No endpoint values or
	credentials were read or recorded. This does not establish whether a service
	is available through another environment. Mixtral is outside this catalog;
	its user-reported operability has not been independently routed or verified.
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
	more than a queue-level candidate: a one-node, five-minute `/bin/true` probe
	with `qsub -A HydraGNN -q capacity -l filesystems=home:flare` was accepted as
	job `8979590.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`. It was
	cancelled immediately; `qstat -x -f` subsequently reported final state `F`.
	This verifies scheduler acceptance of that minimal request, not available
	balance, allocation-owner approval for E1, or authorization for inference.
	Other visible queues were either disabled, required at least 256 nodes, or
	showed ACL restrictions. No model work was started.
- These observations update operational readiness only. They do not approve
	candidate inference, change competence thresholds, authorize model downloads,
	or open the held-out partition.

### Remote API Readiness Snapshot (2026-10-29)

- Credential/configuration variable names were checked without reading values:
	`AMSC_TOKEN` is set; `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
	`MATSIM_MAG_BASE_URL`, and `MATSIM_MAG_PROVIDER` are unset. The documented
	MAG default endpoint was used; the token was not printed or stored.
- A metadata-only authenticated `GET /v1/models` returned HTTP 200 and 1,375
	catalog records. It includes `gpt-6.1-sol@openai` and `gpt-6-luna@openai`
	(and bare aliases); records report chat mode and token limits but no price or
	immutable serving revision. This establishes catalog visibility at request
	time only, not inference entitlement, allocation balance, charge rate, or
	scientific approval. No prompt or completion request was sent.
- Local `o200k_base` tokenization of the runner's exact system prompt/schema
	plus each canonical D prompt and 16 tokens of framing gave 514-551 estimated
	input tokens. A 1,024-token per-question input cap provides margin; use the
	runner's existing 16,384-token output cap and one attempt per question for an
	initial cost envelope. Provider-side tokenization and billing remain
	authoritative and may differ.
- Public standard OpenAI API prices observed on 2026-10-29 were $2/$10 per
	million input/output tokens for GPT-6.1 Sol and $0.10/$0.50 for GPT-6 Luna.
	At the proposed 1,024-input/16,384-output caps, 18 questions, and one attempt,
	the corresponding direct-API reference ceilings are $2.985984 and
	$0.1492992, respectively, before taxes or nonstandard pricing. These are
	scenario calculations, not MAG rates or approved spending ceilings: MAG
	documentation says usage is charged to the project allocation, and its
	applicable charge rate/balance has not been verified. Direct provider keys
	are unset.
- The MAG route's exact backend revision, allocation charge rate, remaining
	balance, project inference authorization, and applicable data-retention/use
	terms remain unverified. Public OpenAI API data-use statements do not by
	themselves establish which terms govern a MAG-routed request. Therefore no
	E1-R candidate manifest or inference is approved. Do not send prompts until
	the PI/allocation owner confirms authorization, the applicable data terms and
	accounting rate are documented, an immutable revision is pinned, and an
	explicit USD ceiling and retry policy are approved. A catalog lookup is not
	that approval.

### E1 Protocol Freeze Register

The 36-question design is selected; ten candidates is an upper bound, not a
minimum screening requirement. The
question wording and proposed rubric remain pending scientific review; neither
selection nor offline validation authorizes inference. Development execution
does not require completing the held-out comparison protocol, but it does
require the development-specific evidence and approvals listed in the versioned
development protocol. The entries below remain pending until the named evidence
is recorded and the relevant reviewers approve it.

| Gate | Required freeze evidence | Status |
|---|---|---|
| Question set and split | Review the exact versioned D packet for scientific adequacy and provenance; confirm the 18/18 split; replace the H packet because its prompt content appeared in tracked history, then verify D/H scientific independence and packet digests; maintain owner-only access before any held-out workflow | Pending domain review |
| Scoring and reviewers | Approve the proposed v1 anchors, thresholds, reviewer assignments/blinding, and adjudication rule before candidate responses are inspected | Pending domain review |
| Panel and conditions | Candidate eligibility and shortlist rule; deterministic B aggregation; C critique/revision schedule; response reuse rules; repeatability policy; confirmation of A-D call accounting | Pending protocol approval |
| Candidate and serving | Approved candidate IDs, revisions and licenses; decoding and seed policy; serving stack/version and hardware placement; response schema, parser, and provenance requirements | Pending candidate approval |
| Facility and execution | Approved account/project, queue, candidate scope, node/walltime and cost ceiling; data handling, stop/retry rules, accounting method, and explicit submission authorization | Pending facility and user approval |

Do not run candidate inference until these gates are approved, the development
manifest and protocol hashes are frozen, and the specific run's resource request
and submission are authorized. Do not begin held-out A-D comparison until cohort
selection and the complete held-out evaluation protocol are both frozen. The
current H packet must also be replaced before held-out use.

The live E1a design remains in `research/paper/manuscript/main.tex`; the older
archive has not been extracted over it. E1a's proposed question bank still
requires independent domain authoring/review, and its budget-matching claim
requires measured token/resource confirmation. The separate E1b proposal is
pending independent scientific review and has no inference authorization.
Preserve all pending results as planned, not completed. No scientific E1
inference, DFT, large HPC job, or held-out inspection is authorized by this
preparation draft. The user has approved only the bounded serving qualification
above; its job submission still requires the separate facility-side
verification and explicit submission approval recorded above. These E1b
preparation notes must not delay independently approved E1a development work.

## Results Ledger

| Candidate | Question ID | Run / model revision | Prompt hash | Response artifact | Parse status | Reviewer scores | Tokens / latency | Serving resources | Failures / notes |
|---|---|---|---|---|---|---|---|---|---|
| Not run | Not run | Not run | Not run | Not run | Not assessed | Not assessed | Unmeasured | Unmeasured | No inference authorized |

## Development Runner and Readiness Gates

The repository contains a development-only runner, exposed as
`matsim-agents e1-development MANIFEST.json --output RUN_DIR`. The strict JSON
manifest accepts only `partition: "development"` and question IDs beginning
with `D-`; it has no held-out input or path. It requires unique question IDs,
one candidate identity/revision, and temperature zero with a maximum of 32,768
output tokens, further bounded by `candidate.max_output_tokens_supported` when
provided. The separately versioned execution amendment records this ceiling;
the frozen scientific protocol remains unchanged. Retries are disabled by
default. Independent E1-L local and E1-R approved-remote cohorts use separate
candidate manifests and output directories. E1-R still requires explicit
route-, model-, data-use-, retention-, request-, and allocation-bound approval
before inference; the example candidate configuration is explicitly
non-dispatchable. The completed GPT-6 Luna original screen and its separately
authorized five-request recovery are recorded in
`runs/e1/e1-r-mag-gpt-6-luna-scoring-2026-10-29/evaluation-record.json`:
13/18 original first-attempt responses and 18/18 recovery-assisted complete
responses across 23 requests. Scientific scoring remains pending independent
blinded review. Software checks and schema validity are not evidence of
scientific competence.

For an approved development packet, create separate candidate manifests
containing only the reviewed development prompts and invoke each command in its
approved environment, independently. The output directory records the exact
manifest, run metadata, and append-only response ledger. A changed manifest
requires a new output directory. Do not put credentials or private endpoints in
the manifest. No production candidate manifest has been approved or run. A
metadata-only MAG catalog request succeeded as recorded above; no prompt,
completion, local model call, or inference request has been made. Direct
OpenAI/Anthropic credentials and explicit MAG provider overrides are unset.

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
caps its workflow budget at 90 minutes (1.5 node-hours). `debug` is limited to
60 minutes and cannot run that request, though its limit matches the proposed
one-hour qualification shape. `capacity` has scheduler limits compatible with
one node and 90 minutes, and a separate minimal five-minute `/bin/true` probe
was accepted and cancelled; the exact 90-minute request has not been submitted
or tested. Thus `capacity` is the scheduler-compatible candidate, not verified
project eligibility or charge authorization. Before any submission, confirm
allocation/account eligibility, balance and charge rules, the one-node resource
plan, and explicit authorization for the exact manifest, candidate, and job.
Do not submit the template or start inference until the protocol,
prompts/evidence, candidate/panel rule, rubric thresholds, serving revision,
resource ceiling, data handling, budget, stop condition, and cost-accounting
procedure are approved. Held-out A-D execution remains unimplemented and
unauthorized.