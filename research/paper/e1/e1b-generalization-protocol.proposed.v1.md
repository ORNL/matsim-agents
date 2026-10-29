# E1b Cross-Domain Generalization Proposal

Status: proposed initial design; pending independent scientific review and
author approval. The target is 12 questions, not a power calculation or a
frozen benchmark. No E1b responses, scores, numerical results, model calls, or
execution authorization exist.

## Purpose and Separation

E1b is a supplementary external-domain assessment of whether the reasoning
behaviors examined in E1a transfer to distinct materials-science concepts. It
does not replace, enlarge, or revise E1a. Preserve the approved E1a prompts,
D/H partition, rubric, panel-selection procedure, and conditions. Freeze the
E1a panel and protocol before E1b questions or responses are exposed to panel
selectors. E1b performance must never be used to revise E1a panels or prompts
and then be described as an independent generalization test.

The separate prompt packet is
`e1b-generalization-questions.proposed.v1.json`. Its `E1B-` IDs are
intentionally incompatible with the existing E1 development runner, which
accepts only `D-` questions. Do not add E1b IDs to the D packet, held-out H
packet, or E1 manifest. Any future E1b execution path, manifest, approvals,
accounting, and output directory require separate design and authorization.

## Proposed Coverage

The 12 qualitative items cover six domains, with two questions per domain:

| Domain | E1b concept focus | Contrast that responses must preserve |
|---|---|---|
| Charged point defects | Charge state, chemical potentials, charge compensation | Cell neutrality vs. defect charge and compensating carriers; formation energetics vs. conductivity |
| Diffusion | Migration pathways and transport | Migration barrier vs. defect formation, path population, prefactors, correlations, and macroscopic rate |
| Surface catalysis | Adsorption and multistep reactions | Adsorption energy vs. activation free energy, coverage, turnover, and selectivity |
| Lattice dynamics | Harmonic phonons and temperature | Numerical convergence vs. harmonic instability; local dynamics vs. thermodynamic and finite-temperature stability |
| Electrochemistry | Voltage and operating mechanism | Balanced free-energy differences per transferred charge vs. kinetic barriers, hysteresis, and mechanism |
| Alloy thermodynamics | Configurational disorder and phase stability | Finite-cell zero-temperature energy vs. composition- and temperature-dependent free energy |

These domains are intentionally distinct from the specific E1a six-system
question contexts. Both items in every domain currently state explicitly
hypothetical qualitative evidence and supply no experimental or computed
numerical values. A later evidence-bearing revision must be independently
verified, cite retrievable sources and exact methods, preserve source/version
provenance, and state its validity limits. Do not fill gaps with remembered
literature values or model-generated references.

## Shared Scoring Core and Domain Anchors

Reuse E1a's core 0-4 rubric and weights without changing its anchors: scientific
correctness (30%), physical reasoning (20%), falsifiability and tests (20%),
evidence awareness and uncertainty (20%), and computational feasibility
(10%). E1b uses the same response fields as a conceptual contract; it is not
currently executable through the E1 runner. Score scientific quality separately
from format compliance. The questions have no single target hypothesis: assess
whether proposed explanations are physically defensible and whether claims are
appropriately scoped to the hypothetical evidence.

The following domain anchors refine interpretation of the existing correctness
and evidence criteria. Scores 1 and 3 interpolate between adjacent anchors.

| Domain | 0: critical conflation | 2: partial distinction | 4: domain-aware, bounded reasoning |
|---|---|---|---|
| Charged point defects | Treats a neutral supercell as proof that a defect is neutral, or ignores chemical-potential/Fermi-level dependence when making a charge-state claim | Recognizes charge compensation or chemical-potential dependence but omits a decisive correction, reference, or charge-balance control | Separates defect charge, compensating species, chemical-potential limits and finite-size/electrostatic controls; limits the claim to the declared range |
| Diffusion | Equates one migration barrier with a macroscopic diffusion rate | Identifies a missing formation or kinetic factor but does not connect pathway statistics and temperature to transport | Separates defect availability, pathway barriers/populations, prefactors and correlated transport; specifies rate-relevant evidence and conditions |
| Surface catalysis | Equates stronger adsorption or a stable intermediate with higher catalytic activity | Recognizes that barriers or coverage matter but omits parts of the reaction cycle or operating conditions | Distinguishes adsorption thermodynamics, transition-state free energies, coverage, turnover and selectivity under comparable conditions |
| Lattice dynamics | Treats a sampled harmonic phonon result as proof of global thermodynamic or finite-temperature stability | Recognizes convergence or anharmonicity concerns but does not distinguish local dynamical evidence from phase stability | Separates numerical artifacts, harmonic local stability, anharmonic finite-temperature behavior and competing-phase thermodynamics |
| Electrochemistry | Infers voltage from unbalanced or incompatible endpoint energies, or equates equilibrium voltage with rate | Recognizes a reference or kinetic limitation but misses stoichiometry, competing phases, or mechanism evidence | Uses balanced free-energy changes and consistent references for scoped equilibrium voltage; treats kinetics and operating mechanism separately |
| Alloy thermodynamics | Infers finite-temperature equilibrium from one ordered-cell energy or arrangement count alone | Notes configurational entropy but lacks an appropriate ensemble, composition, or competitor comparison | Compares composition-specific energetic and entropic contributions with suitable sampling and competing phases over an explicit temperature range |

Two independent domain reviewers should approve each final item and its
evidence-status label. For scored runs, use blinded response IDs, two
independent reviewers, preserved raw scores, and a third adjudicator for
criterion differences of at least two points or different qualification
outcomes, consistent with the E1a development review procedure. Freeze all
anchors and adjudication rules before exposing E1b responses. Report reviewer
agreement, per-domain results, missing or malformed responses, and all
unsupported claims; do not represent 12 questions as a powered cross-domain
superiority test.

## Provenance and Independence Ledger

Maintain a separate, append-only E1b ledger. At minimum record question ID,
domain, prompt version and SHA-256, evidence-status classification, source
identifiers and source hashes if later added, reviewers and approvals, schema
and rubric versions/hashes, model revision, condition, output artifact hash,
parse status, reviewer scores, adjudication, token/latency/resource accounting,
and deviations. Keep the ledger separate from E1a run directories and cost
records. Never place credentials or private endpoints in artifacts.

`e1b-packet-hashes.proposed.v1.json` is a distinct register. Recompute hashes
after any change; a changed packet is a new version, not an in-place frozen
benchmark. Hash registration establishes file identity only, not scientific
validity, independence, approval, or execution authorization.

## Gates

1. Obtain two independent domain reviews of the 12 item briefs, domain anchors,
   evidence labels, and external-generalization claim.
2. Freeze E1a prompts, protocol, panel, and conditions before any E1b exposure.
   E1b review or results do not feed back into E1a.
3. Approve the E1b response contract, independent execution/analysis plan,
   reviewer blinding, missing-data policy, and separate resource/accounting
   limits before any inference.
4. If questions are converted to evidence-bearing packets, verify source
   provenance, methods, numerical compatibility, and claim scope; obtain a new
   version and independent review before scoring.
5. Obtain explicit authorization for each candidate, route, data handling,
   spend/resource ceiling, and execution. This proposal authorizes none of
   these actions.