# Agentic AI Workflow Diagram

This page provides a reusable, standalone diagram for presentations, wiki pages, and architecture notes.

Slide-friendly compact variant: [agentic-ai-workflow-slides.md](agentic-ai-workflow-slides.md)

```mermaid
flowchart TD
    U[User objective or dialogue] --> G{Choose interaction mode}

    subgraph MODES[User-facing modes]
      direction LR
      O[Objective-driven<br/>execution]
      P[Composition<br/>exploration]
      I[Interactive<br/>chat]
    end

    G --> O
    G --> P
    G --> I

    O -->|HydraGNN, UMA, or MACE| S
    P -->|HydraGNN, UMA, or MACE| S
    I -->|HydraGNN or UMA| S

    R{Return to invoking mode}

    subgraph SCIENCE[Shared scientific capabilities]
      S[Normalize objective, composition, or chat request]
      H{Hypothesis-led investigation?}
      HF[Formulate hypothesis]
      DB{Multi-LLM debate enabled?}
      DV[Compare hypotheses and rationale]
      IS[Select investigations and candidate compositions]
      X[Generate candidates, explore phases, and relax with MLIP]

      UQ[Estimate predictive uncertainty<br/>using configured acquisition strategy]
      AD{Acquisition policy selects<br/>candidates for reference labeling?}

      E[Numerical results with provenance]
      V[Assess evidence against the question<br/>convergence, required fidelity, coverage, references]
      EV{Evidence sufficient for<br/>the requested conclusion?}
      IN[Report evidence gap or inconclusive result]
      A[Evidence-informed scientific assessment]
      HR{Evidence review and hypothesis<br/>revision enabled?}
      RV[Revise hypothesis with cited evidence]

      S --> H
      H -->|Yes| HF
      H -->|No| IS
      HF --> DB
      DB -->|Yes| DV
      DB -->|No| IS
      DV --> IS
      IS --> X

      X --> UQ
      UQ --> AD
      X --> E
      E --> V
      V --> EV
      EV -->|No| IN
      EV -->|Yes| A
      A --> HR
      HR -->|Yes| RV
      RV --> HF
      HR -->|No| R
      IN --> R
      A --> R
    end

    subgraph ACTIVE[Conditional active-learning pathway]
      direction TD
      CS[Select candidates under acquisition policy]
      DFT[Run selective DFT labeling]
      QC{Labels valid and converged?}
      DS[Update labeled dataset]
      EX[Exclude failed labels from evidence and training]
      TR{MLIP adaptation enabled?}
      TRAIN[Train candidate MLIP]
      CMP[Compare candidate with incumbent<br/>on validation references]
      PROM{Promotion enabled, validated,<br/>and approved?}
      KEEP[Keep incumbent model]
      USE[Promote candidate model]

      AD -->|Yes| CS
      CS --> DFT
      DFT --> QC
      QC -->|Yes| DS
      QC -->|No| EX
      EX --> E
      DS --> E
      DS --> TR
      TR -->|Yes| TRAIN
      TR -->|No| KEEP
      TRAIN --> CMP
      CMP --> PROM
      PROM -->|No| KEEP
      PROM -->|Yes| USE
      USE -.->|subsequent calculations| X
    end

    AD -->|No| E

    CC[Optional characterization contract<br/>objective, observables, minimum fidelity,<br/>methods, tolerances, references, budget, approvals]
    CC -.-> IS
    CC -.-> V

    subgraph OUTPUTS[Mode-specific presentation]
      direction LR
      OA[Analyst report]
      PS[Composition summary]
      IR[Chat response]
    end

    R --> OA
    R --> PS
    R --> IR

    classDef entry fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef mode fill:#eff6ff,stroke:#2563eb,color:#172554
    classDef science fill:#ecfdf5,stroke:#059669,color:#052e16
    classDef decision fill:#fff7ed,stroke:#ea580c,color:#431407
    classDef optional fill:#fffbeb,stroke:#d97706,color:#78350f
    classDef safeguard fill:#fef2f2,stroke:#b91c1c,color:#450a0a
    classDef output fill:#faf5ff,stroke:#9333ea,color:#3b0764
    class U entry
    class G,H,DB,AD,EV,HR,QC,TR,PROM,R decision
    class O,P,I mode
    class S,HF,DV,IS,X,UQ,E,V,A,CS,DFT,DS,TRAIN,CMP,USE science
    class CC optional
    class IN,EX,KEEP safeguard
    class OA,PS,IR output
```

## Notes

- Objective, composition, and interactive modes are user-facing adapters, not
  separate scientific workflows. They share discovery, relaxation, and
  active-learning capabilities when configured, then format results for their
  invoking mode. Objective and composition workflows support HydraGNN, UMA, and
  MACE; interactive chat currently supports HydraGNN and UMA.
- Hypothesis formulation, multi-LLM debate, evidence review, and revision are
  conditional investigation or campaign capabilities, not steps in every run.
- A characterization contract applies to the characterization pathway; it is
  not a prerequisite for every mode or calculation. Evidence requirements
  depend on the requested observable and claim.
- Predictive uncertainty and acquisition rank candidates for possible reference
  calculations. They do not establish that evidence is sufficient for a
  scientific conclusion. Sufficiency depends on the requested claim, including
  convergence, fidelity, coverage, and compatible references where required.
- Active learning can add selectively DFT-labeled structures to a dataset
  without retraining. Adaptation and model promotion are separate opt-ins;
  promotion additionally requires candidate validation and approval. A
  promoted model may be used in subsequent MLIP calculations.
- Failed or unconverged calculations are excluded as positive scientific
  evidence. Evidence-ledger recording is available in campaign and
  characterization workflows, not imposed on every standalone workflow.
- UQ handoff policy is exposed through mode-specific adapters today; the
  diagram groups those adapters by their common responsibility. Fused
  HydraGNN branch weights can drive that policy directly, while pinned-head
  HydraGNN can use MC-dropout when configured. UMA and MACE do not emit
  HydraGNN branch weights, so this specific handoff signal needs an alternative.
- Active-learning acquisition strategies include ensemble disagreement,
  MC-dropout, random selection, and ensemble-then-dropout, depending on backend
  configuration.
- Interactive conversation-only turns return directly to chat, while `/al`
  invokes active learning directly; those shortcuts are omitted above to keep
  the primary scientific workflow legible.
- UQ policy thresholds are configurable from CLI flags (`--uq-top-weight-threshold`, `--uq-min-unreliable-fraction`, and related handoff options).
- Handoff decisions are auditable via JSONL artifacts when `--al-handoff-audit-path` is set (or through default audit paths).
