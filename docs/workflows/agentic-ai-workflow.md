# Agentic AI Workflow Diagram

This page provides a reusable, standalone diagram for presentations, wiki pages, and architecture notes.

Slide-friendly compact variant: [agentic-ai-workflow-slides.md](agentic-ai-workflow-slides.md)

```mermaid
flowchart TD
    U[User objective or chat dialogue]

    subgraph MODES[User-facing modes]
      O[Objective mode<br/>run] --> OP[Plan and execute tasks]
      P[Composition mode<br/>supervisor-run] --> PP[Prepare and explore composition]
      I[Interactive mode<br/>chat] --> IP[Dialogue, detect composition, confirm action]
    end

    U --> O
    U --> P
    U --> I

    subgraph SCIENCE[Shared scientific capabilities]
      X[Phase search and MLIP relaxation<br/>HydraGNN, UMA, or MACE]
      Q[UQ evaluation<br/>entry-mode policy adapter]
      AL[Active learning loop]
      E[Results and auditable evidence]

      X --> Q
      Q -->|low confidence + policy enabled| AL
      Q -->|otherwise| E
      AL --> E
    end

    OP --> X
    PP --> X
    IP -->|explore or /relax| X
    IP -->|/al| AL
    IP -->|conversation only| IR[Chat response]

    E --> R{Return to invoking mode}
    R -->|objective| OA[Analyst report]
    R -->|composition| PS[Composition summary]
    R -->|interactive| IR
```

## Notes

- Objective, composition, and interactive modes are user-facing adapters, not
  separate scientific workflows. They share discovery, relaxation, and
  active-learning capabilities, then format results for their invoking mode.
- UQ handoff policy is exposed through mode-specific adapters today; the
  diagram groups those adapters by their common responsibility.
- Fused HydraGNN branch weights can drive the UQ policy directly. Pinned-head
  HydraGNN uses MC-dropout when configured; UMA and MACE require another
  acquisition strategy because they do not emit HydraGNN branch weights.
- UQ policy thresholds are configurable from CLI flags (`--uq-top-weight-threshold`, `--uq-min-unreliable-fraction`, and related handoff options).
- Handoff decisions are auditable via JSONL artifacts when `--al-handoff-audit-path` is set (or through default audit paths).
