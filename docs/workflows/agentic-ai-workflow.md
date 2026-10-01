# Agentic AI Workflow Diagram

This page provides a reusable, standalone diagram for presentations, wiki pages, and architecture notes.

Slide-friendly compact variant: [agentic-ai-workflow-slides.md](agentic-ai-workflow-slides.md)

```mermaid
flowchart TD
    U[User objective or chat dialogue]
    U --> R[run graph]
    U --> C[chat REPL]
    U --> S[supervisor graph]

    subgraph RPATH[Core run path]
      R --> RP[planner]
      RP --> RE[executor]
      RE --> RU[uq_gate]
      RU -->|high confidence| RA[analyst]
      RU -->|low confidence + policy enabled| RAL[active learning loop]
      RAL --> RA
    end

    subgraph SPATH[Supervisor path]
      S --> SP[prepare]
      SP --> SX[explore]
      SX --> SU[evaluate_uq]
      SU -->|low confidence + policy enabled| SAL[active learning loop]
      SAL --> SS[summarize]
      SU -->|otherwise| SS[summarize]
    end

    subgraph CPATH[Chat path]
      C --> CC[composition detection / optional relax]
      CC --> CU[uq policy]
      CU -->|low confidence + policy enabled| CAL[active learning loop]
      CAL --> CR[chat response]
      CU -->|otherwise| CR
    end
```

## Notes

- All three orchestration entry points can escalate into the same active-learning loop.
- Fused HydraGNN branch weights can drive the UQ policy directly. Pinned-head
  HydraGNN uses MC-dropout when configured; UMA and MACE require another
  acquisition strategy because they do not emit HydraGNN branch weights.
- UQ policy thresholds are configurable from CLI flags (`--uq-top-weight-threshold`, `--uq-min-unreliable-fraction`, and related handoff options).
- Handoff decisions are auditable via JSONL artifacts when `--al-handoff-audit-path` is set (or through default audit paths).
