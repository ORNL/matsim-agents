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
      RP[planner] --> RE[executor]
      RE --> RU[uq_gate]
      RU -->|high confidence| RA[analyst]
      RU -->|low confidence + policy enabled| AL[active learning loop]
      AL --> RA
    end

    subgraph SPATH[Supervisor path]
      SP[prepare] --> SX[explore]
      SX --> SU[evaluate_uq]
      SU -->|low confidence + policy enabled| AL
      SU -->|otherwise| SS[summarize]
    end

    subgraph CPATH[Chat path]
      CC[composition detection / optional relax] --> CU[uq policy]
      CU -->|low confidence + policy enabled| AL
    end
```

## Notes

- All three orchestration entry points can escalate into the same active-learning loop.
- UQ policy thresholds are configurable from CLI flags (`--uq-top-weight-threshold`, `--uq-min-unreliable-fraction`, and related handoff options).
- Handoff decisions are auditable via JSONL artifacts when `--al-handoff-audit-path` is set (or through default audit paths).
