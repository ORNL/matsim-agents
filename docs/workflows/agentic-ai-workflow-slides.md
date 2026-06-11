# Agentic AI Workflow Diagram (Slides)

Compact, left-to-right variant optimized for slide decks.

```mermaid
flowchart LR
    U[User objective or chat dialogue]

    U --> R0[run]
    R0 --> R1[planner]
    R1 --> R2[executor]
    R2 --> R3[uq_gate]
    R3 -->|high confidence| R4[analyst]
    R3 -->|low confidence + policy| AL[active learning loop]
    AL --> R4

    U --> S0[supervisor-run]
    S0 --> S1[prepare]
    S1 --> S2[explore]
    S2 --> S3[evaluate_uq]
    S3 -->|low confidence + policy| AL
    S3 -->|otherwise| S4[summarize]

    U --> C0[chat]
    C0 --> C1[composition detection or /relax]
    C1 --> C2[uq policy]
    C2 -->|low confidence + policy| AL
```

## Slide Notes

- Use this version when horizontal space is available and text should stay minimal.
- Keep node labels short to reduce line wrapping in presentation exports.
