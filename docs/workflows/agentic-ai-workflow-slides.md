# Agentic AI Workflow Diagram (Slides)

Compact, left-to-right variant optimized for slide decks.

```mermaid
flowchart LR
    U[User] --> O[Objective mode]
    U --> P[Composition mode]
    U --> I[Interactive mode]

    O --> X[Shared scientific execution]
    P --> X
    I -->|explore or relax| X
    I -->|conversation only| IR[Chat response]

    X --> Q[UQ policy adapters]
    Q -->|low confidence| AL[Active learning]
    Q -->|otherwise| E[Evidence]
    AL --> E

    E --> R{Invoking mode}
    R --> OA[Analyst report]
    R --> PS[Composition summary]
    R --> IR
```

## Slide Notes

- Use this version when horizontal space is available and text should stay minimal.
- Keep node labels short to reduce line wrapping in presentation exports.
