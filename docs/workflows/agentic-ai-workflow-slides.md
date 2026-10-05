# Agentic AI Workflow Diagram (Slides)

Compact, left-to-right variant optimized for slide decks.

```mermaid
flowchart LR
    U[User] --> G{Mode}
    G --> O[Objective]
    G --> P[Composition]
    G --> I[Interactive]

    O --> X[Scientific execution]
    P --> X
    I --> X

    X --> Q[UQ policy adapters]
    Q -->|low confidence| AL[Active learning]
    Q -->|otherwise| E[Evidence]
    AL --> E

    E --> R{Return}
    R --> OA[Analyst]
    R --> PS[Summary]
    R --> IR[Chat]

    classDef entry fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef mode fill:#eff6ff,stroke:#2563eb,color:#172554
    classDef science fill:#ecfdf5,stroke:#059669,color:#052e16
    classDef decision fill:#fff7ed,stroke:#ea580c,color:#431407
    classDef output fill:#faf5ff,stroke:#9333ea,color:#3b0764
    class U entry
    class G,R decision
    class O,P,I mode
    class X,Q,AL,E science
    class OA,PS,IR output
```

## Slide Notes

- Use this version when horizontal space is available and text should stay minimal.
- Keep node labels short to reduce line wrapping in presentation exports.
