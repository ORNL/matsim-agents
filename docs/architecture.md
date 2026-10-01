# Package architecture

		U[User objective or dialogue] --> G{Choose interaction mode}
individual experiments or machines.

| Package | Responsibility |
			O[Objective<br/>run] --> OP[Plan and execute]
			P[Composition<br/>supervisor-run] --> PP[Prepare and explore]
			I[Interactive<br/>chat] --> IP[Detect and confirm]
| `active_learning` | Candidate acquisition, uncertainty evaluation, labeling, and adaptation loops |
| `backends.llm` | Configuration-selected language-model providers |
		G --> O
		G --> P
		G --> I

		OP --> X
		PP --> X
		IP --> X
| `workflows` | Composable relaxation, phase-exploration, and investigation policies/results |

The workflow layer uses `execution.contracts` for evidence, validation,
			Q[Evaluate uncertainty]
run directories are owned by `execution.run_directory`; scheduler allocation
discovery and disjoint DFT node grouping are owned by `execution.allocation`.
See [Scientific workflow contracts](scientific-workflows.md) for behavior and
[Run artifacts and restarts](run-artifacts-and-restarts.md) for persistence.

			Q -->|sufficient confidence| E

The CLI exposes objective, composition, and interactive modes over shared
scientific capabilities. Each mode can hand low-confidence work to the same
		U[User objective or chat dialogue]

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
		classDef output fill:#faf5ff,stroke:#9333ea,color:#3b0764
		class U entry
		class G,R decision
		class O,OP,P,PP,I,IP mode
		class X,Q,AL,E science
		class OA,PS,IR output
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

Machine-specific setup and job scripts live in `deployments/`; research-only
paper and Codabench artifacts live in `research/`.

## Five stable interfaces

Each backend boundary is defined by a `@runtime_checkable` Protocol so that
new implementations only need to satisfy the structural contract — no
inheritance required.

| Interface | Import path | Key methods / attributes |
| --- | --- | --- |
| `DFTBackend` | `matsim_agents.backends.dft` | `name`, `run_one(spec) → DFTResult` |
| `MLIPBackend` | `matsim_agents.backends.mlip` | `name`, `as_calculator() → Calculator`, `relax(atoms, *, fmax, max_steps) → RelaxationResult` |
| `LLMBackend` | `matsim_agents.backends.llm` | type alias for `langchain_core.language_models.BaseChatModel` |
| `ExecutionPlatform` | `matsim_agents.execution` | `name`, `submit(cmd, *, resources, workdir) → str`, `available_resources() → ResourceRequest` |
| `RunStore` | `matsim_agents.execution` | `append(record) → None`, `iter_records() → Iterable` |

These backend Protocols are extension boundaries. The newer scientific
workflow models are policy and result contracts layered above them, not
replacement backend interfaces.

`JsonlRunStore` in `matsim_agents.execution.provenance` is the concrete
`RunStore` implementation backed by a newline-delimited JSON file.

## Compatibility policy

The first migration release retains aliases at the former import paths (for
example, `matsim_agents.graph`, `matsim_agents.state`, and
`matsim_agents.tools.relaxation`).  New code should import from the canonical
packages.  The aliases can be deprecated in a later release after downstream
users have migrated.
