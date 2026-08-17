# Package architecture

The library is organized around stable scientific responsibilities rather than
individual experiments or machines.

| Package | Responsibility |
| --- | --- |
| `orchestration` | Typed workflow state, objective/composition graphs, and explicit policies |
| `discovery` | Composition parsing, structure generation, and stability analysis |
| `active_learning` | Candidate acquisition, uncertainty evaluation, labeling, and adaptation loops |
| `backends.llm` | Configuration-selected language-model providers |
| `backends.mlip` | MLIP calculators, relaxation, and exploration interfaces |
| `backends.dft` | Quantum ESPRESSO and VASP interfaces |
| `execution` | Scheduler-neutral resources, launch contracts, and provenance |

Machine-specific setup and job scripts live in `deployments/`; research-only
paper and Codabench artifacts live in `research/`.

## Compatibility policy

The first migration release retains aliases at the former import paths (for
example, `matsim_agents.graph`, `matsim_agents.state`, and
`matsim_agents.tools.relaxation`).  New code should import from the canonical
packages.  The aliases can be deprecated in a later release after downstream
users have migrated.
