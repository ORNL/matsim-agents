# matsim-agents

**Multi-agent AI framework for atomistic materials simulation and discovery.**

`matsim-agents` orchestrates large language models (LLMs), machine-learned interatomic potentials (MLIPs), and ASE-based atomistic workflows. Agents can plan and execute relaxations using HydraGNN, UMA, or MACE, explore candidate structures, assess stability, and report findings. External MCP calls and active-learning handoffs are optional; results require scientific review.

---

## Pages

| Page | Description |
|---|---|
| [[CI-CD]] | GitHub Actions workflows, branch protection, how to run tests locally |
| [[Contributing]] | PR process, required checks, code style, review rules |

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │                  USER                        │
                │  natural-language objective / chat dialogue  │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │              LangGraph workflow              │
                │                                              │
                │   planner → executor loop → analyst          │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │         Relaxations or MCP client calls       │
                │       optional active-learning handoff        │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │   MLIP: HydraGNN / UMA / MACE; ASE           │
                │   DFT (active learning): VASP / QE            │
                └──────────────────────────────────────────────┘
```

Chat and supervisor workflows also use a discovery wrapper for composition
parsing, pymatgen AFLOW prototype enumeration, optional pyXtal random seeds,
relaxation, and stability analysis. See the [architecture guide](https://github.com/ORNL/matsim-agents/blob/main/docs/guide/architecture.md)
and [MCP guide](https://github.com/ORNL/matsim-agents/blob/main/docs/mcp.md).

---

## Quick links

- [Source code](https://github.com/ORNL/matsim-agents)
- [README](https://github.com/ORNL/matsim-agents/blob/main/README.md)
- [License (BSD 3-Clause)](https://github.com/ORNL/matsim-agents/blob/main/LICENSE)
- [pyproject.toml](https://github.com/ORNL/matsim-agents/blob/main/pyproject.toml)

---

*Maintained by the ORNL Multi-Agentic AI for Materials team.*
