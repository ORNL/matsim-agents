# matsim-agents

**Multi-agent AI framework for atomistic materials simulation and discovery.**

`matsim-agents` orchestrates large language models (LLMs), machine-learned interatomic potentials (MLIPs), and ASE-based atomistic workflows into a single agentic loop. The user states a research objective in natural language; agents plan, run HydraGNN-driven simulations, score chemical and dynamical stability, and report the findings — with optional human review at every gate.

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
                │   planner ───► executor ──┐                  │
                │                  ▲        │                  │
                │                  └────────┤  while pending   │
                │                           ▼                  │
                │                        analyst ──► END       │
                └───────────────────────┬──────────────────────┘
                                        │  tool calls
                ┌───────────────────────▼──────────────────────┐
                │              Discovery wrapper               │
                │   composition parsing → phase enumeration    │
                │   → relaxation (HydraGNN+ASE) → stability    │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
                │             Atomistic backends               │
                │   HydraGNN (fused MLFF + BranchWeightMLP)    │
                │   ASE (FIRE / BFGS / BFGSLineSearch)         │
                │   pymatgen (optional prototypes)             │
                └──────────────────────────────────────────────┘
```

---

## Quick links

- [Source code](https://github.com/ORNL/matsim-agents)
- [README](https://github.com/ORNL/matsim-agents/blob/main/README.md)
- [License (BSD 3-Clause)](https://github.com/ORNL/matsim-agents/blob/main/LICENSE)
- [pyproject.toml](https://github.com/ORNL/matsim-agents/blob/main/pyproject.toml)

---

*Maintained by the ORNL Multi-Agentic AI for Materials team.*
