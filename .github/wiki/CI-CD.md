# CI / CD

This page describes the Continuous Integration setup for `matsim-agents`, how to run tests locally, and how branch protection rules prevent broken code from reaching `main`.

---

## Overview

Every push and every pull request targeting `main` triggers the CI pipeline defined in [`.github/workflows/ci.yml`](https://github.com/ORNL/matsim-agents/blob/main/.github/workflows/ci.yml).

**A pull request cannot be merged into `main` until the `CI passed` gate job turns green.**

---

## Workflow jobs

The pipeline has five jobs that run in the order shown below.

```
lint  ─────────────────────────────────────────────────────────► ci-passed
smoke (Python 3.10 / 3.11 / 3.12) ──► integration ──► full ──► ci-passed
```

| Job | Trigger dependency | Python | What it does |
|---|---|---|---|
| **Lint + type check** | always | 3.11 | `ruff check`, `ruff format --check`, `mypy` (informational) |
| **Smoke tests** | always | 3.10, 3.11, 3.12 | LLM provider factory tests — no model weights, CPU only |
| **Integration tests** | after smoke | 3.11 | Full agent graph + chat-flow tests with mocked LLM |
| **Full test suite + coverage** | after smoke + integration | 3.11 | All non-GPU tests, uploads `coverage.xml` artifact |
| **CI passed** | after all four above | — | Gate job: fails if any upstream job failed or was cancelled |

### Lint + type check

Runs [ruff](https://docs.astral.sh/ruff/) in two passes:

```bash
ruff check src/ tests/     # lint rules E, F, I, B, UP, SIM
ruff format --check src/ tests/   # formatting check (black==21.5b1 style)
```

`mypy` runs with `--ignore-missing-imports` and is set to `continue-on-error` — type errors are **informational only** and do not block the pipeline.

### Smoke tests

```bash
pytest tests/smoke/ -v --tb=short -p no:warnings
```

Files: [`tests/smoke/test_llm_providers.py`](https://github.com/ORNL/matsim-agents/blob/main/tests/smoke/test_llm_providers.py)

Tests the `get_chat_model()` factory for every supported provider (Ollama, vLLM, OpenAI, Anthropic, HuggingFace) using `unittest.mock` — no real API calls or model weights are needed.

Runs on Python **3.10, 3.11, 3.12** in parallel.

### Integration tests

```bash
pytest tests/integration/ -v --tb=short -p no:warnings
```

Files:
- [`tests/integration/test_agent_graph.py`](https://github.com/ORNL/matsim-agents/blob/main/tests/integration/test_agent_graph.py) — planner / executor / analyst LangGraph nodes with a fake LLM
- [`tests/integration/test_chat_flow.py`](https://github.com/ORNL/matsim-agents/blob/main/tests/integration/test_chat_flow.py) — `chat_once()` and `DiscoveryChatSession` with mocked exploration

Requires smoke tests to pass first.

### Full test suite + coverage

```bash
pytest tests/ -v --tb=short -m "not gpu" \
  --cov=matsim_agents \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  -p no:warnings
```

Runs everything (smoke + integration + phase explorer + discovery + state/graph tests), skipping tests marked `@pytest.mark.gpu`.

The `coverage.xml` artifact is uploaded and retained for 7 days.

> **Note:** Tests that require `pymatgen` (e.g. `test_double_perovskite_detected`, `test_spinel_detected_for_AB2X4`) are decorated with `@requires_pymatgen` and automatically **skipped** on CI where pymatgen is not installed. They run normally on HPC environments where pymatgen is available.

### CI passed (gate job)

```yaml
needs: [lint, smoke, integration, full]
if: always()
```

This job collects the results of all four upstream jobs. It exits with code 1 if any upstream job **failed** or was **cancelled** (skipped tests are allowed). It is the single required status check registered with GitHub branch protection.

---

## Branch protection rules

The `main` branch is protected by rules configured via [`.github/workflows/branch-protection.yml`](https://github.com/ORNL/matsim-agents/blob/main/.github/workflows/branch-protection.yml), which applies the following settings automatically via the GitHub API when either workflow file changes:

| Rule | Setting |
|---|---|
| Required status check | `CI passed` must be green |
| Branch must be up to date | Yes (`strict: true`) — rebase or merge `main` before merging your PR |
| Required approving reviews | 1 |
| Dismiss stale reviews | No |
| Conversation resolution required | Yes — all PR review comments must be resolved |
| Force-push to `main` | Blocked |
| Deletion of `main` | Blocked |

---

## Running tests locally

### Prerequisites

```bash
cd matsim-agents
pip install -e ".[dev]"
pip install langchain-ollama langchain-openai langchain-anthropic
```

### Run individual tiers

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Smoke tests only (fast, no weights)
pytest tests/smoke/ -v

# Integration tests
pytest tests/integration/ -v

# Full suite (skip GPU tests)
pytest tests/ -v -m "not gpu"

# Full suite with coverage
pytest tests/ -v -m "not gpu" --cov=matsim_agents --cov-report=term-missing
```

### Run a single test file

```bash
pytest tests/test_phase_explorer.py -v
pytest tests/smoke/test_llm_providers.py -v -k "ollama"
```

### Test marks

| Mark | Meaning |
|---|---|
| `@pytest.mark.gpu` | Requires a GPU; skipped in CI |
| `@pytest.mark.skipif(not pymatgen_available, ...)` | Requires pymatgen; skipped when not installed |

---

## Test inventory

| File | Tier | Count | Notes |
|---|---|---|---|
| `tests/smoke/test_llm_providers.py` | Smoke | 12 | LLM factory, all providers, CPU, mocked |
| `tests/integration/test_agent_graph.py` | Integration | 12 | Planner/executor/analyst graph, fake LLM |
| `tests/integration/test_chat_flow.py` | Integration | 9 | `chat_once()`, `DiscoveryChatSession`, mocked exploration |
| `tests/test_phase_explorer.py` | Full | 12 | Crystal-phase seed enumeration (2 require pymatgen) |
| `tests/test_discovery.py` | Full | — | Discovery wrapper |
| `tests/test_state_and_graph.py` | Full | — | `MatSimState`, LangGraph wiring |

---

## Environment variables used in CI

| Variable | Value | Purpose |
|---|---|---|
| `MATSIM_LLM_PROVIDER` | `ollama` | Prevents accidental real provider calls |
| `HF_HUB_OFFLINE` | `1` | Blocks HuggingFace Hub network access |
| `TRANSFORMERS_OFFLINE` | `1` | Blocks Transformers model downloads |
