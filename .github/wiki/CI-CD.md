# CI / CD

This page describes the Continuous Integration setup for `matsim-agents`, how to run tests locally, and how branch protection rules prevent broken code from reaching `main`.

---

## Overview

Every push to `main` or a `feature/**` branch, and every pull request targeting `main`, triggers the CI pipeline defined in [`.github/workflows/ci.yml`](https://github.com/ORNL/matsim-agents/blob/main/.github/workflows/ci.yml).

**A pull request cannot be merged into `main` until the `CI passed` gate job turns green.**

---

## Workflow jobs

The pipeline has five jobs that run in the order shown below.

```
lint  ─────────────────────────────────────────────────────────► ci-passed
smoke (Python 3.11 / 3.12 / 3.13 / 3.14) ──► integration ──► full ──► ci-passed
```

| Job | Trigger dependency | Python | What it does |
|---|---|---|---|
| **Lint + type check** | always | 3.11 | `bash scripts/check.sh lint` (ruff, documentation, deployment, and Codabench validators); `mypy` (informational) |
| **Smoke tests** | always | 3.11, 3.12, 3.13, 3.14 | LLM provider factory tests — no model weights, CPU only |
| **Integration tests** | after smoke | 3.11 | Full agent graph + chat-flow tests with mocked LLM |
| **Full test suite + coverage** | after smoke + integration | 3.11 | `bash scripts/check.sh coverage`: all non-GPU tests, fails below 80% coverage, uploads `coverage.xml` |
| **CI passed** | after all four above | — | Gate job: fails if any upstream job failed or was cancelled |

### Lint + type check

Runs `bash scripts/check.sh lint`, which performs:

```bash
ruff check src/ tests/ scripts/diagnostics/validate_documentation.py
ruff format --check src/ tests/ scripts/diagnostics/validate_documentation.py
python scripts/diagnostics/validate_documentation.py   # links, imports, README CLI reference
python scripts/diagnostics/validate_deployments.py
python benchmarks/codabench/validate_bundle.py
```

`mypy` runs with `--ignore-missing-imports` and is set to `continue-on-error` — type errors are **informational only** and do not block the pipeline.

### Smoke tests

```bash
pytest tests/smoke/ -v --tb=short -p no:warnings
```

Files: [`tests/smoke/test_llm_providers.py`](https://github.com/ORNL/matsim-agents/blob/main/tests/smoke/test_llm_providers.py)

Tests the `get_chat_model()` factory for the supported providers using `unittest.mock` — no real API calls or model weights are needed. The job installs `.[dev]` without the `mcp` extra.

Runs on every supported Python minor version: **3.11, 3.12, 3.13, and 3.14**.

### Integration tests

```bash
pytest tests/integration/ -v --tb=short -p no:warnings
```

Covers the agent graph and chat flow with a fake LLM, the MCP client workflow,
the active-learning dry run, and DFT warm-start drivers with mocked backends.

Requires smoke tests to pass first. The job installs `.[dev,mcp]`.

### Full test suite + coverage

```bash
bash scripts/check.sh coverage
```

Runs every test except those marked `@pytest.mark.gpu`, with coverage, and fails
if total coverage is below 80%. The job installs `.[dev,mcp]` and a CPU build of
PyTorch for the MLIP regression tests.

The `coverage.xml` artifact is uploaded and retained for 7 days.

> **Note:** Tests that need an optional dependency are skipped only where that
> dependency is missing. `pymatgen` is installed with the core `pyxtal`
> dependency, so tests guarded by `@requires_pymatgen` run in CI.

### CI passed (gate job)

```yaml
needs: [lint, smoke, integration, full]
if: always()
```

This job collects the results of all four upstream jobs. It exits with code 1 if any upstream job **failed** or was **cancelled** (skipped tests are allowed). It is the single required status check registered with GitHub branch protection.

---

## Branch protection rules

The `main` branch is protected by rules set in the repository settings (Settings → Branches):

| Rule | Setting |
|---|---|
| Required status check | `CI passed` must be green |
| Branch must be up to date | Yes (`strict: true`) — rebase or merge `main` before merging your PR |
| Required approving reviews | 1; admins can bypass the rules and merge without approval |
| Dismiss stale reviews | No |
| Conversation resolution required | Yes — all PR review comments, including Copilot's, must be resolved |
| Force-push to `main` | Blocked |
| Deletion of `main` | Blocked |

---

## Running tests locally

### Prerequisites

```bash
cd matsim-agents
pip install -e ".[dev,mcp]"
pip install langchain-ollama langchain-openai langchain-anthropic
```

`bash scripts/check.sh` runs the lint step and the non-GPU test suite in one
command; `bash scripts/check.sh docs|lint|test|coverage` runs one stage.

### Run individual tiers

```bash
# Lint, documentation, and deployment checks
bash scripts/check.sh lint

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
pytest tests/test_seeds.py -v
pytest tests/smoke/test_llm_providers.py -v -k "ollama"
```

### Test marks

| Mark | Meaning |
|---|---|
| `@pytest.mark.gpu` | Requires a GPU; skipped in CI |
| `@pytest.mark.skipif(not <dependency>_available, ...)` | Requires an optional dependency; skipped when it is not installed |

---

## Test layout

| Location | Tier | Scope |
|---|---|---|
| `tests/smoke/` | Smoke | LLM provider factory, CPU only, mocked |
| `tests/integration/` | Integration | Agent graph, chat flow, MCP client, active-learning dry run, and DFT warm-start drivers with fakes or mocks |
| `tests/` | Full | Discovery, workflows, backends, documentation and deployment contracts |

Run `pytest --collect-only -q` for the current test inventory.

---

## Environment variables used in CI

| Variable | Value | Purpose |
|---|---|---|
| `MATSIM_LLM_PROVIDER` | `ollama` | Prevents accidental real provider calls |
| `HF_HUB_OFFLINE` | `1` | Blocks HuggingFace Hub network access |
| `TRANSFORMERS_OFFLINE` | `1` | Blocks Transformers model downloads |
