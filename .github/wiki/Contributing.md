# Contributing

Thank you for contributing to `matsim-agents`! This page explains the pull request process, required checks, code style, and review rules.

---

## Quick checklist

Before opening a PR, make sure all of these pass locally:

```bash
pip install -e ".[dev]"
pip install langchain-ollama langchain-openai langchain-anthropic

ruff check src/ tests/          # lint
ruff format src/ tests/         # auto-format
pytest tests/ -v -m "not gpu"   # full test suite (no GPU)
```

---

## Branching model

| Branch pattern | Purpose |
|---|---|
| `main` | Protected. Only merged PRs land here. |
| `dev` | Integration branch for feature work (CI runs here too). |
| `feature/<name>` | Short-lived feature branches. CI runs on push. |

**Never push directly to `main`.** Open a pull request from your branch.

---

## Pull request process

1. **Fork** the repository (external contributors) or create a branch (team members).
2. Write your changes. Keep commits focused — one logical change per commit.
3. Push to your branch and open a PR targeting `main`.
4. **All CI checks must pass** before the PR can be merged:
   - `Lint + type check` — ruff lint, ruff format, mypy
   - `Smoke tests (Python 3.10)`, `Smoke tests (Python 3.11)`, `Smoke tests (Python 3.12)`
   - `Integration tests (agent graph + chat)`
   - `Full test suite + coverage`
   - **`CI passed`** — the gate job that summarises all of the above
5. **At least 1 approving review** is required.
6. **All PR conversations must be resolved** before merging.
7. **The branch must be up to date with `main`** at merge time (`strict` branch protection is on — rebase or use the "Update branch" button).

---

## Code style

### Formatter

`ruff format` is the canonical formatter, configured to match `black==21.5b1` style:

```toml
[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
preview = false
```

Run `ruff format src/ tests/` before pushing. The CI `ruff format --check` step will fail if the code is not formatted.

### Linter

`ruff check` enforces rules `E`, `F`, `I`, `B`, `UP`, `SIM` with `line-length = 100` and `target-version = "py310"`.

Run `ruff check src/ tests/` and fix any errors. Auto-fixable issues can be resolved with `ruff check --fix src/ tests/`.

Some pre-existing files in `src/matsim_agents/` have `per-file-ignores` entries in `pyproject.toml` — do not add new ignores without a good reason.

### Type annotations

`mypy` is run with `--ignore-missing-imports`. Type errors are **informational** (the mypy step is `continue-on-error`) but new public functions should have type annotations where practical.

---

## Testing

### Write tests for new behaviour

- New features → add tests in the appropriate tier (`tests/smoke/`, `tests/integration/`, or root `tests/`).
- Bug fixes → add a regression test that would have caught the bug.

### Test tiers

| Tier | Directory | Scope |
|---|---|---|
| Smoke | `tests/smoke/` | LLM factory, CPU only, fast (<10 s). No model weights, no GPU. |
| Integration | `tests/integration/` | Agent graph + chat flow with fake/mocked LLM. |
| Full | `tests/` (root) | Phase explorer, discovery, state/graph. Run with `-m "not gpu"` on CI. |

### Optional-dependency tests

If your test requires an optional dependency (e.g. `pymatgen`), guard it with:

```python
import importlib.util
import pytest

pymatgen_available = importlib.util.find_spec("pymatgen") is not None
requires_pymatgen = pytest.mark.skipif(
    not pymatgen_available, reason="pymatgen not installed"
)

@requires_pymatgen
def test_my_feature():
    ...
```

Tests guarded this way are **skipped** on CI (where pymatgen is not installed) and run normally on HPC environments.

### GPU tests

Mark tests that require a GPU with `@pytest.mark.gpu`. They are excluded in CI via `-m "not gpu"`.

---

## Commit messages

Use the conventional commits format:

```
<type>(<scope>): <short summary>

[optional body]
```

Common types: `feat`, `fix`, `refactor`, `test`, `ci`, `docs`, `style`, `chore`.

Examples:
```
feat(discovery): add rutile phase for binary 1:2 compositions
fix(tests): skip pymatgen-dependent tests when pymatgen is not installed
ci: add gate job and auto-configure branch protection on main
```

---

## CI / CD details

See the [[CI-CD]] wiki page for a full description of the pipeline, branch protection rules, and how to run tests locally.
