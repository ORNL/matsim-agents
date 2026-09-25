#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-all}"

run_docs() {
    python scripts/diagnostics/validate_documentation.py
    python scripts/diagnostics/validate_deployments.py
    python benchmarks/codabench/validate_bundle.py
}

run_lint() {
    ruff check src/ tests/ scripts/diagnostics/validate_documentation.py
    ruff format --check src/ tests/ scripts/diagnostics/validate_documentation.py
    run_docs
}

run_tests() {
    python -m pytest tests/ -v --tb=short -m "not gpu" -p no:warnings
}

run_coverage() {
    python -m pytest tests/ -v --tb=short -m "not gpu" \
        --cov=matsim_agents --cov-report=term-missing \
        --cov-report=xml:coverage.xml -p no:warnings
}

case "${MODE}" in
    docs) run_docs ;;
    lint) run_lint ;;
    test) run_tests ;;
    coverage) run_coverage ;;
    all)
        run_lint
        run_tests
        ;;
    *)
        echo "Usage: $0 [all|docs|lint|test|coverage]" >&2
        exit 2
        ;;
esac
