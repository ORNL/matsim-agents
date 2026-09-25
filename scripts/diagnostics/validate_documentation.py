#!/usr/bin/env python3
"""Validate documentation claims against the current repository and package."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTATION_ROOTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs",
    REPOSITORY_ROOT / ".github" / "wiki",
    REPOSITORY_ROOT / "deployments",
    REPOSITORY_ROOT / "examples",
    REPOSITORY_ROOT / "benchmarks",
)
MARKDOWN_LINK = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")
PYTEST_PATH = re.compile(r"(?:python\s+-m\s+)?pytest\s+(?P<path>[\w./-]+\.py)\b")
PYTHON_FENCE = re.compile(r"```python\s*\n(?P<code>.*?)```", re.DOTALL)
CLI_REFERENCE = re.compile(r"## CLI reference\s*\n\s*```text\s*\n(?P<body>.*?)```", re.DOTALL)
CLI_COMMAND = re.compile(r"^matsim-agents\s+(?P<command>[\w-]+)", re.MULTILINE)


def documentation_files() -> list[Path]:
    files: set[Path] = set()
    for root in DOCUMENTATION_ROOTS:
        if root.is_file():
            files.add(root)
        elif root.is_dir():
            files.update(root.rglob("*.md"))
    return sorted(files)


def validate_local_file_targets(documents: list[Path]) -> list[str]:
    violations: list[str] = []
    for document in documents:
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in MARKDOWN_LINK.finditer(line):
                file_target = match.group("target").strip().strip("<>").split("#", 1)[0]
                if not file_target or "://" in file_target or file_target.startswith("mailto:"):
                    continue
                if not (document.parent / unquote(file_target)).resolve().exists():
                    relative = document.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number}: missing {file_target}")
    return violations


def validate_documented_test_paths(documents: list[Path]) -> list[str]:
    violations: list[str] = []
    for document in documents:
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in PYTEST_PATH.finditer(line):
                test_path = match.group("path")
                if not (REPOSITORY_ROOT / test_path).exists():
                    relative = document.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number}: missing {test_path}")
    return violations


def validate_documented_imports(documents: list[Path]) -> list[str]:
    violations: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in PYTHON_FENCE.finditer(text):
            try:
                tree = ast.parse(match.group("code"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("matsim_agents"):
                    continue
                try:
                    module = importlib.import_module(node.module)
                except ImportError as exc:
                    relative = document.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}: cannot import {node.module}: {exc}")
                    continue
                for alias in node.names:
                    if alias.name != "*" and not hasattr(module, alias.name):
                        relative = document.relative_to(REPOSITORY_ROOT)
                        violations.append(f"{relative}: {node.module}.{alias.name} is unavailable")
    return violations


def _registered_cli_commands() -> set[str]:
    from matsim_agents.cli import app

    commands = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in app.registered_commands
    }
    commands.update(group.name for group in app.registered_groups if group.name)
    return commands


def validate_cli_reference() -> list[str]:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    match = CLI_REFERENCE.search(readme)
    if match is None:
        return ["README.md: missing fenced CLI reference"]
    documented = set(CLI_COMMAND.findall(match.group("body")))
    registered = _registered_cli_commands()
    if documented == registered:
        return []
    return [
        "README.md: CLI reference mismatch "
        f"(missing={sorted(registered - documented)}, extra={sorted(documented - registered)})"
    ]


def collect_violations() -> list[str]:
    documents = documentation_files()
    return [
        *validate_local_file_targets(documents),
        *validate_documented_test_paths(documents),
        *validate_documented_imports(documents),
        *validate_cli_reference(),
    ]


def main() -> int:
    violations = collect_violations()
    if violations:
        print("Documentation validation failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Documentation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
