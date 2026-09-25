"""Regression checks for repository-local documentation links."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")


def test_local_markdown_file_targets_resolve() -> None:
    broken: list[str] = []
    for document in REPOSITORY_ROOT.rglob("*.md"):
        lines = document.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            for match in MARKDOWN_LINK.finditer(line):
                file_target = match.group("target").strip().strip("<>").split("#", 1)[0]
                if not file_target or "://" in file_target or file_target.startswith("mailto:"):
                    continue
                resolved = (document.parent / unquote(file_target)).resolve()
                if not resolved.exists():
                    relative_document = document.relative_to(REPOSITORY_ROOT)
                    broken.append(f"{relative_document}:{line_number}: {file_target}")

    assert not broken, "Broken local Markdown file targets:\n" + "\n".join(broken)
