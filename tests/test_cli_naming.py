"""Preferred CLI names, backward-compatible aliases, and usage guidance in help text."""

from __future__ import annotations

import re

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from matsim_agents.cli import app


def _help(*args: str) -> str:
    result = CliRunner().invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def _root():
    return get_command(app)


def test_active_learning_group_and_hidden_alias_share_commands():
    root = _root()
    assert root.commands["active-learning"].commands.keys() == root.commands["al"].commands.keys()
    assert "run" in root.commands["active-learning"].commands
    assert "validate-config" in _help("al")
    assert "validate-config" in _help("active-learning")


def test_alias_is_hidden_from_root_help_but_preferred_name_is_listed():
    text = _help()
    assert "active-learning" in text
    assert "supervisor-run" not in text
    assert "e1-development" not in text


@pytest.mark.parametrize(
    "command", ["validate", "plan", "run", "resume", "objective", "chat", "explore"]
)
def test_entry_commands_state_when_to_use(command):
    assert text_has_one_guidance_line(_help(command))


def text_has_one_guidance_line(text: str) -> bool:
    return len(re.findall(r"When\s+to\s+use:", text)) == 1
