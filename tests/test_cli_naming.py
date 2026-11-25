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
    assert "explore-and-learn" in text
    assert "supervisor-run" not in text
    assert "e1-development" not in text


def test_benchmark_e1_is_discoverable_and_old_name_still_works():
    assert "e1" in _root().commands["benchmark"].commands
    new_help = _help("benchmark", "e1")
    old_help = _help("e1-development")
    assert "--retry-errors" in new_help and "--retry-errors" in old_help
    assert "--max-new-requests" in new_help and "--max-new-requests" in old_help


def test_explore_and_learn_and_supervisor_run_share_parameters():
    root = _root()
    new = root.commands["explore-and-learn"]
    old = root.commands["supervisor-run"]
    assert [p.name for p in new.params] == [p.name for p in old.params]
    assert "--mlip-config" in _help("explore-and-learn")
    assert "--mlip-config" in _help("supervisor-run")


@pytest.mark.parametrize("command", ["run", "plan", "chat", "explore", "workflow"])
def test_entry_commands_state_when_to_use(command):
    assert text_has_one_guidance_line(_help(command))


def text_has_one_guidance_line(text: str) -> bool:
    return len(re.findall(r"When\s+to\s+use:", text)) == 1


def test_radiation_damage_help_states_scope():
    text = _help("radiation-damage")
    assert "vacancy-hop BCC workflow" in re.sub(r"\s+", " ", text)
