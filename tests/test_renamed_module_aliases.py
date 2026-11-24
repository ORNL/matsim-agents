"""Old module paths that shipped on main still resolve, with a deprecation warning."""

from __future__ import annotations

import importlib
import sys

import pytest

ALIASES = [
    ("matsim_agents.discovery.seeds", "matsim_agents.discovery.seed_structures"),
    ("matsim_agents.discovery.selection", "matsim_agents.discovery.structural_screening"),
    ("matsim_agents.active_learning.candidates", "matsim_agents.active_learning.md_sampler"),
]


@pytest.mark.parametrize(("old", "new"), ALIASES)
def test_old_path_is_the_new_module_and_warns(old, new):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match=new):
        module = importlib.import_module(old)
    assert module is importlib.import_module(new)


def test_monkeypatching_through_the_old_path_reaches_the_new_module(monkeypatch):
    with pytest.warns(DeprecationWarning):
        sys.modules.pop("matsim_agents.discovery.seeds", None)
        importlib.import_module("matsim_agents.discovery.seeds")
    from matsim_agents.discovery import seed_structures

    monkeypatch.setattr("matsim_agents.discovery.seeds._pyxtal_available", lambda: "patched")
    assert seed_structures._pyxtal_available() == "patched"
