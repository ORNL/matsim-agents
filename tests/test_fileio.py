from __future__ import annotations

import os
import stat

import pytest

from matsim_agents.core.fileio import append_line, atomic_write_text


def test_atomic_write_keeps_the_old_file_when_replace_fails(tmp_path, monkeypatch):
    target = atomic_write_text(tmp_path / "sub" / "state.json", "old")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("matsim_agents.core.fileio.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [path.name for path in target.parent.iterdir()] == ["state.json"]


def test_atomic_write_uses_ordinary_file_permissions(tmp_path):
    umask = os.umask(0o022)
    os.umask(umask)
    target = atomic_write_text(tmp_path / "report.md", "text")
    assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~umask


def test_append_line_terminates_each_record_once(tmp_path):
    ledger = tmp_path / "nested" / "events.jsonl"
    append_line(ledger, '{"a": 1}')
    append_line(ledger, '{"b": 2}\n')
    assert ledger.read_text() == '{"a": 1}\n{"b": 2}\n'
