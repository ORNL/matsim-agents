from __future__ import annotations

from pathlib import Path

import pytest

from matsim_agents.workflows import radiation_capabilities as rc

ROOT = Path(__file__).resolve().parents[1]


def test_names_are_unique():
    names = [c.name for c in rc.RADIATION_CAPABILITIES]
    assert len(names) == len(set(names))


def test_every_entry_point_resolves():
    for capability in rc.RADIATION_CAPABILITIES:
        for entry in capability.entry_points:
            assert rc._resolve(entry) is None, entry


def test_every_declared_test_file_exists():
    for capability in rc.RADIATION_CAPABILITIES:
        assert capability.tests, capability.name
        for test in capability.tests:
            assert (ROOT / test).is_file(), test


def test_declared_backends_are_registered_backends():
    from matsim_agents.execution.capabilities import get_capability

    for capability in rc.RADIATION_CAPABILITIES:
        for backend in capability.backends:
            get_capability(backend)


def test_unresolvable_entry_point_marks_software_unavailable():
    broken = rc.RadiationCapability(
        name="broken",
        summary="x",
        entry_points=("matsim_agents.science.kinetics.rates:no_such_name", "no_such_pkg.mod:f"),
    )
    result = rc.evaluate_capability(broken)
    assert result.software is rc.SoftwareReadiness.UNAVAILABLE
    assert len(result.problems) == 2


def test_missing_backend_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(rc, "check", lambda cap: type("C", (), {"available": False})())
    result = rc.evaluate_capability(rc.get_radiation_capability("neb-preparation"))
    assert result.runtime is rc.RuntimeReadiness.MISSING_BACKEND
    assert result.missing_backends == ("qe",)
    assert result.software is rc.SoftwareReadiness.IMPLEMENTED


def test_present_backend_gives_ready(monkeypatch):
    monkeypatch.setattr(rc, "check", lambda cap: type("C", (), {"available": True})())
    result = rc.evaluate_capability(rc.get_radiation_capability("neb-preparation"))
    assert result.runtime is rc.RuntimeReadiness.READY
    assert result.missing_backends == ()


def test_capability_without_backends_needs_no_runtime():
    result = rc.evaluate_capability(rc.get_radiation_capability("kinetics"))
    assert result.runtime is rc.RuntimeReadiness.NOT_REQUIRED


def test_nothing_is_scientifically_qualified():
    for result in rc.evaluate_all():
        assert result.qualification is rc.QualificationLevel.NOT_QUALIFIED
    assert list(rc.QualificationLevel) == [rc.QualificationLevel.NOT_QUALIFIED]


def test_unknown_capability_raises():
    with pytest.raises(KeyError, match="known:"):
        rc.get_radiation_capability("nope")
