"""Qualification cells and protocol records (software verification only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from matsim_agents.backends.dft.qe_protocols import load_protocols, qualified_by
from matsim_agents.workflows.qualification_cells import NOT_PRODUCTION, MatrixSpec, generate

ROOT = Path(__file__).resolve().parents[1]
SPEC = MatrixSpec(
    matrix_element="Cu",
    structure="fcc",
    lattice_A=3.6,
    supercell=(2, 2, 2),
    solutes=["Ni"],
    interstitial_species=["He"],
    gas=["H", "He"],
    training_configs=2,
)


def test_cells_cover_every_kind_and_are_labelled(tmp_path):
    manifest = generate(SPEC, tmp_path)
    assert manifest["note"] == NOT_PRODUCTION
    kinds = {c["kind"] for c in manifest["cases"]}
    assert {"bulk", "vacancy", "substitution", "interstitial", "gas", "neb_image"} <= kinds
    sizes = {c["name"]: c["natoms"] for c in manifest["cases"]}
    assert sizes["vacancy"] == sizes["bulk"] - 1
    assert sizes["int_He"] == sizes["bulk"] + 1
    assert sizes["gas_He"] == 1 and max(sizes.values()) < 100


def test_generation_is_reproducible(tmp_path):
    generate(SPEC, tmp_path / "a")
    generate(SPEC, tmp_path / "b")
    assert (tmp_path / "a/train0.extxyz").read_text() == (tmp_path / "b/train0.extxyz").read_text()


def test_all_seven_protocols_are_unqualified_and_have_domains():
    protocols, criteria = load_protocols(ROOT / "deployments/common/qe-protocols.yaml")
    assert len(protocols) == 7
    assert criteria.energy_per_atom_meV == 1.0 and criteria.stress_GPa == 0.1
    assert all(p.status == "unqualified" and p.domain for p in protocols.values())
    assert all(v is None for p in protocols.values() for v in p.settings.values())


@pytest.mark.parametrize(
    ("report", "covered", "ok"),
    [
        (
            {"status": "qualified", "label": ""},
            {"bulk", "vacancy", "substitution", "interstitial"},
            True,
        ),
        (
            {"status": "not_converged", "label": ""},
            {"bulk", "vacancy", "substitution", "interstitial"},
            False,
        ),
        (
            {"status": "qualified", "label": "software verification only"},
            {"bulk", "vacancy", "substitution", "interstitial"},
            False,
        ),
        ({"status": "qualified", "label": ""}, {"bulk"}, False),
    ],
)
def test_qualification_needs_real_converged_report_covering_the_domain(report, covered, ok):
    protocols, _ = load_protocols(ROOT / "deployments/common/qe-protocols.yaml")
    assert qualified_by(protocols["point_defects"], report, covered)[0] is ok
