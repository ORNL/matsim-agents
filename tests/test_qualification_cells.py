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


BCC = MatrixSpec(
    matrix_element="Cr",
    structure="bcc",
    lattice_A=2.88,
    supercell=(3, 3, 3),
    solutes=["Ni"],
    interstitial_species=["He"],
    divacancy=True,
    dumbbells=["110", "111"],
    vacancy_gas=["He"],
    alloys=[{"name": "ni10", "fractions": {"Ni": 0.1}}],
    magnetic_elements=["Cr"],
    training_configs=0,
)


def test_defect_complex_alloy_cells_have_the_right_sizes_and_kinds(tmp_path):
    manifest = generate(BCC, tmp_path)
    cases = {c["name"]: c for c in manifest["cases"]}
    n = cases["bulk"]["natoms"]
    assert cases["divacancy_1nn"]["natoms"] == n - 2 and cases["divacancy_2nn"]["natoms"] == n - 2
    assert cases["dumbbell_110"]["natoms"] == n + 1 and cases["dumbbell_111"]["natoms"] == n + 1
    assert cases["vac_He"]["natoms"] == n and "He" in cases["vac_He"]["formula"]
    assert cases["alloy_ni10"]["kind"] == "alloy" and "Ni5" in cases["alloy_ni10"]["formula"]
    assert cases["divacancy_1nn"]["kind"] == "divacancy"
    assert cases["dumbbell_110"]["kind"] == "dumbbell" and cases["vac_He"]["kind"] == "complex"
    assert all(c["magnetic"] for c in cases.values() if "Cr" in c["formula"])


def test_divacancy_partners_are_first_and_second_neighbours(tmp_path):
    from ase.io import read

    generate(BCC, tmp_path)
    bulk_cell = read(tmp_path / "bulk.extxyz")
    for name, factor in (("1nn", 3**0.5 / 2), ("2nn", 1.0)):
        pair = read(tmp_path / f"divacancy_{name}.extxyz")
        removed = [
            p for p in bulk_cell.positions if min(abs(pair.positions - p).sum(axis=1)) > 1e-6
        ]
        assert len(removed) == 2
        distance = bulk_cell.get_distance(
            0,
            [i for i, p in enumerate(bulk_cell.positions) if (p == removed[1]).all()][0],
            mic=True,
        )
        assert distance == pytest.approx(factor * BCC.lattice_A, abs=1e-6)


def test_interstitial_position_does_not_depend_on_supercell_size(tmp_path):
    from ase.io import read

    spec = BCC.model_copy(update={"supercell": (2, 2, 2)})
    generate(spec, tmp_path)
    assert read(tmp_path / "int_He.extxyz").positions[-1] == pytest.approx(
        [BCC.lattice_A / 2, 0.0, BCC.lattice_A / 4]
    )


def test_alloy_cells_are_reproducible_and_keep_the_requested_fraction(tmp_path):
    generate(BCC, tmp_path / "a")
    generate(BCC, tmp_path / "b")
    assert (tmp_path / "a/alloy_ni10.extxyz").read_text() == (
        tmp_path / "b/alloy_ni10.extxyz"
    ).read_text()
