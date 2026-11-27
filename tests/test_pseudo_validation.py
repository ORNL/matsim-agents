"""Pseudopotential qualification: the minimum cutoffs come from the files, never a default."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from ase.build import bulk
from synthetic_upf import write_synthetic_upf

from matsim_agents.backends.dft.pseudo_validation import (
    PseudopotentialError,
    main,
    require_valid_pseudopotentials,
    validate_pseudopotentials,
)
from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input

REAL_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "deployments/common/pseudopotentials/pslibrary-1.0.0-pbe-paw.yaml"
)
REAL_DIR = Path("/lustre/orion/lrn110/proj-shared/cache/pseudopotentials/pslibrary-1.0.0-pbe-paw")


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    write_synthetic_upf(tmp_path, "A.upf", "A", wfc=50.0, rho=300.0)
    write_synthetic_upf(tmp_path, "B.upf", "B", wfc=40.0, rho=600.0)
    return tmp_path


MAPPING = {"A": "A.upf", "B": "B.upf"}


def test_minimum_is_the_maximum_over_elements(pseudo_dir):
    result = validate_pseudopotentials(MAPPING, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0)
    assert result.minimum_ecutwfc_ry == 50.0
    assert result.minimum_ecutrho_ry == 600.0
    assert result.qualified


def test_below_recommendation_is_rejected(pseudo_dir):
    # Satisfies A's wavefunction cutoff and B's density cutoff, but not each element's both.
    result = validate_pseudopotentials(MAPPING, pseudo_dir, ecutwfc_ry=45.0, ecutrho_ry=700.0)
    assert not result.qualified and result.below_recommendation
    with pytest.raises(PseudopotentialError):
        require_valid_pseudopotentials(MAPPING, pseudo_dir, ecutwfc_ry=45.0, ecutrho_ry=700.0)


def test_missing_file_wrong_element_and_wrong_xc(pseudo_dir):
    result = validate_pseudopotentials(
        {"A": "absent.upf", "B": "B.upf"}, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0
    )
    assert not result.qualified and result.problems
    result = validate_pseudopotentials(
        {"A": "B.upf", "B": "B.upf"}, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0
    )
    assert any("element" in p for p in result.problems)
    result = validate_pseudopotentials(
        MAPPING, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0, functional="PBEsol"
    )
    assert not result.qualified and result.problems


def test_unavailable_cutoffs_are_not_guessed(tmp_path):
    (tmp_path / "A.upf").write_text(
        '<UPF version="2.0.1"><PP_HEADER element="A" functional="SLA PW PBX PBC" '
        'z_valence="4"/></UPF>'
    )
    result = validate_pseudopotentials(
        {"A": "A.upf"}, tmp_path, ecutwfc_ry=999.0, ecutrho_ry=9999.0
    )
    assert result.minimum_ecutwfc_ry is None
    assert result.unavailable == ["A"] and not result.qualified


def test_override_is_diagnostic_and_never_qualified(pseudo_dir):
    result = validate_pseudopotentials(
        MAPPING, pseudo_dir, ecutwfc_ry=30.0, ecutrho_ry=200.0, override_reason="scan point"
    )
    assert result.diagnostic_only and not result.qualified
    require_valid_pseudopotentials(
        MAPPING, pseudo_dir, ecutwfc_ry=30.0, ecutrho_ry=200.0, override_reason="scan point"
    )
    # The override covers cutoffs only: a missing file still fails.
    with pytest.raises(PseudopotentialError):
        require_valid_pseudopotentials(
            {"A": "absent.upf", "B": "B.upf"},
            pseudo_dir,
            ecutwfc_ry=30.0,
            ecutrho_ry=200.0,
            override_reason="scan point",
        )


def test_checksum_mismatch_against_manifest(pseudo_dir):
    good = hashlib.sha256((pseudo_dir / "A.upf").read_bytes()).hexdigest()
    entry = {"filename": "A.upf", "sha256": good}
    manifest = {"pseudopotentials": {"A": dict(entry)}}
    ok = validate_pseudopotentials(
        {"A": "A.upf"}, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0, manifest=manifest
    )
    assert ok.qualified
    manifest["pseudopotentials"]["A"]["sha256"] = "0" * 64
    bad = validate_pseudopotentials(
        {"A": "A.upf"}, pseudo_dir, ecutwfc_ry=60.0, ecutrho_ry=700.0, manifest=manifest
    )
    assert any("checksum" in p.lower() or "sha256" in p.lower() for p in bad.problems)


def test_cli_exit_codes(pseudo_dir, capsys):
    args = [str(pseudo_dir), "--map", "A=A.upf", "--map", "B=B.upf"]
    assert main([*args, "--ecutwfc", "60", "--ecutrho", "700"]) == 0
    assert main([*args, "--ecutwfc", "45", "--ecutrho", "700"]) == 1
    capsys.readouterr()


def test_pw_writer_rejects_below_recommendation(pseudo_dir, tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    write_synthetic_upf(pseudo_dir, "Cu.upf", "Cu", wfc=50.0, rho=400.0)
    settings = QESettings(
        pseudo_dir=str(pseudo_dir),
        pseudopotentials={"Cu": "Cu.upf"},
        ecutwfc_ry=40.0,
        ecutrho_ry=320.0,
        kpts=(2, 2, 2),
    )
    with pytest.raises(PseudopotentialError):
        write_pw_input(atoms, settings, str(tmp_path / "pw.in"))
    settings.ecutwfc_ry, settings.ecutrho_ry = 50.0, 400.0
    write_pw_input(atoms, settings, str(tmp_path / "pw.in"))
    assert (tmp_path / "pw.in").is_file()


@pytest.mark.skipif(not REAL_DIR.is_dir(), reason="PSlibrary cache not on this host")
def test_real_library_accepts_60_720_and_rejects_below(tmp_path):
    from matsim_agents.backends.dft.pseudo_validation import load_manifest

    manifest = load_manifest(REAL_MANIFEST)
    mapping = {el: entry["filename"] for el, entry in manifest["pseudopotentials"].items()}
    ok = validate_pseudopotentials(
        mapping, REAL_DIR, ecutwfc_ry=60.0, ecutrho_ry=720.0, manifest=manifest
    )
    assert ok.qualified
    assert ok.minimum_ecutwfc_ry == pytest.approx(53.793, abs=1e-3)
    assert ok.minimum_ecutrho_ry == pytest.approx(645.077, abs=1e-3)
    low = validate_pseudopotentials(
        mapping, REAL_DIR, ecutwfc_ry=51.0, ecutrho_ry=650.0, manifest=manifest
    )
    assert not low.qualified
