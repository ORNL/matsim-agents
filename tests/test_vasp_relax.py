"""Unit tests for matsim_agents.tools.vasp_relax.

These tests exercise the input-deck writer and the parser against canned
data; they never invoke a real VASP binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk

from matsim_agents.tools.vasp_relax import (
    VASPSettings,
    parse_relax_outputs,
    prepare_relax_workdir,
    recommend_settings,
    write_incar,
)


def _write_dummy_potcar(potcar_dir: Path, symbol: str, enmax: float) -> None:
    """Create a minimal POTCAR file for one element with a parseable ENMAX."""
    sub = potcar_dir / symbol
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "POTCAR").write_text(
        f"PAW_PBE {symbol} fake\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  = {enmax * 0.75:.3f} eV\n"
        "   parameters from PSCTR are:\n"
        " End of Dataset\n"
    )


# --------------------------------------------------------------------------- #
# recommend_settings                                                          #
# --------------------------------------------------------------------------- #


def test_recommend_settings_insulator_si() -> None:
    s = recommend_settings(bulk("Si"))
    assert s.ismear == 0
    assert s.sigma == pytest.approx(0.05)
    assert s.kspacing == pytest.approx(0.30)


def test_recommend_settings_metal_fe() -> None:
    s = recommend_settings(bulk("Fe"))
    assert s.ismear == 1
    assert s.sigma == pytest.approx(0.10)
    assert s.kspacing == pytest.approx(0.20)


def test_recommend_settings_encut_from_potcar(tmp_path: Path) -> None:
    _write_dummy_potcar(tmp_path, "Si", enmax=245.345)
    s = recommend_settings(bulk("Si"), potcar_dir=tmp_path)
    # 245.345 * 1.3 = 318.9485 → rounded to 1 dp
    assert s.encut_ev == pytest.approx(318.9, abs=0.05)


def test_recommend_settings_overrides_passthrough() -> None:
    s = recommend_settings(bulk("Si"), calculation="scf", nsw=999, ediff=1e-8)
    assert s.calculation == "scf"
    assert s.nsw == 999
    assert s.ediff == pytest.approx(1e-8)


# --------------------------------------------------------------------------- #
# write_incar                                                                 #
# --------------------------------------------------------------------------- #


def test_write_incar_scf_disables_ionic_motion(tmp_path: Path) -> None:
    s = recommend_settings(bulk("Si"), calculation="scf")
    out = write_incar(s, tmp_path / "INCAR")
    text = Path(out).read_text()
    assert "IBRION = -1" in text
    assert "ISIF   = 2" in text
    assert "NSW    = 0" in text


def test_write_incar_vc_relax_enables_cell(tmp_path: Path) -> None:
    s = recommend_settings(bulk("Si"), calculation="vc-relax", nsw=42)
    out = write_incar(s, tmp_path / "INCAR")
    text = Path(out).read_text()
    assert "IBRION = 2" in text
    assert "ISIF   = 3" in text
    assert "NSW    = 42" in text
    assert "KSPACING = 0.3" in text
    assert "KGAMMA   = .TRUE." in text


def test_write_incar_extra_overrides_appended(tmp_path: Path) -> None:
    s = recommend_settings(bulk("Si"))
    s.extra_incar = {"NCORE": 8, "MAGMOM": "16*0.0"}
    out = write_incar(s, tmp_path / "INCAR")
    text = Path(out).read_text()
    assert "NCORE = 8" in text
    assert "MAGMOM = 16*0.0" in text


def test_write_incar_unknown_calculation_raises(tmp_path: Path) -> None:
    s = VASPSettings(calculation="banana", encut_ev=400.0)
    with pytest.raises(ValueError, match=r"Unknown calculation"):
        write_incar(s, tmp_path / "INCAR")


def test_write_incar_missing_encut_raises(tmp_path: Path) -> None:
    s = VASPSettings(calculation="scf")  # encut_ev left None
    with pytest.raises(ValueError, match=r"encut_ev"):
        write_incar(s, tmp_path / "INCAR")


# --------------------------------------------------------------------------- #
# prepare_relax_workdir                                                       #
# --------------------------------------------------------------------------- #


def test_prepare_relax_workdir_writes_all_files(tmp_path: Path) -> None:
    pot = tmp_path / "potcars"
    _write_dummy_potcar(pot, "Si", enmax=245.0)
    s = recommend_settings(bulk("Si"), potcar_dir=pot)
    wd = prepare_relax_workdir(bulk("Si"), tmp_path / "wd", s, potcar_dir=pot)
    wd_p = Path(wd)
    assert (wd_p / "POSCAR").is_file()
    assert (wd_p / "INCAR").is_file()
    assert (wd_p / "POTCAR").is_file()
    # KSPACING is set in INCAR, so no KPOINTS file should be written.
    assert not (wd_p / "KPOINTS").exists()


def test_prepare_relax_workdir_writes_kpoints_when_no_kspacing(tmp_path: Path) -> None:
    pot = tmp_path / "potcars"
    _write_dummy_potcar(pot, "Si", enmax=245.0)
    s = recommend_settings(bulk("Si"), potcar_dir=pot)
    s.kspacing = None  # force KPOINTS-file fallback
    wd = prepare_relax_workdir(bulk("Si"), tmp_path / "wd", s, potcar_dir=pot)
    assert (Path(wd) / "KPOINTS").is_file()


# --------------------------------------------------------------------------- #
# parse_relax_outputs                                                         #
# --------------------------------------------------------------------------- #


def test_parse_relax_outputs_handles_missing_outputs(tmp_path: Path) -> None:
    """Empty workdir → result with converged=False, no crash."""
    res = parse_relax_outputs(tmp_path, return_code=1)
    assert res.converged is False
    assert res.return_code == 1
    assert res.final_energy_eV is None
    assert res.energies_eV == []


def test_parse_relax_outputs_outcar_only(tmp_path: Path) -> None:
    """If only OUTCAR is present, parser still extracts walltime + convergence."""
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        "----------- Iteration    1(   1) -----------\n"
        "----------- Iteration    1(   2) -----------\n"
        "----------- Iteration    2(   1) -----------\n"
        "reached required accuracy - stopping structural energy minimisation\n"
        "Elapsed time (sec):    12.345\n"
        "General timing and accounting informations for this job:\n"
    )
    res = parse_relax_outputs(tmp_path, return_code=0)
    assert res.job_done is True
    assert res.scf_iterations_per_step == [2, 1]
    assert res.wall_time_sec == pytest.approx(12.345)
    # No vasprun.xml → no energies, so converged stays False (need final_energy).
    assert res.final_energy_eV is None
    assert res.converged is False
