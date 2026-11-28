"""pw.in generation with collinear magnetic configurations (M4)."""

from __future__ import annotations

import re
import tempfile as _tempfile

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.backends.dft.qe import _write_pw_from_template
from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.science.magnetism import generators as gen
from tests.synthetic_upf import write_synthetic_upf

PP_DIR = _tempfile.mkdtemp(prefix="synthetic-upf-")
for _el in ("Fe", "Cr"):
    write_synthetic_upf(PP_DIR, f"{_el}.pbe.UPF", _el)

PP_DIR = _tempfile.mkdtemp(prefix="synthetic-upf-")
for _el in ("Fe", "Cr"):
    write_synthetic_upf(PP_DIR, f"{_el}.pbe.UPF", _el)

A = 2.87
PSEUDOS = {"Fe": "Fe.pbe.UPF", "Cr": "Cr.pbe.UPF"}
TEMPLATE = """&CONTROL
  calculation = 'scf'
  pseudo_dir = '{pseudo_dir}'
  prefix = '{prefix}'
  outdir = '{outdir}'
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1e-8
/
"""


def _atoms():
    atoms = bulk("Fe", "bcc", a=A, cubic=True) * (2, 2, 2)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _polar(cfg):
    return cfg.model_copy(
        update={"qe_starting_magnetization": tuple(m / 8.0 for m in cfg.initial_moments_muB)}
    )


def _settings(**extra):
    return QESettings(
        pseudo_dir=PP_DIR,
        ecutwfc_ry=40.0,
        ecutrho_ry=320.0,
        occupations="smearing",
        kpts=(2, 2, 2),
        pseudopotentials=dict(PSEUDOS),
        **extra,
    )


def _write(tmp_path, atoms, cfg, settings=None):
    path = tmp_path / "pw.in"
    write_pw_input(atoms, settings or _settings(), str(path), magnetic_config=cfg)
    return path.read_text()


def test_nonmagnetic_default_is_unchanged(tmp_path):
    atoms = _atoms()
    text = _write(tmp_path, atoms, None)
    assert "nspin" not in text and "starting_magnetization" not in text
    assert re.search(r"ntyp\s*=\s*1\b", text)
    assert "  Fe  " in text


def test_nonmagnetic_configuration_matches_default_deck(tmp_path):
    atoms = _atoms()
    assert _write(tmp_path, atoms, gen.nonmagnetic(atoms, "s")) == _write(tmp_path, atoms, None)


def test_ferromagnetic_deck(tmp_path):
    atoms = _atoms()
    text = _write(tmp_path, atoms, _polar(gen.ferromagnetic(atoms, "s", {"Fe": 2.2})))
    assert re.search(r"nspin\s*=\s*2", text)
    assert "starting_magnetization(1)" in text
    assert re.search(r"ntyp\s*=\s*1\b", text)


def test_afm_deck_has_two_types_and_labelled_positions(tmp_path):
    atoms = _atoms()
    text = _write(tmp_path, atoms, _polar(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})))
    assert re.search(r"ntyp\s*=\s*2\b", text)
    assert "starting_magnetization(1)" in text and "starting_magnetization(2)" in text
    species = text.split("ATOMIC_SPECIES")[1].split("CELL_PARAMETERS")[0]
    assert species.count("Fe.pbe.UPF") == 2
    positions = text.split("ATOMIC_POSITIONS angstrom")[1].split("K_POINTS")[0].split("\n")
    labels = [p.split()[0] for p in positions if p.strip()]
    assert len(labels) == len(atoms) and set(labels) == {"Fe1", "Fe2"}


def test_contradictory_extra_system_is_rejected(tmp_path):
    atoms = _atoms()
    cfg = _polar(gen.ferromagnetic(atoms, "s", {"Fe": 2.2}))
    for key in ("nspin", "starting_magnetization(1)", "tot_magnetization"):
        with pytest.raises(ValueError, match="controls"):
            _write(tmp_path, atoms, cfg, _settings(extra_system={key: 1}))


def test_extra_system_without_magnetism_still_works(tmp_path):
    atoms = _atoms()
    text = _write(tmp_path, atoms, None, _settings(extra_system={"nspin": 2}))
    assert re.search(r"nspin\s*=\s*2", text)


def _template(tmp_path, body=TEMPLATE):
    path = tmp_path / "t.in"
    path.write_text(body)
    return path


def _render(tmp_path, atoms, cfg, body=TEMPLATE):
    out = tmp_path / "pw.in"
    _write_pw_from_template(
        template_path=_template(tmp_path, body),
        atoms=atoms,
        pseudo_dir=PP_DIR,
        kpts=(2, 2, 2),
        koffset=(0, 0, 0),
        pseudopotentials=dict(PSEUDOS),
        input_path=str(out),
        prefix="pwscf",
        outdir="./tmp",
        magnetic_config=cfg,
    )
    return out.read_text()


def test_template_with_afm_gets_types_and_magnetization(tmp_path):
    atoms = _atoms()
    text = _render(tmp_path, atoms, _polar(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})))
    system = text.split("&SYSTEM")[1].split("&ELECTRONS")[0]
    assert re.search(r"ntyp\s*=\s*2\b", system)
    assert "nspin = 2" in system and "starting_magnetization(2)" in system
    assert system.rstrip().endswith("/")
    assert "Fe1" in text and "Fe2" in text


def test_template_without_magnetism_is_unchanged(tmp_path):
    atoms = _atoms()
    text = _render(tmp_path, atoms, None)
    assert "nspin" not in text and re.search(r"ntyp\s*=\s*1\b", text)


def test_template_with_magnetic_keys_is_rejected(tmp_path):
    atoms = _atoms()
    cfg = _polar(gen.ferromagnetic(atoms, "s", {"Fe": 2.2}))
    body = TEMPLATE.replace("  ecutwfc = 40", "  ecutwfc = 40\n  nspin = 2")
    with pytest.raises(ValueError, match="controls"):
        _render(tmp_path, atoms, cfg, body)


def test_template_validates_effective_pseudopotential_settings(tmp_path):
    atoms = _atoms()
    with pytest.raises(ValueError, match="below the pseudopotential recommendation"):
        _render(tmp_path, atoms, None, TEMPLATE.replace("ecutwfc = 40", "ecutwfc = 1"))
    with pytest.raises(ValueError, match="functional"):
        _render(
            tmp_path,
            atoms,
            None,
            TEMPLATE.replace("ecutwfc = 40", "input_dft = 'PBEsol'\n  ecutwfc = 40"),
        )
    with pytest.raises(ValueError, match="pseudo_dir"):
        _render(
            tmp_path,
            atoms,
            None,
            TEMPLATE.replace("pseudo_dir = '{pseudo_dir}'", "pseudo_dir = '/wrong'"),
        )


def test_template_with_inline_magnetic_key_is_rejected(tmp_path):
    atoms = _atoms()
    cfg = _polar(gen.ferromagnetic(atoms, "s", {"Fe": 2.2}))
    body = TEMPLATE.replace("  ecutwfc = 40", "  ecutwfc = 40, nspin = 2")
    with pytest.raises(ValueError, match="controls"):
        _render(tmp_path, atoms, cfg, body)


def test_magnetic_key_in_comment_or_string_is_ignored(tmp_path):
    atoms = _atoms()
    cfg = _polar(gen.ferromagnetic(atoms, "s", {"Fe": 2.2}))
    body = TEMPLATE.replace("  ecutwfc = 40", "  ecutwfc = 40 ! nspin = 1 set below")
    text = _render(tmp_path, atoms, cfg, body)
    assert "nspin = 2" in text


def test_template_with_inline_ntyp_is_updated(tmp_path):
    atoms = _atoms()
    body = TEMPLATE.replace("  nat = {nat}\n  ntyp = {ntyp}\n", "  nat = {nat}, ntyp = 1\n")
    text = _render(tmp_path, atoms, _polar(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})), body)
    system = text.split("&SYSTEM")[1].split("&ELECTRONS")[0]
    assert re.search(r"ntyp\s*=\s*2\b", system) and not re.search(r"ntyp\s*=\s*1\b", system)


def test_compact_system_terminator_keeps_entries_in_system(tmp_path):
    atoms = _atoms()
    body = TEMPLATE.replace("  ecutwfc = 40", "  ecutwfc = 40 ! Ry, not eV/atom").replace(
        "  degauss = 0.01\n/\n&ELECTRONS", "  degauss = 0.01 /\n&ELECTRONS"
    )
    text = _render(tmp_path, atoms, _polar(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})), body)
    system, electrons = text.split("&SYSTEM")[1].split("&ELECTRONS")
    assert "nspin = 2" in system and "starting_magnetization(2)" in system
    assert "nspin" not in electrons and "starting_magnetization" not in electrons
    assert system.rstrip().endswith("/")
