"""Neb.x input writer (structure-aware checks plus one golden fixture)."""

import hashlib
import re

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_input import render_neb_input, write_neb_input

SPECIES = ("Mo", "W")


def _images(n=5):
    cell = np.eye(3) * 6.0
    out = []
    for k in range(n):
        atoms = Atoms(
            "MoW2",
            positions=[[0.0, 0.0, 0.0], [1.5 + 0.1 * k, 1.5, 1.5], [3.0, 3.0, 3.0]],
            cell=cell,
            pbc=True,
        )
        atoms.set_array("atom_id", np.array([0, 1, 2]))
        out.append(atoms)
    return out


def _cfg(n=5, **over):
    base = {
        "n_images": n,
        "species": SPECIES,
        "pseudo_dir": "/pseudos",
        "pseudopotentials": {"Mo": "Mo.upf", "W": "W.upf"},
        "ecutwfc_ry": 50.0,
        "ecutrho_ry": 400.0,
        "occupations": "smearing",
        "degauss_ry": 0.02,
        "kpts": (2, 2, 2),
        "nspin": 1,
    }
    base.update(over)
    return QENebConfig(**base)


def _sections(text):
    """Split a rendered input into its supercard sequence and the position blocks."""
    lines = [ln.strip() for ln in text.splitlines()]
    markers = [
        ln
        for ln in lines
        if ln
        in {
            "BEGIN",
            "BEGIN_PATH_INPUT",
            "END_PATH_INPUT",
            "BEGIN_ENGINE_INPUT",
            "BEGIN_POSITIONS",
            "FIRST_IMAGE",
            "INTERMEDIATE_IMAGE",
            "LAST_IMAGE",
            "END_POSITIONS",
            "END_ENGINE_INPUT",
            "END",
        }
    ]
    blocks, current = [], None
    for ln in lines:
        if ln in {"FIRST_IMAGE", "INTERMEDIATE_IMAGE", "LAST_IMAGE"}:
            current = []
            blocks.append(current)
        elif ln == "END_POSITIONS":
            current = None
        elif current is not None and ln and not ln.startswith("ATOMIC_POSITIONS"):
            current.append(ln.split())
    return markers, blocks


def _namelist_value(text, key):
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+)$", text, re.MULTILINE)
    assert match, f"{key} not found"
    return match.group(1).strip()


def test_supercard_order_and_image_labels():
    markers, blocks = _sections(render_neb_input(_cfg(), _images()))
    assert markers == [
        "BEGIN",
        "BEGIN_PATH_INPUT",
        "END_PATH_INPUT",
        "BEGIN_ENGINE_INPUT",
        "BEGIN_POSITIONS",
        "FIRST_IMAGE",
        "INTERMEDIATE_IMAGE",
        "INTERMEDIATE_IMAGE",
        "INTERMEDIATE_IMAGE",
        "LAST_IMAGE",
        "END_POSITIONS",
        "END_ENGINE_INPUT",
        "END",
    ]
    assert len(blocks) == 5


def test_positions_reproduce_every_image_in_order():
    images = _images(6)
    _, blocks = _sections(render_neb_input(_cfg(6), images))
    for image, block in zip(images, blocks, strict=True):
        assert [row[0] for row in block] == image.get_chemical_symbols()
        got = np.array([[float(x) for x in row[1:]] for row in block])
        assert np.allclose(got, image.positions, atol=1e-9)


def test_path_namelist_defaults_to_no_ci_and_converted_units():
    text = render_neb_input(_cfg(), _images())
    assert _namelist_value(text, "CI_scheme") == "'no-CI'"
    assert _namelist_value(text, "restart_mode") == "'from_scratch'"
    assert _namelist_value(text, "num_of_images") == "5"
    assert _namelist_value(text, "opt_scheme") == "'broyden'"
    assert float(_namelist_value(text, "path_thr").replace("d", "e")) == pytest.approx(0.05)
    assert _namelist_value(text, "minimum_image") == ".true."
    assert "CLIMBING_IMAGES" not in text


def test_no_climbing_image_from_the_first_iteration():
    for scheme, extra in (("auto", {}), ("manual", {"climbing_images": (3,)})):
        with pytest.raises(ValueError, match="from_scratch"):
            render_neb_input(_cfg(ci_scheme=scheme, **extra), _images())


def test_restart_stage_supports_auto_and_manual_ci():
    auto = render_neb_input(_cfg(restart_mode="restart", ci_scheme="auto"), _images())
    assert _namelist_value(auto, "CI_scheme") == "'auto'" and "CLIMBING_IMAGES" not in auto
    manual = render_neb_input(
        _cfg(restart_mode="restart", ci_scheme="manual", climbing_images=(2, 4)), _images()
    )
    assert _namelist_value(manual, "CI_scheme") == "'manual'"
    lines = [ln.strip() for ln in manual.splitlines()]
    index = lines.index("CLIMBING_IMAGES")
    assert lines[index + 1] == "2, 4"
    assert lines.index("END_PATH_INPUT") == index + 2


def test_engine_namelists_and_cards():
    text = render_neb_input(_cfg(), _images())
    assert _namelist_value(text, "calculation") == "'relax'"
    assert _namelist_value(text, "ntyp") == "2" and _namelist_value(text, "nat") == "3"
    assert _namelist_value(text, "nspin") == "1"
    assert "starting_magnetization" not in text
    assert re.search(r"^\s*Mo\s+95\.9500\s+Mo\.upf$", text, re.MULTILINE)
    assert re.search(r"^\s*W\s+183\.8400\s+W\.upf$", text, re.MULTILINE)
    assert re.search(r"^\s*2 2 2\s+0 0 0$", text, re.MULTILINE)
    assert "K_POINTS automatic" in text and "CELL_PARAMETERS angstrom" in text
    order = [text.index(k) for k in ("ATOMIC_SPECIES", "BEGIN_POSITIONS", "END_ENGINE_INPUT")]
    assert order == sorted(order)


def test_spin_polarized_input_names_each_species_by_card_order():
    cfg = _cfg(nspin=2, starting_magnetization={"Mo": 0.1, "W": 0.0})
    text = render_neb_input(cfg, _images())
    assert _namelist_value(text, "starting_magnetization(1)").startswith("1.0")  # Mo first
    assert float(_namelist_value(text, "starting_magnetization(2)").replace("d", "e")) == 0.0


def test_smearing_keys_only_when_smearing():
    fixed = render_neb_input(_cfg(occupations="fixed", degauss_ry=None), _images())
    assert "degauss" not in fixed and "smearing" not in fixed
    smeared = render_neb_input(_cfg(), _images())
    assert _namelist_value(smeared, "smearing") == "'marzari-vanderbilt'"


def test_mismatched_images_are_rejected():
    cfg = _cfg()
    with pytest.raises(ValueError, match="images supplied"):
        render_neb_input(cfg, _images(4))
    with pytest.raises(ValueError, match="no images"):
        render_neb_input(cfg, [])
    swapped = _images()
    swapped[2] = swapped[2][[0, 2, 1]]
    swapped[2].set_chemical_symbols(["Mo", "W", "W"])
    swapped[2].set_array("atom_id", np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="identities"):
        render_neb_input(cfg, swapped)
    reordered = _images()
    reordered[1] = Atoms("WMoW", positions=reordered[1].positions, cell=reordered[1].cell, pbc=True)
    reordered[1].set_array("atom_id", np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="species or atom order"):
        render_neb_input(cfg, reordered)
    resized = _images()
    resized[3].set_cell(resized[3].cell.array * 1.001)
    with pytest.raises(ValueError, match="cell"):
        render_neb_input(cfg, resized)
    bad = _images()
    bad[2].positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        render_neb_input(cfg, bad)
    open_cell = _images()
    for image in open_cell:
        image.pbc = [True, True, False]
    with pytest.raises(ValueError, match="periodic"):
        render_neb_input(cfg, open_cell)


def test_species_must_match_config():
    other = [Atoms("MoV2", positions=im.positions, cell=im.cell, pbc=True) for im in _images()]
    for image in other:
        image.set_array("atom_id", np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="differ from configured"):
        render_neb_input(_cfg(), other)


def test_write_is_atomic_hashed_and_refuses_overwrite(tmp_path):
    cfg, images = _cfg(), _images()
    path = tmp_path / "run" / "neb.in"
    record = write_neb_input(cfg, images, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256
    assert record.config_hash == cfg.config_hash() and record.n_images == 5
    first = path.read_text()
    with pytest.raises(FileExistsError):
        write_neb_input(cfg, images, path)
    assert path.read_text() == first
    changed = write_neb_input(_cfg(path_thr_ev_ang=0.02), images, path, overwrite=True)
    assert changed.sha256 != record.sha256 and path.read_text() != first
    assert [p.name for p in path.parent.iterdir()] == ["neb.in"]


def test_failed_render_leaves_existing_file_untouched(tmp_path):
    path = tmp_path / "neb.in"
    write_neb_input(_cfg(), _images(), path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_neb_input(_cfg(), _images(4), path, overwrite=True)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["neb.in"]


def test_golden_minimal_input():
    cfg = _cfg(4, kpts=(1, 1, 1))
    images = _images(4)
    expected = """BEGIN
BEGIN_PATH_INPUT
&PATH
  string_method = 'neb'
  restart_mode  = 'from_scratch'
  nstep_path    = 100
  num_of_images = 4
  opt_scheme    = 'broyden'
  CI_scheme     = 'no-CI'
  path_thr      = 5.00000000d-02
  minimum_image = .true.
/
END_PATH_INPUT
BEGIN_ENGINE_INPUT
&CONTROL
  calculation = 'relax'
  prefix      = 'neb'
  pseudo_dir  = '/pseudos'
  outdir      = './tmp'
  tprnfor     = .true.
  verbosity   = 'high'
/
&SYSTEM
  ibrav       = 0
  nat         = 3
  ntyp        = 2
  ecutwfc     = 5.00000000d+01
  ecutrho     = 4.00000000d+02
  occupations = 'smearing'
  smearing    = 'marzari-vanderbilt'
  degauss     = 2.00000000d-02
  nspin       = 1
/
&ELECTRONS
  conv_thr    = 1.00000000d-08
  mixing_beta = 3.00000000d-01
/
&IONS
  ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
  Mo  95.9500  Mo.upf
  W  183.8400  W.upf
BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS angstrom
  Mo        0.0000000000       0.0000000000       0.0000000000
  W        1.5000000000       1.5000000000       1.5000000000
  W        3.0000000000       3.0000000000       3.0000000000
INTERMEDIATE_IMAGE
ATOMIC_POSITIONS angstrom
  Mo        0.0000000000       0.0000000000       0.0000000000
  W        1.6000000000       1.5000000000       1.5000000000
  W        3.0000000000       3.0000000000       3.0000000000
INTERMEDIATE_IMAGE
ATOMIC_POSITIONS angstrom
  Mo        0.0000000000       0.0000000000       0.0000000000
  W        1.7000000000       1.5000000000       1.5000000000
  W        3.0000000000       3.0000000000       3.0000000000
LAST_IMAGE
ATOMIC_POSITIONS angstrom
  Mo        0.0000000000       0.0000000000       0.0000000000
  W        1.8000000000       1.5000000000       1.5000000000
  W        3.0000000000       3.0000000000       3.0000000000
END_POSITIONS
CELL_PARAMETERS angstrom
        6.0000000000       0.0000000000       0.0000000000
        0.0000000000       6.0000000000       0.0000000000
        0.0000000000       0.0000000000       6.0000000000
K_POINTS automatic
  1 1 1  0 0 0
END_ENGINE_INPUT
END
"""
    assert render_neb_input(cfg, images) == expected
