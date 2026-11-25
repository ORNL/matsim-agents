"""QE NEB configuration contract."""

import pytest
from ase import Atoms
from pydantic import ValidationError

from matsim_agents.backends.dft.qe_neb import QENebConfig

SPECIES = ("Mo", "Ta", "W")


def _cfg(**over):
    base = {
        "n_images": 7,
        "species": SPECIES,
        "pseudo_dir": "/nonexistent/pseudos",
        "pseudopotentials": {s: f"{s}.upf" for s in SPECIES},
        "ecutwfc_ry": 50.0,
        "ecutrho_ry": 400.0,
        "occupations": "smearing",
        "degauss_ry": 0.02,
        "kpts": (3, 3, 3),
        "nspin": 1,
    }
    base.update(over)
    return QENebConfig(**base)


def _bad(match, **over):
    with pytest.raises(ValidationError, match=match):
        _cfg(**over)


def test_valid_config_defaults_are_conservative():
    cfg = _cfg()
    assert cfg.ci_scheme == "no-CI" and cfg.restart_mode == "from_scratch"
    assert cfg.climbing_images == () and cfg.minimum_image is True
    assert cfg.path_thr_ev_ang == 0.05 and cfg.nstep_path == 100


def test_hash_is_stable_and_sensitive():
    a, b = _cfg(), _cfg(pseudopotentials={s: f"{s}.upf" for s in reversed(SPECIES)})
    assert a.config_hash() == b.config_hash() and len(a.config_hash()) == 64
    assert _cfg(n_images=9).config_hash() != a.config_hash()
    assert _cfg(path_thr_ev_ang=0.01).config_hash() != a.config_hash()
    assert QENebConfig.model_validate(a.model_dump(mode="json")) == a


def test_config_is_immutable():
    cfg = _cfg()
    with pytest.raises(ValidationError):
        cfg.n_images = 9
    with pytest.raises(TypeError):
        cfg.pseudopotentials["Mo"] = "other.upf"


@pytest.mark.parametrize("n", [0, 3, -5])
def test_too_few_images_rejected(n):
    _bad("n_images", n_images=n)


def test_non_integer_images_and_steps_rejected():
    _bad("n_images", n_images=7.5)
    _bad("n_images", n_images=True)
    _bad("nstep_path", nstep_path=0)


@pytest.mark.parametrize("thr", [0.0, -0.05, float("nan"), float("inf")])
def test_bad_thresholds_rejected(thr):
    _bad("path_thr_ev_ang", path_thr_ev_ang=thr)


def test_unknown_schemes_rejected():
    _bad("opt_scheme", opt_scheme="newton")
    _bad("ci_scheme", ci_scheme="CI")
    _bad("restart_mode", restart_mode="resume")


def test_manual_climbing_image_rules():
    assert _cfg(ci_scheme="manual", climbing_images=(4,)).climbing_images == (4,)
    _bad("needs climbing_images", ci_scheme="manual")
    _bad("interior", ci_scheme="manual", climbing_images=(1,))
    _bad("interior", ci_scheme="manual", climbing_images=(7,))
    _bad("unique", ci_scheme="manual", climbing_images=(3, 3))
    _bad("only valid with", ci_scheme="auto", climbing_images=(3,))
    _bad("only valid with", climbing_images=(3,))


def test_pseudopotential_completeness():
    _bad("missing pseudopotentials", pseudopotentials={"Mo": "Mo.upf", "Ta": "Ta.upf"})
    full = {s: f"{s}.upf" for s in SPECIES}
    _bad("missing pseudopotentials", pseudopotentials={**full, "W": ""})
    _bad("not in the structure", pseudopotentials={**full, "V": "V.upf"})


def test_inconsistent_pw_settings_rejected():
    _bad("ecutrho_ry", ecutrho_ry=150.0)
    _bad("ecutwfc_ry", ecutwfc_ry=-1.0)
    _bad("degauss_ry", degauss_ry=None)
    _bad("degauss_ry", degauss_ry=0.0)
    _bad("only valid with smearing", occupations="fixed")
    assert _cfg(occupations="fixed", degauss_ry=None).occupations == "fixed"
    _bad("kpts", kpts=(0, 3, 3))
    _bad("koffset", koffset=(0, 2, 0))
    _bad("conv_thr_ry", conv_thr_ry=0.0)
    _bad("mixing_beta", mixing_beta=1.5)
    _bad("smearing", smearing="cold")


def test_duplicate_or_empty_species_rejected():
    _bad("unique", species=("Mo", "Mo", "W"))
    _bad("species", species=())


def test_magnetism_must_be_explicit_and_consistent():
    _bad("requires nspin=2", starting_magnetization={"Mo": 0.1})
    _bad("starting_magnetization for", nspin=2, starting_magnetization={"Mo": 0.1})
    full = {s: 0.0 for s in SPECIES}
    assert _cfg(nspin=2, starting_magnetization=full).nspin == 2
    _bad("must be in", nspin=2, starting_magnetization={**full, "W": 2.0})
    _bad("unknown species", nspin=2, starting_magnetization={**full, "V": 0.1})


def test_chromium_requires_recorded_magnetic_treatment():
    species = ("Cr", "W")
    kw = {"species": species, "pseudopotentials": {s: f"{s}.upf" for s in species}}
    _bad("Cr is present", **kw)
    _bad("Cr is present", magnetic_treatment="  ", **kw)
    cfg = _cfg(magnetic_treatment="nonmagnetic approximation (unvalidated)", **kw)
    assert "nonmagnetic" in cfg.magnetic_treatment
    afm = _cfg(
        nspin=2,
        starting_magnetization={"Cr": 0.5, "W": 0.0},
        magnetic_treatment="collinear ferromagnetic start",
        **kw,
    )
    assert afm.starting_magnetization["Cr"] == 0.5


def test_managed_keys_cannot_be_overridden_through_extras():
    _bad("managed", extra_system={"ecutwfc": 10})
    _bad("managed", extra_system={"NSPIN": 2})
    _bad("managed", extra_electrons={"conv_thr": 1e-3})
    _bad("managed", extra_path={"num_of_images": 3})
    _bad("managed", extra_path={"CI_scheme": "auto"})
    _bad("finite", extra_system={"nbnd": float("nan")})
    cfg = _cfg(extra_system={"nbnd": 120}, extra_path={"ds": 1.0})
    assert cfg.extra_system["nbnd"] == 120


def test_execution_settings_validated():
    _bad("prefix", prefix="bad name")
    _bad("prefix", prefix="../x")
    _bad("nimage", nimage=0)
    _bad("neb_executable", neb_executable="  ")
    _bad("launcher", launcher=("srun", ""))
    _bad("set by the config", extra_args=("-inp", "x.in"))
    _bad("set by the config", extra_args=("-nimage",))
    assert _cfg(launcher=("mpirun", "-n", "4"), nimage=2).launcher[0] == "mpirun"


def test_structure_species_must_match():
    cfg = _cfg()
    good = Atoms("MoTaW", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    cfg.check_structure(good)
    with pytest.raises(ValueError, match="differ"):
        cfg.check_structure(Atoms("MoTa", positions=[[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(ValueError, match="differ"):
        cfg.check_structure(Atoms("MoTaWV", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))


def test_runtime_problems_report_missing_files(tmp_path):
    problems = _cfg().runtime_problems()
    assert any("not a directory" in p for p in problems)
    (tmp_path / "Mo.upf").write_text("x")
    (tmp_path / "W.upf").write_text("x")
    cfg = _cfg(pseudo_dir=str(tmp_path))
    assert cfg.runtime_problems() == ["pseudopotential 'Ta.upf' for Ta not found"]
    (tmp_path / "Ta.upf").write_text("x")
    assert _cfg(pseudo_dir=str(tmp_path)).runtime_problems() == []
    absent = _cfg(pseudo_dir=str(tmp_path), neb_executable=str(tmp_path / "neb.x"))
    assert any("not an executable" in p for p in absent.runtime_problems())
    exe = tmp_path / "neb.x"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert _cfg(pseudo_dir=str(tmp_path), neb_executable=str(exe)).runtime_problems() == []


def test_min_images_matches_the_discovery_contract():
    from matsim_agents.backends.dft import qe_neb
    from matsim_agents.science.neb import contracts as neb_contracts

    assert qe_neb.MIN_IMAGES == neb_contracts.MIN_IMAGES
