"""Magnetic configuration contract: modes, validation, identity and serialization."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery.magnetism import (
    MagneticConfiguration,
    qe_polarization_from_moment,
    reject_unsupported,
)


def _atoms():
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _config(mode_, moments, labels=None, **extra):
    n = len(moments)
    labels = labels or [f"g{i}" for i in range(n)]
    base = {
        "structure_id": "s1",
        "config_id": "c1",
        "mode": mode_,
        "atom_ids": tuple(range(n)),
        "species": ("Fe",) * n,
        "group_labels": tuple(labels),
        "spin_polarized": mode_ != "nonmagnetic",
        "initial_moments_muB": tuple(moments),
        "generation_method": "test",
    }
    base.update(extra)
    return MagneticConfiguration(**base)


def test_nonmagnetic_requires_zero_moments_and_is_not_spin_polarized():
    cfg = _config("nonmagnetic", [0.0, 0.0], ["A", "A"])
    assert cfg.spin_polarized is False
    with pytest.raises(ValueError, match="nonzero"):
        _config("nonmagnetic", [1.0, 0.0], ["A", "B"])


def test_spin_polarized_zero_initialization_differs_from_nonmagnetic():
    nm = _config("nonmagnetic", [0.0, 0.0], ["A", "A"])
    zero_sp = _config("custom_collinear", [0.0, 0.0], ["A", "A"], spin_polarized=True)
    assert zero_sp.spin_polarized and not nm.spin_polarized
    assert zero_sp.fingerprint() != nm.fingerprint()


def test_ferromagnetic_needs_single_sign():
    assert _config("ferromagnetic", [2.2, 2.2], ["A", "A"]).mode == "ferromagnetic"
    with pytest.raises(ValueError, match="one sign"):
        _config("ferromagnetic", [2.2, -2.2])
    with pytest.raises(ValueError, match="nonzero"):
        _config("ferromagnetic", [0.0, 0.0], ["A", "A"])


def test_antiferromagnet_allows_nonzero_net_moment_but_needs_antiparallel():
    assert _config("antiferromagnetic", [2.0, -1.5]).mode == "antiferromagnetic"
    with pytest.raises(ValueError, match="antiparallel"):
        _config("antiferromagnetic", [2.0, 2.0], ["A", "A"])


def test_ferrimagnetic_needs_unequal_magnitudes():
    assert _config("ferrimagnetic", [3.0, -1.0]).mode == "ferrimagnetic"
    with pytest.raises(ValueError, match="unequal"):
        _config("ferrimagnetic", [2.0, -2.0])


def test_custom_collinear_accepts_arbitrary_pattern():
    cfg = _config("custom_collinear", [2.0, 0.0, -0.5], spin_polarized=True)
    assert cfg.n_atoms == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_moments_muB": (float("nan"), 1.0)}, ""),
        ({"initial_moments_muB": (1.0,)}, "one value per atom"),
        ({"atom_ids": (0, 0)}, "unique"),
        ({"group_labels": ("1bad", "g1")}, "identifiers"),
        ({"qe_starting_magnetization": (1.5, 0.0)}, r"\[-1, 1\]"),
        ({"mode": "noncollinear"}, ""),
        ({"magnetism_model": "noncollinear"}, ""),
        ({"moment_unit": "Ry"}, ""),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config("ferromagnetic", [2.0, 2.0], ["A", "A"], **kwargs)


def test_group_must_not_mix_assignments():
    with pytest.raises(ValueError, match="mixes"):
        _config("ferromagnetic", [2.0, 3.0], ["A", "A"])


def test_unsupported_magnetism_is_rejected_explicitly():
    for kind in ("noncollinear", "spin_orbit", "spin_spiral", "dlm"):
        with pytest.raises(ValueError, match="not supported"):
            reject_unsupported(kind)
    reject_unsupported("collinear")


def test_json_round_trip_preserves_everything():
    cfg = _config(
        "antiferromagnetic",
        [2.0, -2.0],
        seed=7,
        assumptions=("two sublattices",),
        provenance={"tool": "unit-test"},
    )
    again = MagneticConfiguration.from_json(cfg.to_json())
    assert again == cfg
    assert again.fingerprint() == cfg.fingerprint()


def test_assignments_follow_atom_id_when_atoms_are_reordered():
    atoms = _atoms()
    cfg = _config("antiferromagnetic", [2.0, -2.0])
    assert cfg.moments_for_atoms(atoms).tolist() == [2.0, -2.0]
    reordered = atoms[[1, 0]]
    assert reordered.get_array("atom_id").tolist() == [1, 0]
    assert cfg.moments_for_atoms(reordered).tolist() == [-2.0, 2.0]


def test_fingerprint_is_independent_of_storage_order():
    a = _config("antiferromagnetic", [2.0, -2.0])
    b = MagneticConfiguration(
        structure_id="s1",
        config_id="c-other",
        mode="antiferromagnetic",
        atom_ids=(1, 0),
        species=("Fe", "Fe"),
        group_labels=("g1", "g0"),
        spin_polarized=True,
        initial_moments_muB=(-2.0, 2.0),
        generation_method="other",
    )
    assert a.fingerprint() == b.fingerprint()


def test_mismatched_structure_is_rejected():
    atoms = _atoms()
    atoms.set_array("atom_id", np.array([5, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="do not match"):
        _config("ferromagnetic", [2.0, 2.0], ["A", "A"]).validate_against(atoms)
    plain = bulk("Fe", "bcc", a=2.87, cubic=True)
    with pytest.raises(ValueError, match="atom_id"):
        _config("ferromagnetic", [2.0, 2.0], ["A", "A"]).validate_against(plain)


def test_converged_moments_do_not_overwrite_requested_mode():
    cfg = _config("antiferromagnetic", [2.0, -2.0])
    result = cfg.with_converged((0.0, 0.0), classification="nonmagnetic")
    assert result.mode == "antiferromagnetic"
    assert result.converged_moments_muB == (0.0, 0.0)
    assert result.initial_moments_muB == (2.0, -2.0)


def test_polarization_conversion_requires_verified_valence():
    convert = qe_polarization_from_moment
    with pytest.raises(ValueError, match="verified"):
        convert(2.0, valence_electrons=8, verified_valence=False, qe_version="7.3")
    with pytest.raises(ValueError, match="verified"):
        convert(2.0, valence_electrons=8, verified_valence=True, qe_version="")
    assert qe_polarization_from_moment(
        2.0, valence_electrons=8, verified_valence=True, qe_version="7.3"
    ) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="exceeds"):
        convert(9.0, valence_electrons=8, verified_valence=True, qe_version="7.3")
