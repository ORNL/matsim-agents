"""Full-size (1,024-atom) invariants of the QE campaign gates."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows import full_supercell_campaign as fsc
from tests.synthetic_upf import write_synthetic_upf

CONFIG = Path(__file__).parents[1] / "examples/fusion_bcc_1024/campaign.yaml"


@pytest.fixture(scope="module")
def cfg():
    return fsc.FullSupercellCampaign.from_yaml(CONFIG)


@pytest.fixture(scope="module")
def structures(cfg):
    return fsc.build_structures(cfg)


def test_config_fields(cfg):
    p = cfg.parent_supercell
    assert p.conventional_repeats == (8, 8, 8)
    assert p.expected_parent_atoms == 1024
    assert p.composition == {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}
    assert cfg.numerical_qualification.require_full_parent_supercell
    assert cfg.numerical_qualification.prohibit_smaller_cell_substitution


def test_inconsistent_parent_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        fsc.ParentSupercell(
            conventional_repeats=(8, 8, 8),
            expected_parent_atoms=1024,
            composition={"W": 15, "Mo": 1},
            lattice_parameter=3.16,
        )


def test_parent_exact(structures, cfg):
    parent = structures["parent"]
    assert len(parent) == 1024
    assert fsc._composition(parent) == cfg.parent_supercell.composition
    assert np.allclose(parent.cell.lengths(), 8 * 3.16)


def test_smaller_cell_rejected(cfg):
    small = generate_alloy(
        AlloySpec(
            counts={"W": 10, "Ta": 2, "Cr": 1, "V": 1, "Hf": 1, "Mo": 1}, repeats=(2, 2, 2), seed=0
        )
    )
    with pytest.raises(ValueError, match="smaller-cell substitution"):
        fsc.validate_parent(small, cfg)


def test_chemical_order_preserved(structures, cfg):
    fp = structures["parent_fingerprint"]
    assert occupation_fingerprint(fsc.build_parent(cfg)) == fp
    for atoms, config in structures["defects"].values():
        assert config.parent_ordering_id == fp
        ids = atoms.get_array("atom_id")
        parent_syms = np.array(structures["parent"].get_chemical_symbols())
        kept = ids < 1024
        syms = np.array(atoms.get_chemical_symbols())
        assert (syms[kept] == parent_syms[ids[kept]]).all()


@pytest.mark.parametrize(
    "kind,n",
    [("vacancy", 1023), ("interstitial", 1025), ("frenkel_pair", 1024), ("divacancy", 1022)],
)
def test_defect_atom_counts(structures, kind, n):
    atoms, config = structures["defects"][kind]
    assert len(atoms) == n and config.n_parent_atoms == 1024


def test_defect_gate_rejects_wrong_parent(structures, cfg):
    atoms, config = structures["defects"]["vacancy"]
    with pytest.raises(ValueError, match="parent_ordering_id"):
        fsc.validate_defect(atoms, config, cfg, "not-the-parent")


@pytest.mark.parametrize(
    "key,n", [("neb_vacancy", 1023), ("neb_interstitial", 1025), ("neb_recombination", 1024)]
)
def test_neb_endpoints_compatible(structures, key, n):
    pair = structures["nebs"][key]
    fsc.validate_neb_pair(pair, n)
    assert not np.allclose(pair.initial.positions, pair.final.positions)


def test_magnetic_atom_id_mapping(structures):
    from matsim_agents.science.magnetism.policies import configurations_for_policy, select_policy

    parent = structures["parent"]
    policy = select_policy(parent)
    mags = configurations_for_policy(parent, "p", policy)
    assert {m.mode for m in mags} >= {"nonmagnetic"}
    for m in mags:
        fsc.validate_magnetic(parent, m)
    fm = next(m for m in mags if m.spin_polarized)
    cr = {i for i, s in zip(fm.atom_ids, fm.species, strict=True) if s == "Cr"}
    moments = dict(zip(fm.atom_ids, fm.initial_moments_muB, strict=True))
    assert len(cr) == 64 and all(moments[i] > 0 for i in cr)


def test_manifest_blocked_without_pseudos(cfg, tmp_path):
    blocked = cfg.model_copy(update={"qe": cfg.qe.model_copy(update={"pseudo_dir": str(tmp_path)})})
    entries = fsc.prepare_campaign(blocked, tmp_path / "run", git_revision="abc")
    assert all(e["status"].startswith("INPUT_BLOCKED") for e in entries)
    assert all(e["job_id"] is None and e["input_sha256"] is None for e in entries)
    assert json.loads((tmp_path / "run" / fsc.MANIFEST_FILE).read_text()) == entries


def test_full_size_inputs_and_manifest(cfg, tmp_path):
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    for el in cfg.parent_supercell.composition:
        write_synthetic_upf(pseudo, f"{el}.pbe.UPF", el)
    qe = cfg.qe.model_copy(
        update={
            "pseudo_dir": str(pseudo),
            "pseudopotentials": {el: f"{el}.pbe.UPF" for el in cfg.parent_supercell.composition},
        }
    )
    ready = cfg.model_copy(update={"qe": qe})
    entries = fsc.prepare_campaign(ready, tmp_path / "run", git_revision="abc")
    by_id = {e["calc_id"]: e for e in entries}
    assert by_id["pristine_nonmagnetic"]["qe_atoms"] == 1024
    assert by_id["vacancy"]["qe_atoms"] == 1023
    assert by_id["interstitial"]["qe_atoms"] == 1025
    assert by_id["frenkel_pair"]["qe_atoms"] == 1024
    assert by_id["pristine_ferromagnetic"]["status"].startswith("INPUT_BLOCKED")
    for e in entries:
        assert e["parent_atoms"] == 1024
        if e["calc_id"] == "pristine_ferromagnetic":
            continue
        assert e["status"] == "PREPARED"
        if e["kind"] == "scf":
            text = (Path(e["output_dir"]) / "pw.in").read_text()
            assert re.search(rf"\bnat\s*=\s*{e['qe_atoms']}\b", text)
            assert len(e["input_sha256"]) == 64
