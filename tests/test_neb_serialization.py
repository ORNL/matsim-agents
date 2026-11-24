"""B1.6: pathway serialization round trips (small cells)."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.neb_endpoints import generate_vacancy_hop, vacancy_neighbors
from matsim_agents.discovery.neb_images import generate_linear_images
from matsim_agents.discovery.neb_serialization import read_pathway, write_pathway
from matsim_agents.discovery.point_defects import generate_vacancy, geometry_fingerprint

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))


def _setup():
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config, A)[0]
    pair = generate_vacancy_hop(initial, config, PARENT, mover, A)
    images = list(generate_linear_images(pair.initial, pair.final, 6).images)
    return pair, images


def _write(tmp_path, name="p", **kwargs):
    pair, images = _setup()
    path = tmp_path / name
    pid = write_pathway(path, pair.transition, pair.initial, pair.final, images, **kwargs)
    return pair, images, path, pid


def test_round_trip_is_exact(tmp_path):
    pair, images, path, pid = _write(tmp_path, parameters={"note": "x", "n": 3})
    loaded = read_pathway(path)
    assert loaded.pathway_id == pid
    assert loaded.transition == pair.transition
    assert loaded.parameters == {"note": "x", "n": 3}
    assert geometry_fingerprint(loaded.initial) == geometry_fingerprint(pair.initial)
    assert geometry_fingerprint(loaded.final) == geometry_fingerprint(pair.final)
    assert [geometry_fingerprint(a) for a in loaded.images] == [
        geometry_fingerprint(a) for a in images
    ]
    for got, want in zip(loaded.images, images, strict=True):
        assert np.array_equal(got.get_array("atom_id"), want.get_array("atom_id"))
        assert got.get_chemical_symbols() == want.get_chemical_symbols()
        assert np.array_equal(got.cell.array, want.cell.array)
        assert tuple(got.pbc) == tuple(want.pbc)
    assert loaded.assessment.state == "endpoint_relaxation_pending"


def test_ids_are_repeatable_and_input_dependent(tmp_path):
    _, _, _, first = _write(tmp_path, "a")
    _, _, _, second = _write(tmp_path, "b")
    assert first == second
    _, _, _, third = _write(tmp_path, "c", parameters={"different": 1})
    assert third != first


def test_metadata_holds_no_coordinates_and_files_are_ase_readable(tmp_path):
    pair, images, path, _ = _write(tmp_path)
    text = (path / "pathway.json").read_text()
    meta = json.loads(text)
    assert set(meta) >= {"schema", "pathway_id", "transition", "assessment", "files"}
    assert "positions" not in text
    coordinates = {repr(float(x)) for x in images[2].positions.reshape(-1)}
    long_ones = {c for c in coordinates if len(c) > 8}
    assert long_ones, "probe needs non-trivial coordinates"
    assert not any(c in text for c in long_ones)
    frames = read(path / "images.extxyz", index=":")
    assert len(frames) == len(images)
    assert frames[0].get_chemical_symbols() == images[0].get_chemical_symbols()
    assert len(meta["files"]["images.extxyz"]["fingerprints"]) == len(images)


def test_transition_provenance_and_mechanism_survive(tmp_path):
    pair, _, path, _ = _write(tmp_path)
    loaded = read_pathway(path)
    assert loaded.transition.mechanism == pair.transition.mechanism
    assert loaded.transition.correspondence == pair.transition.correspondence
    assert loaded.transition.initial.defect_config_id == pair.transition.initial.defect_config_id
    assert loaded.transition.generation_parameters == pair.transition.generation_parameters


def test_rejected_pathway_is_written_with_its_reasons(tmp_path):
    pair, images = _setup()
    path = tmp_path / "bad"
    write_pathway(path, pair.transition, pair.initial, pair.final, images[:3])
    loaded = read_pathway(path)
    assert loaded.assessment.state == "rejected"
    assert any("image_count" in r for r in loaded.assessment.reasons)


def test_refuses_overwrite_and_leaves_no_staging_files(tmp_path):
    pair, images, path, _ = _write(tmp_path)
    before = (path / "pathway.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_pathway(path, pair.transition, pair.initial, pair.final, images)
    assert (path / "pathway.json").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["p"]


def test_failed_write_leaves_nothing_behind(tmp_path):
    pair, images = _setup()
    broken = [a.copy() for a in images]
    broken[1].positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_pathway(tmp_path / "x", pair.transition, pair.initial, pair.final, broken)
    assert list(tmp_path.iterdir()) == []


def test_tampering_is_detected(tmp_path):
    for target, mutate, message in [
        ("images.extxyz", lambda p: p.write_text(p.read_text().replace("Mo", "Hf", 1)), "hash"),
        (
            "pathway.json",
            lambda p: p.write_text(p.read_text().replace("Ta", "Hf", 1)),
            "pathway_id",
        ),
        ("final.extxyz", lambda p: p.unlink(), "missing"),
        ("pathway.json", lambda p: p.write_text("{not json"), "JSON"),
    ]:
        _, _, path, _ = _write(tmp_path, f"t-{target}-{message}")
        mutate(path / target)
        with pytest.raises(ValueError, match=message):
            read_pathway(path)


def test_wrong_schema_and_missing_directory(tmp_path):
    _, _, path, _ = _write(tmp_path)
    meta = json.loads((path / "pathway.json").read_text())
    meta["schema"] = "other/9"
    (path / "pathway.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema"):
        read_pathway(path)
    with pytest.raises(ValueError, match="no pathway.json"):
        read_pathway(tmp_path / "missing")


def test_structure_without_identities_is_refused(tmp_path):
    pair, images = _setup()
    bare = Atoms("W2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=np.eye(3) * 3, pbc=True)
    with pytest.raises(ValueError, match="atom_id"):
        write_pathway(tmp_path / "x", pair.transition, bare, pair.final, images)
