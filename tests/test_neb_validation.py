"""Pathway validation and qualification states (small cells)."""

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.neb_contracts import NebCalculationSpec, NebTransition
from matsim_agents.discovery.neb_endpoints import generate_vacancy_hop, vacancy_neighbors
from matsim_agents.discovery.neb_images import generate_linear_images
from matsim_agents.discovery.neb_validation import assess_pathway
from matsim_agents.discovery.point_defects import generate_vacancy

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))


def _setup(n_images=6):
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config, A)[0]
    pair = generate_vacancy_hop(initial, config, PARENT, mover, A)
    images = generate_linear_images(pair.initial, pair.final, n_images).images
    return pair, list(images)


def _with_status(transition, status, **extra):
    """Rebuild endpoint records with a relaxation state (validated by the contract)."""
    updated = {}
    for name in ("initial", "final"):
        endpoint = getattr(transition, name)
        data = endpoint.model_dump()
        data.update(status=status, method="m1", relaxation_converged=True, **extra)
        updated[name] = type(endpoint).model_validate(data)
    return transition.model_copy(update=updated)


def _names(assessment):
    return {c.name for c in assessment.failed}


def test_no_images_is_only_a_candidate():
    pair, _ = _setup()
    assessment = assess_pathway(pair.transition, pair.initial, pair.final)
    assert assessment.state == "candidate" and not assessment.checks


def test_generated_endpoints_cannot_be_qualified():
    pair, images = _setup()
    assessment = assess_pathway(pair.transition, pair.initial, pair.final, images)
    assert assessment.state == "endpoint_relaxation_pending"
    assert not assessment.failed
    assert any("not relaxed" in r for r in assessment.reasons)


def test_relaxed_endpoints_are_geometry_validated_but_not_qualified():
    pair, images = _setup()
    transition = _with_status(pair.transition, "relaxed")
    assert assess_pathway(transition, pair.initial, pair.final, images).state == (
        "geometry_validated"
    )


def test_qualified_endpoints_with_provenance_are_numerically_qualified():
    pair, images = _setup()
    transition = _with_status(
        pair.transition,
        "numerically_qualified",
        qualification_evidence=("residual forces below threshold",),
        provenance={"code": "qe", "input_hash": "abc"},
    )
    assessment = assess_pathway(transition, pair.initial, pair.final, images)
    assert assessment.state == "numerically_qualified" and not assessment.failed


def test_qualified_endpoints_without_provenance_stay_geometry_validated():
    pair, images = _setup()
    transition = _with_status(
        pair.transition, "numerically_qualified", qualification_evidence=("e",)
    )
    assessment = assess_pathway(transition, pair.initial, pair.final, images)
    assert assessment.state == "geometry_validated"
    assert any("provenance" in r for r in assessment.reasons)


def test_mixed_methods_and_mixed_relaxation_are_rejected():
    pair, images = _setup()
    relaxed = _with_status(pair.transition, "relaxed")
    other = relaxed.final.model_copy(update={"method": "m2"})
    assert "method_consistency" in _names(
        assess_pathway(
            relaxed.model_copy(update={"final": other}), pair.initial, pair.final, images
        )
    )
    spec = NebCalculationSpec(n_images=6, engine="qe", method_id="m-other")
    with_spec = relaxed.model_copy(update={"calculation": spec})
    assert "method_consistency" in _names(
        assess_pathway(with_spec, pair.initial, pair.final, images)
    )
    half = pair.transition.model_copy(update={"final": relaxed.final})
    assessment = assess_pathway(half, pair.initial, pair.final, images)
    assert assessment.state == "rejected"
    assert "endpoint_relaxation_consistent" in _names(assessment)


def test_image_count_and_spec_mismatch_are_rejected():
    pair, images = _setup()
    assert "image_count" in _names(
        assess_pathway(pair.transition, pair.initial, pair.final, images[:3])
    )
    spec = NebCalculationSpec(n_images=9, engine="qe", method_id="m")
    transition = pair.transition.model_copy(update={"calculation": spec})
    result = assess_pathway(transition, pair.initial, pair.final, images)
    assert result.state == "rejected" and "image_count" in _names(result)
    assert assess_pathway(pair.transition, pair.initial, pair.final, []).state == "rejected"


def test_structure_not_matching_the_endpoint_record_is_rejected():
    pair, images = _setup()
    moved = pair.final.copy()
    moved.positions[0, 0] += 0.01
    result = assess_pathway(pair.transition, pair.initial, moved, images)
    assert result.state == "rejected" and "final_matches_record" in _names(result)
    swapped = pair.final.copy()
    swapped.symbols[0] = "Hf"
    assert "final_matches_record" in _names(
        assess_pathway(pair.transition, pair.initial, swapped, images)
    )


def test_corrupted_images_are_rejected_with_a_reason():
    pair, images = _setup()
    for label, mutate, check in [
        ("nan", lambda im: im.positions.__setitem__((2, 0), np.nan), "image_identity_species_cell"),
        ("species", lambda im: im.symbols.__setitem__(1, "Hf"), "image_identity_species_cell"),
        ("cell", lambda im: im.set_cell(im.cell.array * 1.01), "image_identity_species_cell"),
        ("ids", lambda im: im.set_array("atom_id", im.get_array("atom_id")[::-1].copy()), None),
    ]:
        broken = [im.copy() for im in images]
        mutate(broken[2])
        result = assess_pathway(pair.transition, pair.initial, pair.final, broken)
        assert result.state == "rejected", label
        if check:
            assert check in _names(result), label


def test_discontinuity_overlap_and_fixed_endpoints_are_detected():
    pair, images = _setup()
    jump = [im.copy() for im in images]
    shift = np.linalg.norm(images[-1].positions - images[0].positions, axis=1)
    index = int(np.flatnonzero(shift)[0])
    jump[3].positions[index] += np.array([0.0, 0.0, 6.0])
    assert "continuity" in _names(assess_pathway(pair.transition, pair.initial, pair.final, jump))
    crowded = [im.copy() for im in images]
    crowded[2].positions[5] = crowded[2].positions[6] + [0.2, 0.0, 0.0]
    assert "no_overlaps" in _names(
        assess_pathway(pair.transition, pair.initial, pair.final, crowded)
    )
    unfixed = [im.copy() for im in images]
    unfixed[0].positions[0, 0] += 0.01
    assert "endpoints_fixed" in _names(
        assess_pathway(pair.transition, pair.initial, pair.final, unfixed)
    )
    big = assess_pathway(pair.transition, pair.initial, pair.final, images, max_step=0.1)
    assert "continuity" in _names(big)


def test_wrong_correspondence_is_rejected():
    pair, images = _setup()
    shuffled = pair.transition.correspondence.model_copy(
        update={"final_indices": pair.transition.correspondence.final_indices[::-1]}
    )
    fields = {**dict(pair.transition), "correspondence": shuffled}
    transition = NebTransition.model_construct(**fields)
    assert "correspondence" in _names(assess_pathway(transition, pair.initial, pair.final, images))


def test_unrelated_atoms_object_is_rejected_not_raised():
    pair, images = _setup()
    odd = Atoms("W4", positions=np.eye(4, 3), cell=np.eye(3) * 5, pbc=True)
    result = assess_pathway(pair.transition, odd, pair.final, images)
    assert result.state == "rejected" and "endpoint_identities" in _names(result)
    with pytest.raises(ValidationError):
        result.state = "numerically_qualified"
