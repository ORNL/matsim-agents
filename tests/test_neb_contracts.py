"""NEB contracts (synthetic structures; no calculations)."""

import numpy as np
import pytest
from pydantic import ValidationError

from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.neb.contracts import (
    MigrationMechanism,
    NebCalculationSpec,
    NebImage,
    NebImageSequence,
    NebResult,
    NebTransition,
    build_correspondence,
    check_endpoint_compatibility,
    endpoint_from_atoms,
)
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

SPEC = AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1)
PARENT = generate_alloy(SPEC)


def _hop_pair():
    """Vacancy at site 0; the atom nearest to it moves into the vacancy (initial/final)."""
    initial, config = generate_vacancy(PARENT, "p", 0)
    final = initial.copy()
    dist = np.linalg.norm(PARENT.positions[1:] - PARENT.positions[0], axis=1)
    mover = int(np.argmin(dist))  # index in `initial` (ids shift by one)
    final.positions[mover] = PARENT.positions[0]
    return initial, final, config, int(initial.get_array("atom_id")[mover])


def _endpoints(**kw):
    initial, final, config, mover = _hop_pair()
    common = {"parent_id": "p", "defect_type": "vacancy", "defect_config": config}
    return (
        endpoint_from_atoms(initial, "i", "initial", **common),
        endpoint_from_atoms(final, "f", "final", **common, **kw),
        mover,
    )


def test_compatible_endpoints_give_an_identity_correspondence():
    initial, final, mover = _endpoints()
    check_endpoint_compatibility(initial, final)
    corr = build_correspondence(initial, final)
    assert len(corr.atom_ids) == len(PARENT) - 1
    assert set(initial.atom_ids) == set(final.atom_ids)
    assert mover in corr.atom_ids and initial.status == "generated"


def test_reordered_final_keeps_identity_not_index_correspondence():
    initial_atoms, final_atoms, config, _ = _hop_pair()
    reordered = final_atoms[list(reversed(range(len(final_atoms))))]
    common = {"parent_id": "p", "defect_type": "vacancy", "defect_config": config}
    a = endpoint_from_atoms(initial_atoms, "i", "initial", **common)
    b = endpoint_from_atoms(reordered, "f", "final", **common)
    corr = build_correspondence(a, b)
    for atom_id, k in zip(corr.atom_ids, corr.final_indices, strict=True):
        assert b.atom_ids[k] == atom_id
    assert corr.final_indices != corr.initial_indices


@pytest.mark.parametrize("change", ["species", "ids", "count", "cell", "parent", "type"])
def test_incompatible_endpoints_are_rejected(change):
    initial, final, _ = _endpoints()
    if change == "species":
        symbols = list(final.species)
        symbols[0] = "Hf" if symbols[0] != "Hf" else "W"
        final = final.model_copy(update={"species": tuple(symbols)})
        match = "species of atom"
    elif change == "ids":
        ids = list(final.atom_ids)
        ids[0] = 999
        final = final.model_copy(update={"atom_ids": tuple(ids)})
        match = "same atomic identities"
    elif change == "count":
        final = final.model_copy(
            update={"atom_ids": final.atom_ids[:-1], "species": final.species[:-1]}
        )
        match = "atom counts differ"
    elif change == "cell":
        cell = tuple(tuple(x * 1.01 for x in row) for row in final.cell)
        final = final.model_copy(update={"cell": cell})
        match = "cells differ"
    elif change == "parent":
        final = final.model_copy(update={"parent_id": "other"})
        match = "different parent"
    else:
        final = final.model_copy(update={"defect_type": "interstitial"})
        match = "different defect types"
    with pytest.raises(ValueError, match=match):
        check_endpoint_compatibility(initial, final)


def test_swapped_roles_and_open_boundaries_are_rejected():
    initial, final, _ = _endpoints()
    with pytest.raises(ValueError, match="initial endpoint and a final"):
        check_endpoint_compatibility(final, initial)
    open_final = final.model_copy(update={"pbc": (True, True, False)})
    open_initial = initial.model_copy(update={"pbc": (True, True, False)})
    with pytest.raises(ValueError, match="periodic"):
        check_endpoint_compatibility(open_initial, open_final)


def test_endpoint_needs_identities_and_finite_coordinates():
    atoms = PARENT.copy()
    with pytest.raises(ValueError, match="atom_id"):
        endpoint_from_atoms(atoms, "x", "initial", parent_id="p", defect_type="vacancy")
    atoms.set_array("atom_id", np.arange(len(atoms)))
    atoms.positions[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        endpoint_from_atoms(atoms, "x", "initial", parent_id="p", defect_type="vacancy")


def test_endpoint_states_are_distinct_and_not_promoted_implicitly():
    initial, final, _ = _endpoints()
    with pytest.raises(ValidationError, match="method identity"):
        type(final).model_validate({**final.model_dump(), "status": "relaxed"})
    relaxed = {
        **final.model_dump(),
        "status": "relaxed",
        "method": "m",
        "relaxation_converged": True,
    }
    assert type(final).model_validate(relaxed).status == "relaxed"
    with pytest.raises(ValidationError, match="converged relaxation"):
        type(final).model_validate({**relaxed, "relaxation_converged": False})
    with pytest.raises(ValidationError, match="evidence"):
        type(final).model_validate({**relaxed, "status": "numerically_qualified"})
    qualified = {**relaxed, "status": "numerically_qualified", "qualification_evidence": ("e1",)}
    assert type(final).model_validate(qualified).status == "numerically_qualified"
    with pytest.raises(ValidationError, match="only valid for a qualified"):
        type(final).model_validate({**relaxed, "qualification_evidence": ("e1",)})


def test_image_sequence_and_calculation_spec_validate_counts():
    with pytest.raises(ValidationError, match="at least 4"):
        NebCalculationSpec(n_images=3, engine="qe", method_id="m")
    images = [
        NebImage(index=k, structure_ref=f"i{k}.extxyz", geometry_fingerprint="a") for k in range(4)
    ]
    assert len(NebImageSequence(images=images, method="linear").images) == 4
    with pytest.raises(ValidationError, match="at least 4"):
        NebImageSequence(images=images[:3], method="linear")
    with pytest.raises(ValidationError, match="0..n-1"):
        NebImageSequence(images=[images[1], images[0], images[2], images[3]], method="linear")


def test_result_states_are_not_equivalent():
    base = {"method_id": "m", "n_images": 5}
    with pytest.raises(ValidationError, match="cannot carry"):
        NebResult(state="failed", forward_barrier_eV=0.3, **base)
    unconverged = NebResult(
        state="unconverged", relative_energies_eV=(0.0, 0.2, 0.4, 0.2, 0.1), **base
    )
    assert unconverged.state == "unconverged"
    with pytest.raises(ValidationError, match="qualified endpoints and both barriers"):
        NebResult(
            state="qualified",
            relative_energies_eV=(0.0, 0.2, 0.4, 0.2, 0.1),
            forward_barrier_eV=0.4,
            reverse_barrier_eV=0.3,
            endpoints_qualified=False,
            **base,
        )
    ok = NebResult(
        state="qualified",
        relative_energies_eV=(0.0, 0.2, 0.4, 0.2, 0.1),
        forward_barrier_eV=0.4,
        reverse_barrier_eV=0.3,
        endpoints_qualified=True,
        **base,
    )
    assert ok.state == "qualified"
    with pytest.raises(ValidationError):
        NebResult(state="converged_unqualified", forward_barrier_eV=float("nan"), **base)


def test_transition_serialization_round_trip_and_consistency():
    initial, final, mover = _endpoints()
    transition = NebTransition(
        transition_id="t1",
        initial=initial,
        final=final,
        correspondence=build_correspondence(initial, final),
        mechanism=MigrationMechanism(
            kind="vacancy_hop", moving_atom_ids=(mover,), parameters={"shell": 1}
        ),
        calculation=NebCalculationSpec(n_images=7, engine="quantum_espresso", method_id="m"),
        generation_parameters={"n_images": 7},
    )
    assert NebTransition.model_validate_json(transition.model_dump_json()) == transition
    with pytest.raises(ValidationError, match="does not match the endpoints"):
        NebTransition(
            transition_id="t1",
            initial=initial,
            final=final,
            correspondence=build_correspondence(
                initial, initial.model_copy(update={"role": "final"})
            ).model_copy(update={"final_indices": tuple(reversed(range(len(initial.atom_ids))))}),
            mechanism=MigrationMechanism(kind="vacancy_hop", moving_atom_ids=(mover,)),
        )
    with pytest.raises(ValidationError, match="not in the endpoints"):
        NebTransition(
            transition_id="t1",
            initial=initial,
            final=final,
            correspondence=build_correspondence(initial, final),
            mechanism=MigrationMechanism(kind="vacancy_hop", moving_atom_ids=(999,)),
        )
