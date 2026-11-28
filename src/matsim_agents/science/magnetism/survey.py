"""Nonmagnetic, ferromagnetic and antiferromagnetic starting states for every structure.

The magnetic elements and their starting values come from configuration (``MagneticSurvey``).
The parent gets all three states; a defect structure inherits each state from the parent by
``atom_id`` (see ``defects.derive_defect_configuration``), with inserted atoms starting
unpolarized. The antiferromagnetic pattern is the two-colouring of the parent's
nearest-neighbour graph, so it is a collinear two-sublattice starting state only; DFT decides the
final moments and the lowest converged energy among the evaluated states picks the reference.
"""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from matsim_agents.science.magnetism import generators as gen
from matsim_agents.science.magnetism.config import ZERO_TOLERANCE, MagneticConfiguration
from matsim_agents.science.magnetism.defects import InsertedMoment, derive_defect_configuration

SURVEY_MODES: tuple[str, ...] = ("nonmagnetic", "ferromagnetic", "antiferromagnetic")


class ElementStart(BaseModel):
    """Starting values for one magnetic element."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    moment_muB: float = Field(gt=0, allow_inf_nan=False)
    qe_starting_magnetization: float = Field(gt=0, le=1, allow_inf_nan=False)


class MagneticSurvey(BaseModel):
    """Elements that are tried nonmagnetic, ferromagnetic and antiferromagnetic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    elements: dict[str, ElementStart] = Field(min_length=1)

    @model_validator(mode="after")
    def _non_empty_symbols(self) -> MagneticSurvey:
        if any(not symbol.strip() for symbol in self.elements):
            raise ValueError("magnetic element symbols must be non-empty")
        return self


def _with_polarization(
    config: MagneticConfiguration, survey: MagneticSurvey
) -> MagneticConfiguration:
    """Attach QE starting magnetizations: sign of the moment times the element's value."""
    if not config.spin_polarized:
        return config
    assert config.initial_moments_muB is not None
    polarization = tuple(
        0.0
        if abs(moment) <= ZERO_TOLERANCE or species not in survey.elements
        else (1.0 if moment > 0 else -1.0) * survey.elements[species].qe_starting_magnetization
        for species, moment in zip(config.species, config.initial_moments_muB, strict=True)
    )
    return MagneticConfiguration.model_validate(
        {**config.model_dump(), "qe_starting_magnetization": polarization}
    )


def survey_configurations(
    parent: Atoms, structure_id: str, survey: MagneticSurvey, *, nn_cutoff_A: float
) -> list[MagneticConfiguration]:
    """Nonmagnetic, ferromagnetic and antiferromagnetic states of an ideal parent cell.

    A parent without any surveyed element has only the nonmagnetic state.
    """
    nonmagnetic = gen.nonmagnetic(parent, structure_id)
    present = set(parent.get_chemical_symbols())
    moments = {
        symbol: start.moment_muB for symbol, start in survey.elements.items() if symbol in present
    }
    if not moments:
        return [nonmagnetic]
    ferromagnetic = gen.ferromagnetic(parent, structure_id, moments)
    sublattices = gen.bipartite_sublattices(parent, nn_cutoff_A)
    try:
        antiferromagnetic = gen.antiferromagnetic(parent, structure_id, sublattices, moments)
    except ValidationError as error:
        raise ValueError(
            "antiferromagnetic order needs the magnetic atoms on both sublattices of the parent"
        ) from error
    return [
        nonmagnetic,
        _with_polarization(ferromagnetic, survey),
        _with_polarization(antiferromagnetic, survey),
    ]


def derive_survey_for_defect(
    parent_configs: Sequence[MagneticConfiguration],
    defect_atoms: Atoms,
    defect_structure_id: str,
    defect_kind: str,
    survey: MagneticSurvey,
) -> list[MagneticConfiguration]:
    """Each parent state carried to a defect cell; atoms new to the cell start with no moment."""
    zero = InsertedMoment(policy="zero_documented")
    return [
        _with_polarization(
            derive_defect_configuration(
                parent,
                defect_atoms,
                defect_structure_id=defect_structure_id,
                defect_kind=defect_kind,
                inserted_policy=zero,
            ),
            survey,
        )
        for parent in parent_configs
    ]
