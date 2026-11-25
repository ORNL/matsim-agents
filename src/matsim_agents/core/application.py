"""Application configuration: what a scientific application asks the framework to do.

An application (a campaign, a study, a demo) lives outside the package, under ``examples/``
or in a user's own repository, and is described by one YAML file. The framework reads the
file with :func:`load_application` and never hardcodes what is in it: the material system,
the supercell, the species, the properties and the workflow are all application data.

Relative paths in the file (``policy``, ``workflow_config``) resolve against the file's
directory.
"""

from __future__ import annotations

import os
from math import prod
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

ATOMS_PER_CONVENTIONAL_CELL = {"sc": 1, "bcc": 2, "fcc": 4}


class SupercellTarget(BaseModel):
    """A fixed-composition supercell on a cubic parent lattice, checked for consistency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    composition: dict[str, StrictInt] = Field(min_length=1)
    supercell: tuple[StrictInt, StrictInt, StrictInt]
    parent_lattice: Literal["sc", "bcc", "fcc"]
    expected_sites: StrictInt = Field(gt=0)
    reference: str = ""

    @property
    def n_sites(self) -> int:
        return ATOMS_PER_CONVENTIONAL_CELL[self.parent_lattice] * prod(self.supercell)

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(sorted(self.composition))

    @property
    def formula(self) -> str:
        return "".join(f"{el}{n}" for el, n in self.composition.items())

    @model_validator(mode="after")
    def _consistent(self) -> SupercellTarget:
        if any(n <= 0 for n in self.supercell):
            raise ValueError("supercell repeats must be positive")
        if self.n_sites != self.expected_sites:
            raise ValueError(
                f"target must have {self.expected_sites} sites; "
                f"{self.parent_lattice} supercell {self.supercell} gives {self.n_sites}"
            )
        if any(n <= 0 for n in self.composition.values()):
            raise ValueError("every target element needs a positive atom count")
        if sum(self.composition.values()) != self.expected_sites:
            raise ValueError(f"target composition must sum to {self.expected_sites} atoms")
        return self


class ApplicationConfig(BaseModel):
    """One application: its material system, workflow and the files it brings along."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    paradigm: str = Field(
        min_length=1, description="Kind of study, e.g. supercell_alloy, prototype_search."
    )
    workflow: str = Field(min_length=1, description="Framework workflow the application drives.")
    target: SupercellTarget | None = None
    compositions: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    properties: tuple[str, ...] = ()
    policy: Path | None = None
    workflow_config: Path | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _consistent(self) -> ApplicationConfig:
        if self.target is None and not self.compositions:
            raise ValueError("an application needs a target supercell or a list of compositions")
        if self.target is not None and self.species:
            missing = sorted(set(self.target.elements) - set(self.species))
            if missing:
                raise ValueError(f"species must include every target element; missing {missing}")
        return self

    def resolve(self, path: Path | None) -> Path | None:
        """A path from the file, made absolute against the file's directory."""
        if path is None or path.is_absolute() or self.source is None:
            return path
        return self.source.parent / path


def load_application(path: str | os.PathLike[str]) -> ApplicationConfig:
    """Read and validate an application YAML file; referenced files must exist."""
    source = Path(path).resolve()
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{source} does not contain a mapping")
    app = ApplicationConfig.model_validate({**data, "source": source})
    for field in ("policy", "workflow_config"):
        resolved = app.resolve(getattr(app, field))
        if resolved is not None and not resolved.is_file():
            raise FileNotFoundError(f"{field} {resolved} referenced by {source} does not exist")
    return app


__all__ = [
    "ATOMS_PER_CONVENTIONAL_CELL",
    "ApplicationConfig",
    "SupercellTarget",
    "load_application",
]
