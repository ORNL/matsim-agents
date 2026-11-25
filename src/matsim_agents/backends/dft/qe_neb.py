"""Configuration contract for Quantum ESPRESSO ``neb.x`` calculations.

Everything that influences the numbers is explicit and validated here; nothing is inferred from
the composition (no ``recommend_settings`` defaults), so a stored config fully describes the
calculation. No magnetic treatment is assumed: ``nspin`` and ``starting_magnetization`` must be
stated, and a structure containing Cr additionally needs a recorded ``magnetic_treatment``.

Units: ``path_thr_ev_ang`` is the NEB path convergence threshold in eV/Angstrom (the unit used
by ``neb.x``); ``prefix.dat`` residuals are also in eV/Angstrom (QE ``path_base.f90``).

This module checks consistency, not physical adequacy. A valid config says nothing about whether
cutoffs, k-points or pseudopotentials are converged for the material.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ase import Atoms
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

# QE neb.x needs more than three images; tests pin this to discovery.neb_contracts.MIN_IMAGES.
MIN_IMAGES = 4

OptScheme = Literal["sd", "broyden", "broyden2", "quick-min", "langevin"]
CIScheme = Literal["no-CI", "auto", "manual"]
RestartMode = Literal["from_scratch", "restart"]
Occupations = Literal["smearing", "fixed"]
Smearing = Literal["gaussian", "methfessel-paxton", "marzari-vanderbilt", "fermi-dirac"]

Scalar = str | bool | int | float

_PREFIX = re.compile(r"^[A-Za-z0-9_.-]+$")

# Keys that this contract owns; allowing them in the free-form dictionaries would let a caller
# silently contradict a validated field.
_MANAGED_SYSTEM = frozenset(
    {
        "ibrav",
        "nat",
        "ntyp",
        "ecutwfc",
        "ecutrho",
        "occupations",
        "smearing",
        "degauss",
        "nspin",
        "starting_magnetization",
    }
)
_MANAGED_ELECTRONS = frozenset({"conv_thr", "mixing_beta"})
_MANAGED_PATH = frozenset(
    {
        "string_method",
        "restart_mode",
        "nstep_path",
        "num_of_images",
        "opt_scheme",
        "ci_scheme",
        "path_thr",
        "minimum_image",
        "first_last_opt",
    }
)


def _freeze(value: Mapping) -> Mapping:
    return MappingProxyType(dict(sorted(value.items())))


def _check_extra(name: str, value: Mapping, managed: frozenset[str]) -> None:
    clash = sorted(k for k in value if k.lower() in managed)
    if clash:
        raise ValueError(f"{name} may not set keys managed by the config: {clash}")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name}[{key!r}] must be finite")


class QENebConfig(BaseModel):
    """Validated settings for one ``neb.x`` calculation."""

    model_config = ConfigDict(frozen=True)

    # --- NEB path ---------------------------------------------------------------------------
    n_images: StrictInt
    nstep_path: StrictInt = 100
    path_thr_ev_ang: float = 0.05
    opt_scheme: OptScheme = "broyden"
    ci_scheme: CIScheme = "no-CI"
    climbing_images: tuple[StrictInt, ...] = ()  # 1-based image numbers, manual CI only
    restart_mode: RestartMode = "from_scratch"
    minimum_image: StrictBool = True
    extra_path: Mapping[str, Scalar] = Field(default_factory=dict)

    # --- PW engine --------------------------------------------------------------------------
    species: tuple[str, ...] = Field(min_length=1)
    pseudo_dir: str = Field(min_length=1)
    pseudopotentials: Mapping[str, str]
    ecutwfc_ry: float
    ecutrho_ry: float
    occupations: Occupations
    smearing: Smearing = "marzari-vanderbilt"
    degauss_ry: float | None = None
    kpts: tuple[StrictInt, StrictInt, StrictInt]
    koffset: tuple[StrictInt, StrictInt, StrictInt] = (0, 0, 0)
    conv_thr_ry: float = 1.0e-8
    mixing_beta: float = 0.3
    nspin: Literal[1, 2]
    starting_magnetization: Mapping[str, float] = Field(default_factory=dict)
    magnetic_treatment: str | None = None
    extra_system: Mapping[str, Scalar] = Field(default_factory=dict)
    extra_electrons: Mapping[str, Scalar] = Field(default_factory=dict)

    # --- Execution --------------------------------------------------------------------------
    neb_executable: str = "neb.x"
    launcher: tuple[str, ...] = ()  # e.g. ("srun", "-n", "8"); empty runs neb.x directly
    nimage: StrictInt = 1  # neb.x -nimage
    npool: StrictInt = 1  # neb.x -npool
    extra_args: tuple[str, ...] = ()
    workdir: str = Field(default=".", min_length=1)
    outdir: str = Field(default="./tmp", min_length=1)
    prefix: str = "neb"

    @field_serializer(
        "extra_path",
        "pseudopotentials",
        "starting_magnetization",
        "extra_system",
        "extra_electrons",
    )
    def _serialize_mapping(self, value: Mapping) -> dict:
        return dict(value)

    @field_validator(
        "extra_path",
        "pseudopotentials",
        "starting_magnetization",
        "extra_system",
        "extra_electrons",
    )
    @classmethod
    def _freeze_mapping(cls, value: Mapping) -> Mapping:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> QENebConfig:
        if self.n_images < MIN_IMAGES:
            raise ValueError(f"n_images must be at least {MIN_IMAGES}, got {self.n_images}")
        if self.nstep_path < 1:
            raise ValueError("nstep_path must be at least 1")
        if not (math.isfinite(self.path_thr_ev_ang) and self.path_thr_ev_ang > 0):
            raise ValueError("path_thr_ev_ang must be a positive finite number (eV/Angstrom)")
        self._check_climbing()
        self._check_pw()
        self._check_magnetism()
        self._check_execution()
        _check_extra("extra_path", self.extra_path, _MANAGED_PATH)
        _check_extra("extra_system", self.extra_system, _MANAGED_SYSTEM)
        _check_extra("extra_electrons", self.extra_electrons, _MANAGED_ELECTRONS)
        return self

    def _check_climbing(self) -> None:
        if self.ci_scheme == "manual":
            if not self.climbing_images:
                raise ValueError("ci_scheme 'manual' needs climbing_images")
            if len(set(self.climbing_images)) != len(self.climbing_images):
                raise ValueError("climbing_images must be unique")
            bad = [i for i in self.climbing_images if not 2 <= i <= self.n_images - 1]
            if bad:
                raise ValueError(
                    f"climbing images {bad} must be interior images (2..{self.n_images - 1})"
                )
        elif self.climbing_images:
            raise ValueError(
                f"climbing_images is only valid with ci_scheme 'manual', not {self.ci_scheme!r}"
            )

    def _check_pw(self) -> None:
        if len(set(self.species)) != len(self.species) or any(not s for s in self.species):
            raise ValueError("species must be unique nonempty element symbols")
        missing = [s for s in self.species if not self.pseudopotentials.get(s)]
        if missing:
            raise ValueError(f"missing pseudopotentials for: {missing}")
        extra = sorted(set(self.pseudopotentials) - set(self.species))
        if extra:
            raise ValueError(f"pseudopotentials given for species not in the structure: {extra}")
        for name, value in (("ecutwfc_ry", self.ecutwfc_ry), ("ecutrho_ry", self.ecutrho_ry)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number")
        if self.ecutrho_ry < 4.0 * self.ecutwfc_ry:
            raise ValueError("ecutrho_ry must be at least 4 * ecutwfc_ry")
        if any(k < 1 for k in self.kpts):
            raise ValueError("kpts must be positive integers")
        if any(o not in (0, 1) for o in self.koffset):
            raise ValueError("koffset entries must be 0 or 1")
        if not (math.isfinite(self.conv_thr_ry) and self.conv_thr_ry > 0):
            raise ValueError("conv_thr_ry must be a positive finite number")
        if not (math.isfinite(self.mixing_beta) and 0 < self.mixing_beta <= 1):
            raise ValueError("mixing_beta must be in (0, 1]")
        if self.occupations == "smearing":
            if self.degauss_ry is None or not (
                math.isfinite(self.degauss_ry) and self.degauss_ry > 0
            ):
                raise ValueError("smearing occupations need a positive degauss_ry")
        elif self.degauss_ry is not None:
            raise ValueError("degauss_ry is only valid with smearing occupations")

    def _check_magnetism(self) -> None:
        if self.nspin == 1:
            if self.starting_magnetization:
                raise ValueError("starting_magnetization requires nspin=2")
        else:
            absent = [s for s in self.species if s not in self.starting_magnetization]
            if absent:
                raise ValueError(f"nspin=2 needs starting_magnetization for: {absent}")
            for symbol, value in self.starting_magnetization.items():
                if symbol not in self.species:
                    raise ValueError(f"starting_magnetization for unknown species {symbol!r}")
                if not (math.isfinite(value) and -1.0 <= value <= 1.0):
                    raise ValueError(f"starting_magnetization[{symbol!r}] must be in [-1, 1]")
        if "Cr" in self.species and not (self.magnetic_treatment or "").strip():
            raise ValueError(
                "Cr is present: record how its magnetism is treated in magnetic_treatment "
                "(there is no universal default)"
            )

    def _check_execution(self) -> None:
        if not self.neb_executable.strip():
            raise ValueError("neb_executable must be nonempty")
        if any(not a for a in (*self.launcher, *self.extra_args)):
            raise ValueError("launcher and extra_args entries must be nonempty")
        if self.nimage < 1 or self.npool < 1:
            raise ValueError("nimage and npool must be at least 1")
        if not _PREFIX.match(self.prefix):
            raise ValueError("prefix may only contain letters, digits, '_', '.' and '-'")
        for name in ("-inp", "-i", "-in", "-input", "-nimage", "-ni", "-npool", "-nk"):
            if name in self.extra_args:
                raise ValueError(f"extra_args may not contain {name}; it is set by the config")

    # --- Derived views ----------------------------------------------------------------------
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form; stable across runs and key order."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def check_structure(self, atoms: Atoms) -> None:
        """Raise ``ValueError`` unless ``atoms`` has exactly the configured species."""
        found = sorted(set(atoms.get_chemical_symbols()))
        if found != sorted(self.species):
            raise ValueError(
                f"structure species {found} differ from configured species {sorted(self.species)}"
            )

    def runtime_problems(self) -> list[str]:
        """Problems that depend on the filesystem; empty when the config can be run here."""
        problems: list[str] = []
        pseudo_dir = Path(self.pseudo_dir)
        if not pseudo_dir.is_dir():
            problems.append(f"pseudo_dir {self.pseudo_dir!r} is not a directory")
        else:
            for symbol in self.species:
                name = self.pseudopotentials[symbol]
                if not (pseudo_dir / name).is_file():
                    problems.append(f"pseudopotential {name!r} for {symbol} not found")
        exe = self.neb_executable
        if os.sep in exe and not (Path(exe).is_file() and os.access(exe, os.X_OK)):
            problems.append(f"neb executable {exe!r} is not an executable file")
        return problems


__all__ = [
    "CIScheme",
    "OptScheme",
    "QENebConfig",
    "RestartMode",
]
