"""Gas and transmutation species (Re, Os, H, He) in magnetic alloys: support, spin and MLIP gates.

Nothing here assigns a permanent moment to Re, Os, H or He. Inserted or transmuted atoms start
from an explicit value or a documented zero. Local re-initialization around a species and
alternative magnetic solutions are explicit overrides that carry their own assumption text.
Pseudopotential identity is read from the UPF header, never inferred from the file name, and a
missing file blocks the calculation. MLIP element coverage is checked before any prediction; an
unsupported species yields ``MLIP_UNSUPPORTED`` and no number is invented.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.discovery.magnetism import generators as gen
from matsim_agents.discovery.magnetism.config import (
    MagneticConfiguration,
    qe_polarization_from_moment,
)
from matsim_agents.discovery.magnetism.defects import InsertedMoment, derive_defect_configuration

SUPPORT_STATEMENT = (
    "species metadata and starting-state conventions only; magnetic behaviour of Re, Os, H and "
    "He in the reference alloy is not established"
)


class SpeciesSupportError(ValueError):
    """Raised for an invalid species, override or unsupported request."""


class SpeciesInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    atomic_number: int
    role: Literal["transmutation_solute", "interstitial_gas"]
    default_inserted_policy: Literal["zero_documented"] = "zero_documented"
    permanent_moment_assigned: Literal[False] = False
    notes: str


SPECIES: Mapping[str, SpeciesInfo] = {
    "Re": SpeciesInfo(
        symbol="Re",
        atomic_number=75,
        role="transmutation_solute",
        notes="5d solute replacing W; starts unpolarized; induced moments in Cr-containing "
        "neighbourhoods must be tested, not assumed.",
    ),
    "Os": SpeciesInfo(
        symbol="Os",
        atomic_number=76,
        role="transmutation_solute",
        notes="5d solute replacing W; starts unpolarized; induced moments in Cr-containing "
        "neighbourhoods must be tested, not assumed.",
    ),
    "H": SpeciesInfo(
        symbol="H",
        atomic_number=1,
        role="interstitial_gas",
        notes="starts unpolarized; an induced moment or polarized neighbours is an explicit "
        "alternative solution.",
    ),
    "He": SpeciesInfo(
        symbol="He",
        atomic_number=2,
        role="interstitial_gas",
        notes="closed shell; starts unpolarized; neighbouring host moments may still change.",
    ),
}
NEW_SPECIES = tuple(SPECIES)


def species_info(symbol: str) -> SpeciesInfo:
    try:
        return SPECIES[symbol]
    except KeyError as exc:
        raise SpeciesSupportError(
            f"{symbol!r} is not a registered gas/transmutation species"
        ) from exc


# --------------------------------------------------------------------------- #
# Magnetic handling                                                           #
# --------------------------------------------------------------------------- #


def inserted_species_moments(
    defect_atoms: Atoms,
    parent: MagneticConfiguration,
    explicit: Mapping[str, float] | None = None,
) -> dict[int, InsertedMoment]:
    """Per-atom moments for atoms of new species or new ids: explicit per species, else zero."""
    explicit = dict(explicit or {})
    for symbol in explicit:
        species_info(symbol)
    parent_species = dict(zip(parent.atom_ids, parent.species, strict=True))
    out: dict[int, InsertedMoment] = {}
    for atom_id, symbol in zip(
        defect_atoms.get_array("atom_id"), defect_atoms.get_chemical_symbols(), strict=True
    ):
        atom_id = int(atom_id)
        if parent_species.get(atom_id) == symbol:
            continue
        if symbol in explicit:
            out[atom_id] = InsertedMoment(moment_muB=float(explicit[symbol]), policy="explicit")
        elif symbol in SPECIES:
            out[atom_id] = InsertedMoment(policy="zero_documented")
    return out


def spin_variants(
    parent: MagneticConfiguration,
    defect_atoms: Atoms,
    *,
    defect_structure_id: str,
    defect_kind: str,
    explicit_moments_muB: Mapping[str, float] | None = None,
) -> dict[str, MagneticConfiguration]:
    """Spin-unpolarized and spin-polarized configurations of a gas/transmuted structure.

    The polarized variant keeps every surviving atom's parent moment (by ``atom_id``); atoms of
    registered new species get an explicit moment or a documented zero.
    """
    variants = {"unpolarized": gen.nonmagnetic(defect_atoms, defect_structure_id)}
    if parent.spin_polarized:
        inserted = inserted_species_moments(defect_atoms, parent, explicit_moments_muB)
        variants["polarized"] = derive_defect_configuration(
            parent,
            defect_atoms,
            defect_structure_id=defect_structure_id,
            defect_kind=defect_kind,
            inserted=inserted,
        )
    return variants


def override_moments(
    config: MagneticConfiguration,
    overrides_muB: Mapping[int, float],
    *,
    variant_label: str,
    assumption: str,
) -> MagneticConfiguration:
    """Explicit local re-initialization or alternative solution; the assumption is mandatory."""
    if not assumption.strip():
        raise SpeciesSupportError("an override must document its assumption")
    if not config.spin_polarized or config.initial_moments_muB is None:
        raise SpeciesSupportError("overrides need a spin-polarized configuration with moments")
    unknown = sorted(set(overrides_muB) - set(config.atom_ids))
    if unknown:
        raise SpeciesSupportError(f"override atom_ids not in the configuration: {unknown}")
    moments = list(config.initial_moments_muB)
    groups = list(config.group_labels)
    labels: dict[tuple[str, float], str] = {}
    for i, atom_id in enumerate(config.atom_ids):
        if atom_id in overrides_muB:
            value = float(overrides_muB[atom_id])
            moments[i] = value
            key = (config.species[i], value)
            groups[i] = labels.setdefault(key, f"{config.species[i]}_{variant_label}_{len(labels)}")
    base = config.model_dump(
        exclude={"config_id", "mode", "initial_moments_muB", "group_labels", "assumptions"}
    )
    base["qe_starting_magnetization"] = None
    base["converged_moments_muB"] = None
    base["converged_classification"] = None
    base["initial_moments_muB"] = tuple(moments)
    base["group_labels"] = tuple(groups)
    base["assumptions"] = (
        *config.assumptions,
        f"{variant_label}: {assumption}",
        "QE starting polarizations must be regenerated after the override",
    )
    base["provenance"] = {
        **config.provenance,
        "override_of_config_id": config.config_id,
        "override_variant": variant_label,
        "override_atom_ids": sorted(overrides_muB),
    }
    for mode in (config.mode, "custom_collinear"):
        try:
            cfg = MagneticConfiguration(config_id=f"{config.structure_id}-tmp", mode=mode, **base)
        except ValueError:
            continue
        return cfg.model_copy(
            update={"config_id": f"{config.structure_id}-{variant_label}-{cfg.fingerprint()[:10]}"}
        )
    raise SpeciesSupportError("override cannot be represented as a collinear configuration")


def select_policy_for_host(atoms: Atoms):
    """Policy chosen from the host elements only; Re/Os/H/He never silently extend a policy.

    Returns ``(policy, note)``. The note records which registered species were set aside so the
    caller can store it as an assumption.
    """
    from matsim_agents.discovery.magnetism.policies import POLICIES

    symbols = set(atoms.get_chemical_symbols())
    host = symbols - set(SPECIES)
    set_aside = sorted(symbols & set(SPECIES))
    policy = next((p for p in POLICIES if p.matches(host)), None)
    note = (
        f"policy {policy.name if policy else None} chosen from host elements {sorted(host)}; "
        f"{set_aside} start unpolarized unless an explicit moment is given"
    )
    return policy, note


# --------------------------------------------------------------------------- #
# QE pseudopotential identity and compatibility                               #
# --------------------------------------------------------------------------- #

_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class UPFHeader:
    element: str
    z_valence: float
    functional: str
    pseudo_type: str
    is_paw: bool
    has_so: bool
    core_correction: bool
    wfc_cutoff_ry: float | None
    rho_cutoff_ry: float | None
    sha256: str


def parse_upf_header(path: str | Path) -> UPFHeader:
    """Read identity from the UPF v2 ``PP_HEADER`` element (not from the file name)."""
    path = Path(path)
    text = path.read_bytes()
    match = re.search(rb"<PP_HEADER(.*?)/?>", text, re.S)
    if match is None:
        raise SpeciesSupportError(f"{path.name}: no UPF v2 PP_HEADER found")
    attrs = {k: v for k, v in _ATTR.findall(match.group(1).decode("latin-1"))}
    try:
        z_valence = float(attrs["z_valence"])
        element = attrs["element"].strip()
    except (KeyError, ValueError) as exc:
        raise SpeciesSupportError(f"{path.name}: header lacks element or z_valence") from exc

    def number(key: str) -> float | None:
        return float(attrs[key]) if key in attrs else None

    def flag(key: str) -> bool:
        return attrs.get(key, "false").strip().lower() in ("true", ".true.", "t")

    return UPFHeader(
        element=element,
        z_valence=z_valence,
        functional=" ".join(attrs.get("functional", "").split()),
        pseudo_type=attrs.get("pseudo_type", "").strip(),
        is_paw=flag("is_paw"),
        has_so=flag("has_so"),
        core_correction=flag("core_correction"),
        wfc_cutoff_ry=number("wfc_cutoff"),
        rho_cutoff_ry=number("rho_cutoff"),
        sha256=hashlib.sha256(text).hexdigest(),
    )


PseudoStatus = Literal["VERIFIED", "MISSING", "MISMATCH"]


class PseudoRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    species: str
    status: PseudoStatus
    file: str | None = None
    sha256: str | None = None
    header_element: str | None = None
    z_valence: float | None = None
    functional: str | None = None
    pseudo_type: str | None = None
    wfc_cutoff_hint_ry: float | None = None
    rho_cutoff_hint_ry: float | None = None
    problems: tuple[str, ...] = ()


class PseudoCompatibility(BaseModel):
    """Per-species identity plus a cross-species consistency verdict."""

    model_config = ConfigDict(frozen=True)

    records: tuple[PseudoRecord, ...]
    functionals: tuple[str, ...]
    consistent_functional: bool
    ready: bool
    cutoff_hint_ry: tuple[float | None, float | None]
    numerical_compatibility: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    statement: str = (
        "cutoff values are UPF-header maxima, not convergence-tested; mixing species is "
        "software-checked only (functional and element identity), not numerically validated"
    )

    @property
    def blockers(self) -> tuple[str, ...]:
        out = [f"{r.species}: {p}" for r in self.records for p in r.problems]
        if not self.consistent_functional:
            out.append(f"mixed exchange-correlation functionals: {list(self.functionals)}")
        return tuple(out)


def assess_pseudopotentials(
    species: Iterable[str],
    pseudo_dir: str | Path,
    files: Mapping[str, str],
    *,
    spin_polarized: bool = False,
) -> PseudoCompatibility:
    """Verify that each species has a UPF whose header names that element."""
    pdir = Path(pseudo_dir)
    records: list[PseudoRecord] = []
    for symbol in sorted(set(species)):
        name = files.get(symbol)
        path = pdir / name if name else None
        if path is None or not path.is_file():
            records.append(
                PseudoRecord(
                    species=symbol,
                    status="MISSING",
                    file=name,
                    problems=("no pseudopotential file; calculation is INPUT_BLOCKED",),
                )
            )
            continue
        header = parse_upf_header(path)
        problems: list[str] = []
        if header.element != symbol:
            problems.append(f"header element {header.element!r} does not match species")
        if header.has_so:
            problems.append("fully relativistic pseudopotential; collinear spin is not defined")
        if spin_polarized and not header.core_correction:
            problems.append("no nonlinear core correction; spin-polarized use is questionable")
        records.append(
            PseudoRecord(
                species=symbol,
                status="MISMATCH" if problems else "VERIFIED",
                file=name,
                sha256=header.sha256,
                header_element=header.element,
                z_valence=header.z_valence,
                functional=header.functional,
                pseudo_type=header.pseudo_type,
                wfc_cutoff_hint_ry=header.wfc_cutoff_ry,
                rho_cutoff_hint_ry=header.rho_cutoff_ry,
                problems=tuple(problems),
            )
        )
    functionals = tuple(sorted({r.functional for r in records if r.functional}))
    wfc = [r.wfc_cutoff_hint_ry for r in records if r.wfc_cutoff_hint_ry is not None]
    rho = [r.rho_cutoff_hint_ry for r in records if r.rho_cutoff_hint_ry is not None]
    return PseudoCompatibility(
        records=tuple(records),
        functionals=functionals,
        consistent_functional=len(functionals) <= 1,
        ready=all(r.status == "VERIFIED" for r in records) and len(functionals) <= 1,
        cutoff_hint_ry=(max(wfc) if wfc else None, max(rho) if rho else None),
    )


def starting_polarization(moment_muB: float, record: PseudoRecord, qe_version: str) -> float:
    """mu_B -> QE starting_magnetization using the verified valence of ``record``."""
    if record.status != "VERIFIED" or record.z_valence is None:
        raise SpeciesSupportError(f"{record.species}: valence is not verified")
    return qe_polarization_from_moment(
        moment_muB,
        valence_electrons=record.z_valence,
        verified_valence=True,
        qe_version=qe_version,
    )


# --------------------------------------------------------------------------- #
# MLIP capability                                                             #
# --------------------------------------------------------------------------- #

MLIPStatus = Literal["SUPPORTED", "MLIP_UNSUPPORTED", "COVERAGE_UNDECLARED"]


class MLIPSupportCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    status: MLIPStatus
    required: tuple[str, ...]
    missing: tuple[str, ...] = ()
    prediction_allowed: bool
    dft_only_path_allowed: bool = True
    message: str = ""
    new_species_missing: tuple[str, ...] = Field(default=())


def check_mlip_support(
    model_id: str,
    required_species: Iterable[str],
    trained_elements: Iterable[str] | None,
) -> MLIPSupportCheck:
    """Gate MLIP use on declared training-element coverage.

    ``trained_elements=None`` means the model never declared its coverage, which is treated as
    not usable. An unsupported model must not produce a prediction; a budgeted DFT-only route
    stays allowed.
    """
    required = tuple(sorted(set(required_species)))
    if trained_elements is None:
        return MLIPSupportCheck(
            model_id=model_id,
            status="COVERAGE_UNDECLARED",
            required=required,
            prediction_allowed=False,
            message="model element coverage is undeclared; no prediction is made",
        )
    covered = set(trained_elements)
    missing = tuple(el for el in required if el not in covered)
    if missing:
        return MLIPSupportCheck(
            model_id=model_id,
            status="MLIP_UNSUPPORTED",
            required=required,
            missing=missing,
            prediction_allowed=False,
            new_species_missing=tuple(el for el in missing if el in SPECIES),
            message=f"model has no trained support for {list(missing)}; use the DFT-only path",
        )
    return MLIPSupportCheck(
        model_id=model_id, status="SUPPORTED", required=required, prediction_allowed=True
    )


def require_mlip_support(check: MLIPSupportCheck) -> None:
    """Raise instead of returning a number from an unsupported model."""
    if not check.prediction_allowed:
        raise SpeciesSupportError(f"{check.status}: {check.message}")
