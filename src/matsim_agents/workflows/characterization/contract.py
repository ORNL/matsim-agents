"""The characterization contract: the question, its methods and tolerances, and loading."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.dft.qe_relax import QESettings
from matsim_agents.backends.dft.vasp_relax import VASPSettings
from matsim_agents.backends.mlip.config import MLIPConfig, resolve_mlip_block
from matsim_agents.core.composition import parse_composition
from matsim_agents.core.contracts import (
    ApprovalPolicy,
    ComputeBudget,
    EvidenceLevel,
    ObservableRequirement,
    at_least,
)
from matsim_agents.core.hashing import sha256_file
from matsim_agents.core.yamlio import load_yaml_mapping
from matsim_agents.science.properties.calculators import PropertyConditions
from matsim_agents.science.properties.recipes import (
    PROPERTY_RECIPES,
)
from matsim_agents.science.properties.recipes import RECIPES as PROPERTY_RECIPE_SPECS
from matsim_agents.science.thermodynamics.stability import (
    ReferenceEnergySet,
)
from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    GeometryControls,
)

Recipe = Literal[
    "relaxation",
    "energy_ranking",
    "thermodynamic_stability",
    "mechanical_elasticity",
    "electrical_transport",
    "thermal_transport",
    "multiproperty_characterization",
]


RECIPE_OBSERVABLES: dict[str, list[tuple[str, str, EvidenceLevel]]] = {
    "relaxation": [
        ("relaxed_structure", "structure", EvidenceLevel.MLIP_RELAXATION),
        ("energy_per_atom", "eV/atom", EvidenceLevel.MLIP_RELAXATION),
        ("residual_max_force", "eV/A", EvidenceLevel.MLIP_RELAXATION),
    ],
    "energy_ranking": [
        ("relative_energy_per_atom", "eV/atom", EvidenceLevel.MLIP_RELAXATION),
    ],
    "thermodynamic_stability": [
        ("formation_energy_per_atom", "eV/atom", EvidenceLevel.CONVERGED_DFT),
        ("energy_above_hull", "eV/atom", EvidenceLevel.CONVERGED_DFT),
    ],
    **{
        name: [(n, u, level) for n, u, level in recipe.observables]
        for name, recipe in PROPERTY_RECIPE_SPECS.items()
    },
}


# DFT settings the relaxation workflow sets itself, per backend.
_WORKFLOW_SETTINGS = {
    "qe": {"calculation", "pseudo_dir", "nstep", "forc_conv_thr_ry_au"},
    "vasp": {"calculation", "nsw", "ediffg_eV_per_A"},
}


_SETTINGS_CLASSES = {"qe": QESettings, "vasp": VASPSettings}


# Input keys the recipe's geometry, convergence, and method settings control;
# a raw override (QE namelists, VASP INCAR tags) may not replace them.
_CONTROLLED_INPUT_KEYS = {
    "qe": {
        "extra_control": {"calculation", "pseudo_dir", "nstep", "forc_conv_thr", "etot_conv_thr"},
        "extra_system": {"ecutwfc", "ecutrho", "occupations", "smearing", "degauss"},
        "extra_electrons": {"conv_thr", "mixing_beta"},
        "extra_ions": set(),
        "extra_cell": {"cell_dofree", "press"},
    },
    "vasp": {
        "extra_incar": {
            "IBRION",
            "ISIF",
            "NSW",
            "EDIFFG",
            "ENCUT",
            "PREC",
            "EDIFF",
            "ISYM",
            "ISMEAR",
            "SIGMA",
            "KSPACING",
            "KGAMMA",
            "ISPIN",
            "LREAL",
            "LWAVE",
            "LCHARG",
            "ALGO",
            "NELM",
            "NELMIN",
        },
    },
}


# Settings the backends otherwise choose per composition (cutoff, smearing,
# pseudopotentials, k-point density); a hull needs one method for all phases.
STABILITY_PINNED_SETTINGS = {
    "qe": ("ecutwfc_ry", "ecutrho_ry", "occupations", "pseudopotentials"),
    "vasp": ("encut_ev", "ismear", "sigma", "kspacing"),
}


MAX_RECIPE_EVIDENCE = EvidenceLevel.CONVERGED_DFT


# Contract fields that control execution, not the question: changing them
# keeps the run's completed calculations.
OPERATIONAL_FIELDS = {"budget", "approvals", "dft_approved", "output_root"}


class CharacterizationMethods(BaseModel):
    """How relaxations are computed."""

    mlip: MLIPConfig | None = Field(
        None, description="MLIP backend and its settings, as for `matsim-agents relax`."
    )
    dft: DFTBackendConfig | None = None
    dft_method_signature: str | None = Field(
        None,
        description="The DFT method's scientific signature; thermodynamic stability "
        "requires it to equal the reference set's.",
    )


class NumericalTolerances(BaseModel):
    force_tolerance_eV_per_A: float = Field(0.02, gt=0)
    max_relaxation_steps: int = Field(200, ge=1)
    degeneracy_tolerance_eV_per_atom: float = Field(0.01, ge=0)
    hull_tolerance_eV_per_atom: float = Field(
        0.0, ge=0, description="Largest energy above the hull reported as on the hull."
    )


class CharacterizationContract(BaseModel):
    """A characterization question, the evidence that answers it, and its limits."""

    schema_version: Literal[1] = 1
    contract_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    objective: str = Field(min_length=1, description="The question, in words.")
    formula: str
    structures: list[Path] = Field(min_length=1)
    recipe: Recipe
    observables: list[ObservableRequirement] = Field(
        default_factory=list, description="The recipe's observables when empty."
    )
    methods: CharacterizationMethods = Field(default_factory=CharacterizationMethods)
    geometry: GeometryControls = Field(
        default_factory=lambda: GeometryControls(relax_atoms=True, relax_cell=True),
        description="Degrees of freedom of every relaxation; atoms and cell by default.",
    )
    tolerances: NumericalTolerances = Field(default_factory=NumericalTolerances)
    reference_energies: Path | None = Field(
        None, description="ReferenceEnergySet JSON (thermodynamic_stability)."
    )
    property_evidence: list[Path] = Field(
        default_factory=list,
        description="JSON files of PropertyEvidence records (property recipes).",
    )
    conditions: PropertyConditions = Field(
        default_factory=PropertyConditions,
        description="Operating conditions the property recipes must be evaluated at.",
    )
    mlip_supports_stress: bool = False
    transport_executables: dict[str, str] = Field(default_factory=dict)
    node_hours_per_calculation: float | None = Field(None, gt=0)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    approvals: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    dft_approved: bool = False
    output_root: Path = Path("./runs")

    @model_validator(mode="after")
    def _consistent(self) -> CharacterizationContract:
        if parse_composition(self.formula) is None:
            raise ValueError(f"cannot parse formula {self.formula!r}")
        missing_files = [str(path) for path in self.structures if not path.is_file()]
        if missing_files:
            raise ValueError(f"structures do not exist: {missing_files}")
        defaults = {
            name: ObservableRequirement(name=name, unit=unit, minimum_evidence_level=level)
            for name, unit, level in RECIPE_OBSERVABLES[self.recipe]
        }
        given = {requirement.name: requirement for requirement in self.observables}
        unknown = sorted(set(given) - set(defaults))
        if unknown:
            raise ValueError(f"the {self.recipe} recipe does not produce {unknown}")
        if not self.geometry.relax_atoms:
            raise ValueError(
                "cell-only relaxation (relax_atoms: false) is not supported: the MLIP and "
                "DFT relaxations would still move the atoms"
            )
        if self.geometry.pressure_GPa:
            raise ValueError(
                "relaxation under pressure (geometry.pressure_GPa) is not supported: the "
                "relaxations run at zero pressure"
            )
        self.observables = [given.get(name, default) for name, default in defaults.items()]
        unreachable = [
            r.name
            for r in self.observables
            if self.recipe not in PROPERTY_RECIPES
            and not at_least(MAX_RECIPE_EVIDENCE, r.minimum_evidence_level)
        ]
        if unreachable:
            raise ValueError(
                f"{unreachable} require more than {MAX_RECIPE_EVIDENCE.value}, the highest "
                "fidelity a characterization recipe produces"
            )
        if self.recipe == "thermodynamic_stability":
            weak = [
                r.name
                for r in self.observables
                if not at_least(r.minimum_evidence_level, EvidenceLevel.CONVERGED_DFT)
            ]
            if weak:
                raise ValueError(
                    f"thermodynamic stability needs converged DFT for {weak}: MLIP or "
                    "unconverged energies cannot establish it"
                )
            if self.reference_energies is None or not self.reference_energies.is_file():
                raise ValueError("thermodynamic stability needs a reference_energies file")
            if not self.methods.dft_method_signature:
                raise ValueError("thermodynamic stability needs methods.dft_method_signature")
            if not (self.geometry.relax_atoms and self.geometry.relax_cell):
                raise ValueError(
                    "thermodynamic stability compares fully relaxed energies: "
                    "geometry must relax atoms and cell"
                )
            dft = self.methods.dft
            if dft is not None:
                unpinned = [
                    key
                    for key in STABILITY_PINNED_SETTINGS[dft.backend]
                    if dft.settings.get(key) is None
                ]
                if unpinned:
                    raise ValueError(
                        f"thermodynamic stability needs methods.dft.settings to fix {unpinned}: "
                        f"{dft.backend} otherwise chooses them per composition, and the "
                        "references and the material must share one method"
                    )
        absent = [str(p) for p in self.property_evidence if not p.is_file()]
        if absent:
            raise ValueError(f"property_evidence files do not exist: {absent}")
        if self.recipe in PROPERTY_RECIPES and self.recipe != "mechanical_elasticity":
            t = self.conditions.temperature_K
            if t is None or t <= 0:
                raise ValueError(f"the {self.recipe} recipe needs conditions.temperature_K > 0")
        if self.recipe == "energy_ranking" and len(self.structures) < 2:
            raise ValueError("an energy ranking needs at least two structures")
        if self.needs_dft() and self.methods.dft is None:
            raise ValueError("DFT-level observables need methods.dft")
        if self.methods.dft is not None:
            backend = self.methods.dft.backend
            known = {f.name for f in dataclasses.fields(_SETTINGS_CLASSES[backend])}
            allowed = known - _WORKFLOW_SETTINGS[backend]
            invalid = sorted(set(self.methods.dft.settings) - allowed)
            if invalid:
                raise ValueError(
                    f"methods.dft.settings {invalid} are not {backend} settings the "
                    f"contract may set; allowed: {sorted(allowed)}"
                )
            conflicts = [
                f"{group}.{key}"
                for group, controlled in _CONTROLLED_INPUT_KEYS[backend].items()
                for key in (self.methods.dft.settings.get(group) or {})
                if key.lower() in {c.lower() for c in controlled}
            ]
            if conflicts:
                raise ValueError(
                    f"methods.dft.settings overrides {conflicts}, which the recipe controls "
                    "(geometry, convergence, and the method's fields): set the "
                    "corresponding contract or settings field instead"
                )
        return self

    def required_level(self) -> EvidenceLevel:
        return max(
            (r.minimum_evidence_level for r in self.observables),
            key=list(EvidenceLevel).index,
        )

    def needs_dft(self) -> bool:
        return at_least(self.required_level(), EvidenceLevel.LOW_FIDELITY_DFT)

    def digest(self) -> str:
        """Fingerprint of the scientific question: its fields and its input files' contents.

        Budget, approvals, and output_root are excluded, so raising a budget or
        approving DFT resumes the same run.
        """
        inputs = [*self.structures, *self.property_evidence]
        if self.reference_energies is not None:
            inputs.append(self.reference_energies)
        payload = {
            "contract": self.model_dump(mode="json", exclude=OPERATIONAL_FIELDS),
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _references(contract: CharacterizationContract) -> ReferenceEnergySet:
    return ReferenceEnergySet.model_validate_json(
        Path(str(contract.reference_energies)).read_text()
    )


def load_contract(path: str | Path) -> CharacterizationContract:
    """Read a YAML or JSON contract; relative paths in it are relative to the file."""
    path = Path(path).resolve()
    data = load_yaml_mapping(path)

    def resolve(value: Any) -> Any:
        return str(path.parent / value) if isinstance(value, str) else value

    data["structures"] = [resolve(item) for item in data.get("structures") or []]
    data.setdefault("output_root", "runs")
    for key in ("reference_energies", "output_root"):
        if key in data:
            data[key] = resolve(data[key])
    resolve_mlip_block(data.get("methods") or {}, path.parent)
    dft = (data.get("methods") or {}).get("dft")
    if isinstance(dft, dict):
        for key in ("pseudo_dir", "potcar_dir"):
            if dft.get(key):
                dft[key] = resolve(dft[key])
    return CharacterizationContract.model_validate(data)
