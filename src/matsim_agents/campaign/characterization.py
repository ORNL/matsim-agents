"""Characterize a material against an explicit contract with deterministic recipes.

A characterization answers one question about given structures of one
composition, with no optimization objective. Its contract
(:class:`CharacterizationContract`) declares the objective, the observables
with their units and minimum fidelity, the methods, the numerical tolerances,
the references, the budget, and the approvals. A recipe plans the calculations
as an ordered task graph (:func:`plan_recipe`); the numerical work is done by
the existing deterministic operators (:func:`matsim_agents.workflows.relaxation.
run_relaxation`, :func:`matsim_agents.discovery.stability.score_stability`), never
by an LLM. The answer (:class:`~matsim_agents.execution.contracts.
CharacterizationReport`) states a conclusion only when the evidence suffices
and says otherwise what is missing:

- ``relaxation``: relaxed structures with converged residual forces;
- ``energy_ranking``: the lowest-energy structure among same-composition
  candidates, which is not a thermodynamic-stability claim;
- ``thermodynamic_stability``: the energy above the convex hull of declared,
  method-compatible DFT elemental and competing-phase references. MLIP
  energies, clean relaxations, or a low total energy are never sufficient; a
  provisional reference set gives ``inconclusive``.

DFT runs only when a required observable needs DFT fidelity. Every step's
outcome is saved as it completes, so running the same contract again resumes
with the outstanding steps and never repeats a completed calculation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.dft.qe_relax import QESettings
from matsim_agents.backends.dft.vasp_relax import VASPSettings
from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.datasets.governance import sha256_file
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.stability import (
    RankingMode,
    ReferenceEnergySet,
    StabilityReport,
    score_stability,
)
from matsim_agents.execution.contracts import (
    DFT_EVIDENCE,
    ApprovalPolicy,
    CharacterizationReport,
    CharacterizationStatus,
    ComputeBudget,
    EvidenceItem,
    EvidenceLevel,
    ObservableRequirement,
    ObservableValue,
    at_least,
)
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
    hashes_of,
    software_versions,
)
from matsim_agents.execution.locking import LockTimeoutError, exclusive_lock
from matsim_agents.execution.recovery import (
    DEFAULT_POLICY,
    FailureClass,
    adjusted_settings,
    can_retry,
    classify_failure,
)
from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    GeometryControls,
    RelaxationMode,
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
    run_relaxation,
)

log = logging.getLogger(__name__)

Recipe = Literal["relaxation", "energy_ranking", "thermodynamic_stability"]
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
STATE_FILE = "characterization-state.json"
LEDGER_FILE = "evidence-ledger.jsonl"
REPORT_FILE = "characterization-report.json"


class CharacterizationMethods(BaseModel):
    """How relaxations are computed."""

    mlip: dict[str, Any] = Field(
        default_factory=dict, description="MLIP settings, as for `matsim-agents relax`."
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
            if not at_least(MAX_RECIPE_EVIDENCE, r.minimum_evidence_level)
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
        inputs = [*self.structures]
        if self.reference_energies is not None:
            inputs.append(self.reference_energies)
        payload = {
            "contract": self.model_dump(mode="json", exclude=OPERATIONAL_FIELDS),
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class RecipeStep(BaseModel):
    """One node of a recipe's task graph."""

    step_id: str
    operator: Literal["relax", "rank", "hull"]
    fidelity: Literal["mlip", "dft"] | None = None
    structure: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str]
    needs_dft_approval: bool = False
    max_attempts: int = Field(
        1,
        description="Attempts before recovery; failed DFT attempts are retried per "
        "execution.recovery.DEFAULT_POLICY within the budget.",
    )


def plan_recipe(contract: CharacterizationContract) -> list[RecipeStep]:
    """The ordered task graph of a contract's recipe (deterministic)."""
    fidelity: Literal["mlip", "dft"] = "dft" if contract.needs_dft() else "mlip"
    relax = [
        RecipeStep(
            step_id=f"relax-{fidelity}-{index:03d}",
            operator="relax",
            fidelity=fidelity,
            structure=str(path),
            produces=["relaxed_structure", "energy_per_atom", "residual_max_force"],
            needs_dft_approval=fidelity == "dft" and contract.approvals.before_dft,
        )
        for index, path in enumerate(contract.structures)
    ]
    if contract.recipe == "relaxation":
        return relax
    final = (
        RecipeStep(
            step_id="rank",
            operator="rank",
            requires=[step.step_id for step in relax],
            produces=["relative_energy_per_atom"],
        )
        if contract.recipe == "energy_ranking"
        else RecipeStep(
            step_id="hull",
            operator="hull",
            requires=[step.step_id for step in relax],
            produces=["formation_energy_per_atom", "energy_above_hull"],
        )
    )
    return [*relax, final]


Relax = Callable[[ScientificRelaxationConfig], ScientificRelaxationResult]


def _relaxation_config(
    contract: CharacterizationContract,
    step: RecipeStep,
    work_dir: Path,
    *,
    structure: str | None = None,
    dft: DFTBackendConfig | None = None,
) -> ScientificRelaxationConfig:
    return ScientificRelaxationConfig(
        mode=RelaxationMode.DFT if step.fidelity == "dft" else RelaxationMode.MLIP,
        structure_path=str(structure or step.structure),
        output_root=str(work_dir),
        geometry=contract.geometry,
        mlip=contract.methods.mlip,
        dft=(dft or contract.methods.dft) if step.fidelity == "dft" else None,
        approvals=contract.approvals,
        dft_approved=contract.dft_approved,
        max_steps=contract.tolerances.max_relaxation_steps,
        force_tolerance_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
    )


_ALLOCATION = ("nodes", "devices", "device_kind", "gpus")


def _relax_record(result: ScientificRelaxationResult) -> dict[str, Any]:
    """What a relaxation step contributes: a numerical outcome or an infrastructure failure."""
    stage = result.stages[-1] if result.stages else None
    if result.status == "failed" or stage is None:
        return {
            "status": "failed",
            "failure_reason": result.failure_reason or "no stage result",
            "failure_type": result.failure_type,
            "return_code": stage.return_code if stage else None,
            **({name: getattr(stage, name) for name in _ALLOCATION} if stage else {}),
        }
    diagnostics = {
        "return_code": stage.return_code,
        "scf_converged": stage.scf_converged,
        **{name: getattr(stage, name) for name in _ALLOCATION},
    }
    if (
        stage.energy_eV is None
        or stage.max_force_eV_per_A is None
        or stage.optimized_structure_path is None
    ):
        # The code ran but produced no usable result (e.g. it exited before the first step).
        return {
            "status": "failed",
            "failure_reason": stage.failure_reason
            or "the relaxation produced no energy, force, or structure",
            "run_directory": result.run_directory,
            **diagnostics,
        }
    if not (math.isfinite(stage.energy_eV) and math.isfinite(stage.max_force_eV_per_A)):
        return {
            "status": "failed",
            "failure_reason": (
                f"non-finite relaxation output: energy {stage.energy_eV} eV, "
                f"max force {stage.max_force_eV_per_A} eV/A"
            ),
            "run_directory": result.run_directory,
            **diagnostics,
        }
    return {
        "status": "done",
        "run_id": result.run_id,
        "run_directory": result.run_directory,
        "evidence_level": stage.evidence_level.value,
        "converged": stage.converged,
        "energy_eV": stage.energy_eV,
        "max_force_eV_per_A": stage.max_force_eV_per_A,
        "steps": stage.steps,
        "input_structure_path": stage.input_structure_path,
        "optimized_structure_path": stage.optimized_structure_path,
        "optimized_structure_sha256": sha256_file(stage.optimized_structure_path),
        "failure_reason": stage.failure_reason,
        **diagnostics,
    }


class _Run:
    """The persisted progress of one contract (``<output_root>/characterization-<id>``)."""

    def __init__(self, contract: CharacterizationContract) -> None:
        self.contract = contract
        self.directory = Path(contract.output_root) / f"characterization-{contract.contract_id}"
        self.directory.mkdir(parents=True, exist_ok=True)
        # One execution owns the run: its state is read, checked, and written under the
        # lock (a lock file, since flock is unsupported on NERSC CFS compute nodes).
        self._stack = contextlib.ExitStack()
        try:
            self._stack.enter_context(exclusive_lock(self.directory / ".lock", timeout=0))
        except LockTimeoutError:
            raise RuntimeError(
                f"{self.directory} is in use by another execution of this contract"
            ) from None
        try:
            self._open(contract)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> _Run:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _open(self, contract: CharacterizationContract) -> None:
        self.state_path = self.directory / STATE_FILE
        digest = contract.digest()
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text())
            if self.state["contract_sha256"] != digest:
                raise ValueError(
                    f"{self.directory} holds a different contract or different input "
                    "files; give the changed contract a new contract_id"
                )
            self.state["contract"] = contract.model_dump(mode="json")
        else:
            self.state = {
                "schema_version": 1,
                "contract_sha256": digest,
                "contract": contract.model_dump(mode="json"),
                "attempts": {"mlip": 0, "dft": 0},
                "steps": {},
            }
        self.state.setdefault("recovery", {})
        self._save()

    def _save(self) -> None:
        staged = self.state_path.with_name(f".{self.state_path.name}.tmp")
        staged.write_text(json.dumps(self.state, indent=2) + "\n")
        staged.replace(self.state_path)

    def attempts(self, fidelity: str) -> int:
        return int(self.state["attempts"][fidelity])

    def charge(self, fidelity: str) -> None:
        """Count a calculation against the budget before it starts."""
        self.state["attempts"][fidelity] += 1
        self._save()

    def record(self, step: RecipeStep, outcome: dict[str, Any]) -> None:
        self.state["steps"][step.step_id] = outcome
        self._save()

    def done(self, step_id: str) -> dict[str, Any] | None:
        outcome = self.state["steps"].get(step_id)
        return outcome if outcome is not None and outcome["status"] == "done" else None

    def pending(self, step_id: str) -> dict[str, Any] | None:
        """The next recovery attempt of a step that the budget stopped, if any."""
        return self.state["recovery"].get(step_id)

    def set_pending(self, step_id: str, recovery: dict[str, Any] | None) -> None:
        if recovery is None:
            self.state["recovery"].pop(step_id, None)
        else:
            self.state["recovery"][step_id] = recovery
        self._save()

    def record_attempt(
        self, step: RecipeStep, outcome: dict[str, Any], recovery: dict[str, Any] | None
    ) -> None:
        """Save an attempt's outcome and the step's next retry (None once terminal) together."""
        self.state["steps"][step.step_id] = outcome
        self.set_pending(step.step_id, recovery)

    def forget(self, step_id: str) -> None:
        self.state["steps"].pop(step_id, None)
        self._save()


def outstanding_steps(contract: CharacterizationContract) -> list[RecipeStep]:
    """Steps of the contract's recipe that have no completed outcome yet."""
    state_path = (
        Path(contract.output_root) / f"characterization-{contract.contract_id}" / STATE_FILE
    )
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    steps, recovery = state.get("steps", {}), state.get("recovery", {})
    return [
        step
        for step in plan_recipe(contract)
        if steps.get(step.step_id, {}).get("status") != "done" or step.step_id in recovery
    ]


def _references(contract: CharacterizationContract) -> ReferenceEnergySet:
    return ReferenceEnergySet.model_validate_json(
        Path(str(contract.reference_energies)).read_text()
    )


def _preflight(
    contract: CharacterizationContract,
) -> tuple[CharacterizationStatus, list[str]] | None:
    """Problems that make calculations pointless, found before any of them runs."""
    from ase.io import read

    expected = parse_composition(contract.formula)
    assert expected is not None
    wrong = []
    for path in contract.structures:
        try:
            atoms = read(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable file is a protocol error
            wrong.append(f"{path} is not a readable structure ({type(exc).__name__}: {exc})")
            continue
        found = parse_composition(atoms.get_chemical_formula(empirical=True))
        if found is None or found.formula != expected.formula:
            wrong.append(f"{path} is not {expected.formula}")
    if wrong:
        return CharacterizationStatus.INVALID_PROTOCOL, wrong
    if contract.recipe != "thermodynamic_stability":
        return None
    try:
        references = _references(contract)
    except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        return CharacterizationStatus.INVALID_PROTOCOL, [
            f"a valid reference-energy set: {contract.reference_energies} is not one "
            f"({type(exc).__name__}: {exc})"
        ]
    if references.method_signature != contract.methods.dft_method_signature:
        return CharacterizationStatus.INVALID_PROTOCOL, [
            f"references computed with the contract's DFT method: the reference set uses "
            f"{references.method_signature!r}, the relaxations "
            f"{contract.methods.dft_method_signature!r}"
        ]
    backend = contract.methods.dft.backend if contract.methods.dft is not None else None
    declared = [("the reference set", references.backend)]
    declared += [(f"competing phase {e.phase_id}", e.backend) for e in references.phase_entries]
    declared += [
        (f"elemental reference {e.phase_id}", e.backend)
        for e in [
            *references.elemental_entries.values(),
            *references.elemental_reference_candidates,
        ]
    ]
    foreign = [
        f"references computed with the contract's DFT backend: {source} uses "
        f"{value!r}, the relaxations {backend!r}"
        for source, value in declared
        if value is not None and value != backend
    ]
    if foreign:
        return CharacterizationStatus.INVALID_PROTOCOL, foreign
    from pymatgen.core import Composition as PMGComposition

    uncovered = []
    for phase_id, formula, _energy in references.competing_phase_records():
        try:
            elements = {str(element) for element in PMGComposition(formula).elements}
        except Exception:  # noqa: BLE001 - pymatgen raises several types for bad formulas
            elements = None
        missing = (
            sorted(elements - set(references.elemental_energies_eV_per_atom))
            if elements
            else [formula]
        )
        if missing:
            uncovered.append(
                f"elemental energies for {missing}, needed by competing phase {phase_id}"
            )
    if uncovered:
        return CharacterizationStatus.INVALID_PROTOCOL, uncovered
    completeness = references.audit_completeness(expected.elements)
    if completeness.provisional:
        # The hull would be provisional whatever the relaxations give: no DFT is spent.
        return CharacterizationStatus.INCONCLUSIVE, (
            [f"elemental reference: {e}" for e in completeness.missing_elemental_references]
            + [f"required phase: {f}" for f in completeness.missing_required_formulas]
            + [
                f"competing phase in the {pair} subsystem"
                for pair in completeness.missing_binary_subsystems
            ]
            + (["a ternary competing phase"] if completeness.missing_ternary_competitor else [])
        )
    return None


def run_characterization(
    contract: CharacterizationContract,
    *,
    relax: Relax = run_relaxation,
    ledger: EvidenceLedger | None = None,
) -> CharacterizationReport:
    """Run the outstanding steps of the contract's recipe and assess the evidence.

    Relaxations stop at the contract's budget and, for DFT, at its approval; the
    report then says which evidence is missing. ``relax`` runs one relaxation
    (the provenance-tracked relaxation workflow by default). Every relaxation,
    failed or not, and every conclusive assessment is appended to ``ledger``
    (by default ``evidence-ledger.jsonl`` in the run directory); the report
    cites the ledger's evidence identifiers.
    """
    with _Run(contract) as run:
        if ledger is None:
            ledger = EvidenceLedger(run.directory / LEDGER_FILE)
        return _execute(contract, run, relax, ledger)


def _method_record(
    contract: CharacterizationContract, fidelity: str, dft_settings: dict | None = None
) -> MethodRecord:
    if fidelity == "dft":
        dft = contract.methods.dft
        assert dft is not None
        directories = {"pseudo_dir": dft.pseudo_dir, "potcar_dir": dft.potcar_dir}
        return MethodRecord(
            backend=dft.backend,
            method_signature=contract.methods.dft_method_signature,
            settings={
                **(dft.settings if dft_settings is None else dft_settings),
                **{k: v for k, v in directories.items() if v},
            },
            software_versions=software_versions("ase"),
        )
    mlip = contract.methods.mlip
    return MethodRecord(
        backend=str(mlip.get("mlip_backend", "hydragnn")),
        model=next(
            (str(mlip[key]) for key in ("uma_model_name", "mace_model", "logdir") if mlip.get(key)),
            None,
        ),
        settings=dict(mlip),
        software_versions=software_versions("ase", "fairchem-core", "mace-torch"),
    )


def _relaxation_evidence(
    contract: CharacterizationContract, step: RecipeStep, outcome: dict[str, Any]
) -> EvidenceRecord:
    """The ledger record of one relaxation attempt: a result or a failure to compute it."""
    fidelity = step.fidelity or "mlip"
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    seconds = float(outcome.get("wall_seconds") or 0.0)
    failed = outcome["status"] != "done" or outcome.get("return_code") not in (None, 0)
    nodes, devices = int(outcome.get("nodes") or 0), int(outcome.get("devices") or 0)
    gpus, kind, hours = (
        float(outcome.get("gpus") or 0.0),
        outcome.get("device_kind"),
        seconds / 3600,
    )
    cost = (
        CostRecord(failed_jobs=1, failed_seconds=seconds)
        if failed
        else CostRecord(dft_calculations=1, dft_seconds=seconds)
        if fidelity == "dft"
        else CostRecord(mlip_relaxations=1, mlip_seconds=seconds)
    ) + CostRecord(
        nodes=nodes,
        node_hours=nodes * hours,
        devices={kind: devices} if kind and devices else {},
        device_hours={kind: devices * hours} if kind and devices else {},
        gpus=gpus,
        gpu_hours=gpus * hours,
    )
    diagnosis = {"failure_class": outcome["failure_class"]} if outcome.get("failure_class") else {}
    common: dict[str, Any] = {
        "kind": "relaxation",
        "subject": contract.formula,
        "characterization_contract_id": contract.contract_id,
        "calculation_id": step.step_id,
        "task_id": f"attempt-{outcome['attempt']}",
        "run_id": outcome.get("run_id"),
        "method": _method_record(contract, fidelity, outcome.get("dft_settings")),
        "input_sha256": hashes_of([outcome.get("input_structure") or step.structure]),
        "cost": cost,
        "physical_assumptions": [
            f"relax_atoms={contract.geometry.relax_atoms}, "
            f"relax_cell={contract.geometry.relax_cell}, "
            f"pressure_GPa={contract.geometry.pressure_GPa}"
        ],
    }
    if outcome["status"] != "done":
        return EvidenceRecord(
            **common,
            evidence_level=(
                EvidenceLevel.LOW_FIDELITY_DFT
                if fidelity == "dft"
                else EvidenceLevel.MLIP_RELAXATION
            ),
            outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
            failure_reason=outcome["failure_reason"],
            numerical_convergence=diagnosis,
        )
    from ase.io import read

    force = float(outcome["max_force_eV_per_A"])
    converged = bool(outcome["converged"]) and force <= tolerance
    n_atoms = len(read(outcome["optimized_structure_path"]))
    return EvidenceRecord(
        **common,
        evidence_level=EvidenceLevel(outcome["evidence_level"]),
        outcome=EvidenceOutcome.MEASURED if converged else EvidenceOutcome.NOT_CONVERGED,
        failure_reason=None
        if converged
        else outcome.get("failure_reason") or f"residual force above {tolerance} eV/A",
        values=[
            LedgerQuantity(name="total_energy", value=float(outcome["energy_eV"]), unit="eV"),
            LedgerQuantity(
                name="energy_per_atom", value=float(outcome["energy_eV"]) / n_atoms, unit="eV/atom"
            ),
            LedgerQuantity(name="residual_max_force", value=force, unit="eV/A"),
        ],
        numerical_convergence={
            "converged": converged,
            "max_force_eV_per_A": force,
            "force_tolerance_eV_per_A": tolerance,
            "steps": outcome.get("steps"),
            "max_steps": contract.tolerances.max_relaxation_steps,
            **diagnosis,
        },
        output_artifact_sha256={
            outcome["optimized_structure_path"]: outcome["optimized_structure_sha256"]
        },
    )


_ASSESSMENT_KIND = {
    "relaxation": "characterization",
    "energy_ranking": "energy_ranking",
    "thermodynamic_stability": "stability",
}


def _assessment_evidence(
    contract: CharacterizationContract, report: CharacterizationReport, run: _Run
) -> EvidenceRecord:
    """The ledger record of a sufficiently characterized answer, citing its relaxations."""
    relaxed = [
        outcome
        for step in plan_recipe(contract)
        if step.operator == "relax" and (outcome := run.done(step.step_id)) is not None
    ]
    stability = contract.recipe == "thermodynamic_stability"
    return EvidenceRecord(
        kind=_ASSESSMENT_KIND[contract.recipe],
        subject=contract.formula,
        characterization_contract_id=contract.contract_id,
        calculation_id=contract.recipe,
        method=_method_record(contract, "dft" if contract.needs_dft() else "mlip"),
        evidence_level=min(
            (EvidenceLevel(o["evidence_level"]) for o in relaxed), key=list(EvidenceLevel).index
        ),
        outcome=EvidenceOutcome.MEASURED,
        values=[
            LedgerQuantity(name=v.name, value=v.value, unit=v.unit, structure=v.structure)
            for v in report.observables
            if v.unit != "structure"
        ],
        numerical_convergence={"characterization_status": report.status.value},
        reference_set_id=_references(contract).identifier if stability else None,
        reference_complete=True if stability else None,
        input_sha256=hashes_of([contract.reference_energies]) if stability else {},
        physical_assumptions=report.limitations,
        parent_record_ids=[o["evidence_id"] for o in relaxed],
    )


def _attempt(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    relax: Relax,
    ledger: EvidenceLedger,
    structure: str,
    dft: DFTBackendConfig | None,
) -> dict[str, Any]:
    fidelity = step.fidelity or "mlip"
    run.charge(fidelity)
    attempt = run.attempts(fidelity)
    work_dir = run.directory / step.step_id
    if attempt > 1:
        work_dir = work_dir / f"attempt-{attempt}"
    started = time.monotonic()

    def finished(outcome: dict[str, Any]) -> dict[str, Any]:
        outcome = {
            **outcome,
            "structure": step.structure,
            "attempt": attempt,
            "wall_seconds": time.monotonic() - started,
        }
        if structure != step.structure:
            outcome["input_structure"] = structure
        if dft is not None and dft != contract.methods.dft:
            outcome["dft_settings"] = dft.settings
        return outcome

    try:
        outcome = _relax_record(
            relax(_relaxation_config(contract, step, work_dir, structure=structure, dft=dft))
        )
    except Exception as exc:  # noqa: BLE001 - a failed tool is not evidence
        outcome = {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_type": type(exc).__name__,
        }
    except BaseException as exc:
        # The charged attempt's cost is recorded; the step and its saved retry are kept.
        interrupted = {
            "status": "failed",
            "failure_reason": f"interrupted by {type(exc).__name__}",
            "failure_type": type(exc).__name__,
        }
        ledger.append(_relaxation_evidence(contract, step, finished(interrupted)))
        raise
    outcome = finished(outcome)
    # QE's ionic convergence message can precede a failed final SCF.
    if outcome["status"] == "done" and outcome.get("scf_converged") is False:
        outcome["converged"] = False
        outcome["evidence_level"] = EvidenceLevel.LOW_FIDELITY_DFT.value
        outcome["failure_reason"] = outcome.get("failure_reason") or "SCF did not converge"
    # A run killed after its last ionic step (e.g. a timeout) can leave finite results.
    if outcome["status"] == "done" and outcome.get("return_code") not in (None, 0):
        outcome["converged"] = False
        outcome["evidence_level"] = EvidenceLevel.LOW_FIDELITY_DFT.value
        outcome["failure_reason"] = (
            outcome.get("failure_reason") or f"exited with return code {outcome['return_code']}"
        )
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    if not (
        outcome["status"] == "done"
        and outcome["converged"]
        and float(outcome["max_force_eV_per_A"]) <= tolerance
    ):
        outcome["failure_class"] = classify_failure(
            outcome, max_steps=contract.tolerances.max_relaxation_steps
        ).value
    return outcome


def _recovery_decision(
    contract: CharacterizationContract,
    step: RecipeStep,
    *,
    attempt: int,
    evidence_id: str,
    failure: FailureClass,
    action: str,
    rationale: str,
    used: int,
    budget: int | None,
) -> DecisionRecord:
    rule = DEFAULT_POLICY.rule(failure)
    return DecisionRecord(
        campaign_id=contract.contract_id,
        iteration=attempt,
        action=f"{action}:{failure.value}",
        subject=step.step_id,
        evidence_ids=[evidence_id],
        rationale=rationale,
        actor=f"recovery-policy:{DEFAULT_POLICY.version}",
        policy_version=DEFAULT_POLICY.version,
        thresholds={
            "max_retries": float(rule.max_retries),
            "retries_used": float(used),
            **({"dft_budget": float(budget)} if budget is not None else {}),
        },
        expected_cost=CostRecord(dft_calculations=1) if action == "recover" else None,
    )


def _resume_recovery(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    ledger: EvidenceLedger,
    pending: dict[str, Any],
    budget: int | None,
) -> None:
    """Record the decision to run a saved retry; a decision already recorded is not repeated."""
    decision = pending.get("decision")
    if decision is None:
        failure = FailureClass(pending["failure"])
        rule = DEFAULT_POLICY.rule(failure)
        used = pending["retries"][failure.value] - 1
        decision = _recovery_decision(
            contract,
            step,
            attempt=pending["attempt"],
            evidence_id=pending["evidence_id"],
            failure=failure,
            action="recover",
            rationale=f"retry {used + 1} of {rule.max_retries}, deferred by the budget: "
            f"{rule.action}",
            used=used,
            budget=budget,
        ).model_dump(mode="json", exclude={"decision_id", "recorded_at"})
        run.set_pending(step.step_id, {**pending, "decision": decision})
    ledger.append(DecisionRecord.model_validate(decision))


def _relax_with_recovery(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    relax: Relax,
    ledger: EvidenceLedger,
    budget: int | None,
) -> None:
    """Relax one step; a failed DFT attempt is retried as the recovery policy directs.

    Every retry is saved before it runs and kept until an outcome without a further
    retry is saved, so an interrupted or budget-stopped retry resumes with its
    structure, settings, and counters.
    """
    structure = str(step.structure)
    dft = contract.methods.dft if step.fidelity == "dft" else None
    retries: dict[FailureClass, int] = {}
    if (pending := run.pending(step.step_id)) is not None:
        structure = pending["structure"]
        dft = DFTBackendConfig.model_validate(pending["dft"])
        retries = {FailureClass(k): v for k, v in pending["retries"].items()}
        _resume_recovery(contract, step, run, ledger, pending, budget)
    while True:
        outcome = _attempt(contract, step, run, relax, ledger, structure, dft)
        evidence = ledger.append(_relaxation_evidence(contract, step, outcome))
        evidence_id = evidence.evidence_id  # type: ignore[union-attr]
        recorded = {**outcome, "evidence_id": evidence_id}
        if dft is None or "failure_class" not in outcome:
            run.record_attempt(step, recorded, None)
            return
        failure = FailureClass(outcome["failure_class"])
        rule = DEFAULT_POLICY.rule(failure)
        used = retries.get(failure, 0)
        within_budget = budget is None or run.attempts("dft") < budget
        allowed = can_retry(failure, timeout_sec=dft.timeout_sec) and used < rule.max_retries
        retry = allowed and within_budget
        if retry:
            rationale = f"retry {used + 1} of {rule.max_retries}: {rule.action}"
        elif not can_retry(failure, timeout_sec=dft.timeout_sec) or rule.max_retries == 0:
            rationale = f"no recovery for {failure.value}"
        elif not within_budget:
            rationale = f"the DFT budget ({budget}) is spent; the retry is saved"
        else:
            rationale = f"{rule.max_retries} retries for {failure.value} exhausted"
        decision = _recovery_decision(
            contract,
            step,
            attempt=outcome["attempt"],
            evidence_id=evidence_id,
            failure=failure,
            action="recover" if retry else "defer" if allowed else "give_up",
            rationale=rationale,
            used=used,
            budget=budget,
        )
        checkpoint = None
        if allowed:
            retries[failure] = used + 1
            if failure == FailureClass.IONIC_STEP_LIMIT and outcome.get("optimized_structure_path"):
                structure = outcome["optimized_structure_path"]
            settings, timeout = adjusted_settings(
                failure, dft.backend, dft.settings, dft.timeout_sec
            )
            dft = dft.model_copy(update={"settings": settings, "timeout_sec": timeout})
            checkpoint = {
                "structure": structure,
                **(
                    {"structure_sha256": sha256_file(structure)}
                    if structure != step.structure
                    else {}
                ),
                "dft": dft.model_dump(mode="json"),
                "retries": {k.value: v for k, v in retries.items()},
                "attempt": outcome["attempt"],
                "evidence_id": evidence_id,
                "failure": failure.value,
                "decision": decision.model_dump(mode="json", exclude={"decision_id", "recorded_at"})
                if retry
                else None,
            }
        run.record_attempt(step, recorded, checkpoint)
        ledger.append(decision)
        if not retry:
            return


def _execute(
    contract: CharacterizationContract, run: _Run, relax: Relax, ledger: EvidenceLedger
) -> CharacterizationReport:
    problem = _preflight(contract)
    if problem is not None:
        status, missing = problem
        report = CharacterizationReport(
            contract_id=contract.contract_id,
            contract_sha256=contract.digest(),
            recipe=contract.recipe,
            status=status,
            missing=missing,
            limitations=["No calculation was run: the contract cannot be answered as given."],
            recommended_next_test=(
                "Compute the missing references with the contract's DFT method."
                if status == CharacterizationStatus.INCONCLUSIVE
                else "Correct the contract (structures, references, or DFT method)."
            ),
        )
        (run.directory / REPORT_FILE).write_text(report.model_dump_json(indent=2) + "\n")
        return report
    steps = plan_recipe(contract)
    _verify_completed_artifacts(run, steps)
    blockers: list[str] = []
    budget = {
        "mlip": contract.budget.max_mlip_relaxations,
        "dft": contract.budget.max_dft_calculations,
    }
    final = steps[-1] if steps[-1].operator != "relax" else None
    for step in steps:
        if step.operator != "relax":
            continue
        fidelity = step.fidelity or "mlip"
        recovering = run.pending(step.step_id) is not None
        if run.done(step.step_id) is not None and not recovering:
            continue
        if step.needs_dft_approval and not contract.dft_approved:
            blockers.append(f"{step.step_id}: DFT is required but not approved (dft_approved)")
            continue
        if budget[fidelity] is not None and run.attempts(fidelity) >= budget[fidelity]:
            blockers.append(
                f"{step.step_id}: beyond the {fidelity} relaxation budget "
                f"({run.attempts(fidelity)} attempted)"
            )
            continue
        if recovering and final is not None:
            run.forget(final.step_id)
        _relax_with_recovery(contract, step, run, relax, ledger, budget[fidelity])
    report_path = run.directory / REPORT_FILE
    if final is not None and run.done(final.step_id) is not None and report_path.is_file():
        return CharacterizationReport.model_validate_json(report_path.read_text())
    report = _assess(contract, steps, run, blockers)
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    inputs_done = all(run.done(step.step_id) for step in steps if step.operator == "relax")
    if report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED:
        ledger.append(_assessment_evidence(contract, report, run))
    if final is not None and inputs_done:
        # Assessed from complete inputs, whatever the answer: not repeated on resume.
        run.record(final, {"status": "done", "characterization_status": report.status.value})
    return report


def _verify_completed_artifacts(run: _Run, steps: list[RecipeStep]) -> None:
    """Refuse to resume from relaxation outputs, completed or saved for a retry, that changed."""
    changed = []
    for step in steps:
        pending = run.pending(step.step_id) if step.operator == "relax" else None
        if pending is not None and pending["structure"] != step.structure:
            path = Path(pending["structure"])
            recorded = pending.get("structure_sha256")
            if not path.is_file():
                changed.append(f"{step.step_id}: retry structure {path} no longer exists")
            elif recorded is None or sha256_file(path) != recorded:
                changed.append(f"{step.step_id}: retry structure {path} has changed")
        outcome = run.done(step.step_id) if step.operator == "relax" else None
        if outcome is None:
            continue
        path = Path(outcome["optimized_structure_path"])
        recorded = outcome.get("optimized_structure_sha256")
        if not path.is_file():
            changed.append(f"{step.step_id}: {path} no longer exists")
        elif recorded is None or sha256_file(path) != recorded:
            changed.append(f"{step.step_id}: {path} differs from the relaxation's output")
    if changed:
        raise ValueError(
            f"{run.directory} cannot be resumed; completed relaxation outputs changed: "
            + "; ".join(changed)
            + ". Give the contract a new contract_id to recompute them."
        )


# --------------------------------------------------------------------------- #
# Assessment                                                                  #
# --------------------------------------------------------------------------- #


def _evidence_item(contract: CharacterizationContract, step_id: str, outcome: dict) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=outcome["evidence_id"],
        kind="relaxation",
        evidence_level=EvidenceLevel(outcome["evidence_level"]),
        method=outcome.get("run_id", ""),
        artifact=outcome.get("optimized_structure_path"),
        artifact_sha256=outcome.get("optimized_structure_sha256"),
        converged=outcome["converged"],
    )


def _relaxation_results(relaxed: list[tuple[str, dict]]) -> list[RelaxationResult]:
    return [
        RelaxationResult(
            structure_path=outcome["structure"],
            optimized_structure_path=outcome["optimized_structure_path"],
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=outcome["energy_eV"],
            final_max_force_eV_per_A=outcome["max_force_eV_per_A"],
            num_steps=outcome["steps"],
            converged=outcome["converged"],
        )
        for _step_id, outcome in relaxed
    ]


def _assess(
    contract: CharacterizationContract,
    steps: list[RecipeStep],
    run: _Run,
    blockers: list[str],
) -> CharacterizationReport:
    relax_steps = [step for step in steps if step.operator == "relax"]
    outcomes = {step.step_id: run.state["steps"].get(step.step_id) for step in relax_steps}
    relaxed = [
        (step_id, outcome)
        for step_id, outcome in outcomes.items()
        if outcome is not None and outcome["status"] == "done"
    ]
    evidence = [_evidence_item(contract, step_id, outcome) for step_id, outcome in relaxed]
    base = {
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest(),
        "recipe": contract.recipe,
        "evidence": evidence,
        "methods": {
            "relaxation": "dft" if contract.needs_dft() else "mlip",
            **(
                {"dft_method_signature": contract.methods.dft_method_signature}
                if contract.methods.dft_method_signature
                else {}
            ),
        },
    }
    failed = [
        f"{step_id}: {outcome['failure_reason']}"
        for step_id, outcome in outcomes.items()
        if outcome is not None and outcome["status"] == "failed"
    ]
    if failed:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.FAILED_INFRASTRUCTURE,
            missing=failed + blockers,
            limitations=["A calculation could not run; this is not evidence about the material."],
            recommended_next_test="Fix the failed calculations and run the contract again "
            "(completed steps are kept).",
        )
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    observables = []
    for _step_id, outcome in relaxed:
        level = EvidenceLevel(outcome["evidence_level"])
        cite = [outcome["evidence_id"]]
        structure = outcome["structure"]
        observables += [
            ObservableValue(
                name="relaxed_structure",
                value=str(outcome["optimized_structure_path"]),
                unit="structure",
                evidence_level=level,
                structure=structure,
                evidence_ids=cite,
            ),
            ObservableValue(
                name="residual_max_force",
                value=float(outcome["max_force_eV_per_A"]),
                unit="eV/A",
                evidence_level=level,
                structure=structure,
                evidence_ids=cite,
            ),
        ]
    unconverged = [
        step_id
        for step_id, outcome in relaxed
        if not outcome["converged"] or outcome["max_force_eV_per_A"] > tolerance
    ]
    missing = blockers + [
        f"{step_id}: not converged to {tolerance} eV/A (numerical; not evidence against "
        "the material)"
        for step_id in unconverged
    ]
    if missing:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=missing,
            recommended_next_test=(
                "Approve the required DFT calculations"
                if any("not approved" in item for item in blockers)
                else "Complete the missing relaxations (more steps or budget) and run the "
                "contract again"
            ),
        )
    level = _level(relaxed)
    below = [
        f"{r.name} at {r.minimum_evidence_level.value} (the evidence is {level.value})"
        for r in contract.observables
        if not at_least(level, r.minimum_evidence_level)
    ]
    if below:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=below,
            recommended_next_test="Relax the structures at the required fidelity.",
        )
    return {
        "relaxation": _assess_relaxation,
        "energy_ranking": _assess_ranking,
        "thermodynamic_stability": _assess_stability,
    }[contract.recipe](contract, relaxed, observables, base)


def _energy_observables(
    contract: CharacterizationContract, relaxed: list[tuple[str, dict]]
) -> list[ObservableValue]:
    from ase.io import read

    values = []
    for _step_id, outcome in relaxed:
        n_atoms = len(read(outcome["optimized_structure_path"]))
        values.append(
            ObservableValue(
                name="energy_per_atom",
                value=float(outcome["energy_eV"]) / n_atoms,
                unit="eV/atom",
                evidence_level=EvidenceLevel(outcome["evidence_level"]),
                structure=outcome["structure"],
                evidence_ids=[outcome["evidence_id"]],
            )
        )
    return values


def _level(relaxed: list[tuple[str, dict]]) -> EvidenceLevel:
    return min(
        (EvidenceLevel(outcome["evidence_level"]) for _step, outcome in relaxed),
        key=list(EvidenceLevel).index,
    )


def _assess_relaxation(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=f"All {len(relaxed)} structure(s) of {contract.formula} relaxed to residual "
        f"forces at most {contract.tolerances.force_tolerance_eV_per_A} eV/A ({level.value}).",
        observables=observables + _energy_observables(contract, relaxed),
        limitations=[
            "A converged relaxation is a local minimum of the method's energy; it says "
            "nothing about stability against other structures or compositions."
        ],
    )


def _assess_ranking(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    report = score_stability(
        contract.formula,
        _relaxation_results(relaxed),
        force_tol_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
        degeneracy_tol_eV_per_atom=contract.tolerances.degeneracy_tolerance_eV_per_atom,
        evidence_level=level,
    )
    observables = (
        observables
        + _energy_observables(contract, relaxed)
        + _ranking_values(contract, relaxed, report)
    )
    limitations = [
        "A same-composition ranking; it does not establish formation energy, convex-hull "
        "stability, or synthesizability."
    ]
    if report.near_degenerate_structure_paths:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=[
                f"a ground state distinguishable within "
                f"{contract.tolerances.degeneracy_tolerance_eV_per_atom} eV/atom: "
                f"{len(report.near_degenerate_structure_paths)} structure(s) are as low"
            ],
            limitations=limitations,
            recommended_next_test="Rank the near-degenerate structures at higher fidelity.",
        )
    ground = report.ground_state
    gap = report.ranking[1].delta_e_above_min_eV_per_atom if len(report.ranking) > 1 else None
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=f"Lowest-energy structure of {contract.formula} among "
        f"{len(report.ranking)} ranked: {ground.structure_path}"
        + (f", {gap:.4g} eV/atom below the next" if gap is not None else "")
        + f" ({level.value}).",
        observables=observables,
        limitations=limitations,
    )


def _ranking_values(
    contract: CharacterizationContract, relaxed: list[tuple[str, dict]], report: StabilityReport
) -> list[ObservableValue]:
    evidence_by_input = {outcome["structure"]: outcome["evidence_id"] for _s, outcome in relaxed}
    return [
        ObservableValue(
            name="relative_energy_per_atom",
            value=phase.delta_e_above_min_eV_per_atom,
            unit="eV/atom",
            evidence_level=report.evidence_level,
            structure=phase.structure_path,
            evidence_ids=[evidence_by_input[phase.structure_path]],
        )
        for phase in report.ranking
    ]


def _assess_stability(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    limitations = [
        "Stability is relative to the declared reference phases (the lowest among those "
        "tested, not a global ground-state search), at 0 K without zero-point or "
        "finite-temperature contributions."
    ]
    if level not in DFT_EVIDENCE or not at_least(level, EvidenceLevel.CONVERGED_DFT):
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=[f"converged DFT energies (the evidence is {level.value})"],
            limitations=limitations,
            recommended_next_test="Relax the structures with converged DFT.",
        )
    references = _references(contract)
    report = score_stability(
        contract.formula,
        _relaxation_results(relaxed),
        force_tol_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
        degeneracy_tol_eV_per_atom=contract.tolerances.degeneracy_tolerance_eV_per_atom,
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_energies=references,
        method_signature=contract.methods.dft_method_signature,
        evidence_level=level,
    )
    ground = report.ground_state
    cite = [
        outcome["evidence_id"]
        for _step_id, outcome in relaxed
        if outcome["structure"] == ground.structure_path
    ]
    observables = (
        observables
        + _energy_observables(contract, relaxed)
        + [
            ObservableValue(
                name=name,
                value=float(value),
                unit="eV/atom",
                evidence_level=level,
                structure=ground.structure_path,
                evidence_ids=cite,
            )
            for name, value in (
                ("formation_energy_per_atom", ground.formation_energy_eV_per_atom),
                ("energy_above_hull", ground.energy_above_hull_eV_per_atom),
            )
        ]
    )
    e_hull = float(ground.energy_above_hull_eV_per_atom)
    on_hull = e_hull <= contract.tolerances.hull_tolerance_eV_per_atom + 1e-8
    decomposition = ", ".join(sorted(ground.decomposition))
    conclusion = (
        f"{contract.formula} ({ground.structure_path}) is on the convex hull of the declared "
        f"reference phases within {contract.tolerances.hull_tolerance_eV_per_atom} eV/atom "
        f"({e_hull:.4g} eV/atom above it; {level.value}, "
        f"{references.method_signature})."
        if on_hull
        else f"{contract.formula} ({ground.structure_path}) lies {e_hull:.4g} eV/atom above "
        f"the convex hull of the declared reference phases and decomposes to "
        f"{decomposition} ({level.value}, {references.method_signature})."
    )
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=conclusion,
        observables=observables,
        limitations=limitations,
    )


def load_contract(path: str | Path) -> CharacterizationContract:
    """Read a YAML or JSON contract; relative paths in it are relative to the file."""
    import yaml

    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a characterization contract mapping")

    def resolve(value: Any) -> Any:
        return str(path.parent / value) if isinstance(value, str) else value

    data["structures"] = [resolve(item) for item in data.get("structures") or []]
    data.setdefault("output_root", "runs")
    for key in ("reference_energies", "output_root"):
        if key in data:
            data[key] = resolve(data[key])
    dft = (data.get("methods") or {}).get("dft")
    if isinstance(dft, dict):
        for key in ("pseudo_dir", "potcar_dir"):
            if dft.get(key):
                dft[key] = resolve(dft[key])
    return CharacterizationContract.model_validate(data)


__all__ = [
    "CharacterizationContract",
    "CharacterizationMethods",
    "NumericalTolerances",
    "RecipeStep",
    "load_contract",
    "outstanding_steps",
    "plan_recipe",
    "run_characterization",
]
