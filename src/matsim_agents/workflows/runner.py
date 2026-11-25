"""Run any maintained config file through the Python API and record what happened.

Every file kind has a preflight (static validation plus blocker detection) and an executor
that calls the native workflow. The CLI and agents are thin callers of this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from matsim_agents.execution.capabilities import check, get_capability
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.execution.variables import unset_variables
from matsim_agents.workflows.run_records import (
    RunRecord,
    RunState,
    find_run,
    load_record,
    start_run,
    write_record,
)


class UnrunnableInput(ValueError):
    """The file is an auxiliary input (policy, model config, manifest), not a run request."""


@dataclass(frozen=True)
class RunOptions:
    """Caller decisions that a config file must not make for itself."""

    mlip_config: Path | None = None
    allow_template: bool = False
    dispatch: bool = False
    approved_by: str | None = None


@dataclass
class Preflight:
    summary: dict[str, Any]
    blockers: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    state: RunState
    native_run_directory: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None


@dataclass(frozen=True)
class Handler:
    preflight: Callable[[Path, RunOptions], Preflight]
    execute: Callable[[Path, ScientificRunDirectory, RunOptions], Outcome]
    resume: Callable[[RunRecord], Outcome] | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: malformed YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return raw


def load_workflow_config(path: Path) -> dict[str, Any]:
    """Read a workflow config; its resources: block gets ${VAR} expansion."""
    from matsim_agents.execution.placement import ResourcePlan

    raw = load_yaml(path)
    # Saved resolved configs carry "resources": null once their servers are resolved.
    if raw.get("resources") is not None:
        plan = ResourcePlan.from_mapping(raw, source=path)
        raw = {key: value for key, value in raw.items() if key != "vars"}
        raw["resources"] = plan
    return raw


def backend_blockers(names: Iterable[str]) -> list[str]:
    """Why each named backend cannot run here, from the capability registry."""
    blockers = []
    for name in dict.fromkeys(names):
        try:
            result = check(get_capability(name))
        except KeyError:
            continue
        if not result.available:
            blockers.append(f"backend {name} unavailable: missing {', '.join(result.missing)}")
    return blockers


_WORKFLOW_STATE: dict[str, RunState] = {
    "complete": "complete",
    "partial": "partial",
    "failed": "failed",
    "rejected": "failed",
    "planned": "blocked",
    "running": "interrupted",
}


# --- relaxation -------------------------------------------------------------------------


def _relax_config(path: Path):
    from matsim_agents.backends.mlip.config import resolve_mlip_paths
    from matsim_agents.workflows.relaxation import ScientificRelaxationConfig

    raw = load_yaml(path)
    if isinstance(raw.get("mlip"), dict):
        resolve_mlip_paths(raw["mlip"], path.resolve().parent)
    return ScientificRelaxationConfig.model_validate(raw)


def relaxation_blockers(cfg) -> list[str]:
    """Missing structure, backend, or approval for a relaxation config."""
    blockers = []
    if not Path(cfg.structure_path).is_file():
        blockers.append(f"structure_path {cfg.structure_path} does not exist")
    if cfg.mlip is not None:
        blockers += backend_blockers([cfg.mlip.backend])
    if cfg.dft is not None:
        blockers += backend_blockers([cfg.dft.backend])
        if cfg.approvals.before_dft and not cfg.dft_approved:
            blockers.append("DFT needs approval: set dft_approved: true after review")
    return blockers


def _relax_preflight(path: Path, options: RunOptions) -> Preflight:
    cfg = _relax_config(path)
    return Preflight(
        {"mode": cfg.mode.value, "structure_path": cfg.structure_path},
        relaxation_blockers(cfg),
    )


def _relax_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.relaxation import run_relaxation

    result = run_relaxation(_relax_config(path))
    return Outcome(
        _WORKFLOW_STATE[result.status.value],
        result.run_directory,
        {
            "mode": result.mode.value,
            "final_structure_path": result.final_structure_path,
            "stages": [
                {"stage": s.stage, "converged": s.converged, "energy_eV": s.energy_eV}
                for s in result.stages
            ],
        },
        result.failure_reason,
    )


# --- debate and LLM check ---------------------------------------------------------------


def _debate_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.workflows.debate import ScientificDebateConfig

    cfg = ScientificDebateConfig.model_validate(load_workflow_config(path))
    providers = [p.provider for p in cfg.participants if p.provider]
    return Preflight(
        {"hypothesis": cfg.hypothesis, "rounds": cfg.rounds}, backend_blockers(providers)
    )


def _debate_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.debate import ScientificDebateConfig, run_scientific_debate

    result = run_scientific_debate(
        ScientificDebateConfig.model_validate(load_workflow_config(path))
    )
    return Outcome(
        _WORKFLOW_STATE[result.status.value],
        result.run_directory,
        {"rounds_completed": result.rounds_completed, "verdicts": len(result.verdicts)},
    )


def _llm_check_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.workflows.llm_check import LLMCheckConfig

    cfg = LLMCheckConfig.model_validate(load_workflow_config(path))
    return Preflight(
        {"provider": cfg.provider, "model": cfg.model}, backend_blockers([cfg.provider])
    )


def _llm_check_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    result = run_llm_check(LLMCheckConfig.model_validate(load_workflow_config(path)))
    return Outcome(
        _WORKFLOW_STATE[result.status.value],
        result.run_directory,
        {"provider": result.provider, "model": result.model, "stages": result.stages},
        result.failure_reason,
    )


# --- characterization -------------------------------------------------------------------


def _characterize_preflight(path: Path, options: RunOptions) -> Preflight:
    from pydantic import ValidationError

    from matsim_agents.workflows.characterization import load_contract

    try:
        contract = load_contract(path)
    except ValidationError as error:
        # A user-supplied reference file that is absent is a blocker; any other error is invalid.
        if all("needs a reference_energies file" in item["msg"] for item in error.errors()):
            reference = load_yaml(path).get("reference_energies")
            return Preflight(
                {"contract_id": load_yaml(path).get("contract_id")},
                [f"reference_energies file {reference} does not exist; supply it (see docs)"],
            )
        raise
    return Preflight({"contract_id": contract.contract_id, "recipe": contract.recipe})


def _characterize_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.characterization import load_contract, run_characterization

    report = run_characterization(load_contract(path))
    failed = report.status in {"invalid_protocol", "failed_infrastructure"}
    return Outcome(
        "failed" if failed else ("complete" if report.status == "complete" else "partial"),
        None,
        {"status": report.status, "conclusion": report.conclusion, "missing": list(report.missing)},
        report.conclusion if failed else None,
    )


# --- exploration ------------------------------------------------------------------------


def _explore_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.workflows.exploration import api

    cfg = api.load_exploration_config(path)
    result = api.check_exploration(cfg)
    return Preflight({"formula": cfg.formula}, [] if result.ready else list(result.errors))


def _explore_outcome(summary) -> Outcome:
    return Outcome(
        "complete" if summary.status == "completed" else "failed",
        str(summary.run_directory),
        {"outcome": summary.outcome, "counts": dict(summary.counts)},
        "; ".join(summary.errors) or None,
    )


def _explore_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.exploration import api

    return _explore_outcome(api.run_exploration(api.load_exploration_config(path)))


def _explore_resume(record: RunRecord) -> Outcome:
    from matsim_agents.workflows.exploration import api

    if not record.native_run_directory:
        raise ValueError("the run has no exploration directory to resume")
    return _explore_outcome(api.resume_exploration(record.native_run_directory))


# --- structure generation ---------------------------------------------------------------


def _alloy_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.science.structures.alloys import AlloySpec

    spec = AlloySpec.model_validate(load_yaml(path))
    return Preflight(
        {"crystal_structure": spec.crystal_structure, "counts": dict(spec.counts or {})}
    )


def _alloy_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from ase.io import write

    from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

    atoms = generate_alloy(AlloySpec.model_validate(load_yaml(path)))
    target = run.path / "structure.extxyz"
    write(target, atoms)
    formula = atoms.get_chemical_formula()
    return Outcome(
        "complete",
        str(run.path),
        {"structure": str(target), "formula": formula, "atoms": len(atoms)},
    )


# --- active learning --------------------------------------------------------------------

TEMPLATE_ENV = "MATSIM_ALLOW_TEMPLATE_CONFIG"
REPO = Path(__file__).resolve().parents[3]


def _mlip_override(options: RunOptions):
    if options.mlip_config is None:
        return None
    from matsim_agents.backends.mlip.config import load_mlip_config

    return load_mlip_config(options.mlip_config)


def _al_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.workflows.active_learning import ALConfig
    from matsim_agents.workflows.active_learning.campaign_init import is_repo_template

    cfg = ALConfig.from_yaml(path, mlip=_mlip_override(options))
    blockers = backend_blockers([cfg.mlip.backend, cfg.dft.backend])
    if is_repo_template(path, REPO) and not (
        options.allow_template or os.environ.get(TEMPLATE_ENV) == "1"
    ):
        blockers.append(
            f"{path} is a repository example, not your campaign; create one with "
            f"`matsim-agents al init-campaign` or opt in with {TEMPLATE_ENV}=1"
        )
    return Preflight({"mlip": cfg.mlip.backend, "dft": cfg.dft.backend}, blockers)


def _al_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.active_learning import ALConfig
    from matsim_agents.workflows.active_learning.loop import run_active_learning

    run_active_learning(ALConfig.from_yaml(path, mlip=_mlip_override(options)))
    return Outcome("complete", None, {"note": "see the active-learning run directory"})


# --- NEB --------------------------------------------------------------------------------

_NEB_KEYS = {"parent", "defect", "lattice_parameter", "qe", "images", "method_id"}


def _neb_inputs(path: Path):
    from matsim_agents.backends.dft.qe_neb import QENebConfig
    from matsim_agents.science.defects.point_defects import generate_vacancy
    from matsim_agents.science.neb.endpoints import generate_vacancy_hop, vacancy_neighbors
    from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

    data = load_yaml(path)
    if unknown := set(data) - _NEB_KEYS:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    parent = generate_alloy(AlloySpec(**data["parent"]))
    lattice = float(data["lattice_parameter"])
    defect = data["defect"]
    if defect["mechanism"] != "vacancy_hop":
        raise ValueError(f"unsupported mechanism {defect['mechanism']!r}; build the pair in Python")
    initial, defect_config = generate_vacancy(parent, "parent", int(defect["site"]))
    neighbors = vacancy_neighbors(initial, defect_config, lattice)
    if not neighbors:
        raise ValueError("the vacancy has no nearest neighbours")
    mover = int(defect.get("mover", neighbors[0]))
    pair = generate_vacancy_hop(initial, defect_config, parent, mover, lattice)
    return data, pair, QENebConfig(**data["qe"])


def _neb_preflight(path: Path, options: RunOptions) -> Preflight:
    data, pair, _ = _neb_inputs(path)
    blockers = []
    if options.dispatch:
        blockers += backend_blockers(["qe"])
    return Preflight(
        {"mechanism": data["defect"]["mechanism"], "dispatch": options.dispatch}, blockers
    )


def _neb_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.neb import run_neb_workflow

    data, pair, qe = _neb_inputs(path)
    images = {
        k: float(v) for k, v in data.get("images", {}).items() if k in ("min_distance", "max_step")
    }
    outcome = run_neb_workflow(
        pair,
        qe,
        run.path / "neb",
        dispatch=options.dispatch,
        method_id=data.get("method_id"),
        **images,
    )
    plan = outcome.plan
    if plan.status == "rejected":
        return Outcome("failed", str(run.path / "neb"), failure_reason="; ".join(plan.reasons))
    summary = {
        "pathway_id": plan.pathway_id,
        "dry_run": plan.dry_run,
        "input": str(plan.input_path),
    }
    return Outcome("complete", str(run.path / "neb"), summary)


# --- external tools ---------------------------------------------------------------------


def _external_campaign(path: Path):
    from matsim_agents.backends.external.spectra_pka.config import load_config_file

    parsed = load_config_file(path)
    if parsed.campaign is None:
        raise ValueError(f"{path} has no 'campaign' section")
    return parsed


def _external_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.backends.external.spectra_pka.availability import check_availability

    parsed = _external_campaign(path)
    result = check_availability(parsed.external_tools.spectra_pka, parsed.campaign, invoke=False)
    blockers = (
        []
        if result.status.value == "READY"
        else [
            f"SPECTRA-PKA {result.status.value}: "
            + "; ".join(f"{c.name} ({c.detail})" for c in result.checks if c.ok is False)
        ]
    )
    return Preflight(
        {"tool": "spectra-pka", "status": result.status.value}, blockers if options.dispatch else []
    )


def _external_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.backends.external.spectra_pka import adapter

    parsed = _external_campaign(path)
    result = adapter.run(
        parsed.external_tools.spectra_pka, parsed.campaign, execute=options.dispatch
    )
    failed = result.status in {"EXECUTION_FAILED", "PARSE_FAILED", "STRUCTURALLY_INVALID"}
    return Outcome(
        "failed" if failed else "complete",
        str(result.run_dir),
        {
            "status": result.status,
            "command": list(result.command),
            "numerical_validation": "NOT_VALIDATED",
        },
        result.message or None if failed else None,
    )


# --- application ------------------------------------------------------------------------


def _application_preflight(path: Path, options: RunOptions) -> Preflight:
    from matsim_agents.workflows import application_runner

    return application_runner.preflight(path, options)


def _application_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows import application_runner

    return application_runner.execute(path, run, options)


# --- investigation ----------------------------------------------------------------------

INVESTIGATION_SEED = 0


def _investigation_config(path: Path):
    from matsim_agents.workflows.investigation import InvestigationConfig

    return InvestigationConfig.model_validate(load_workflow_config(path))


def _investigation_blockers(cfg, options: RunOptions) -> list[str]:
    discussion = cfg.hypothesis_discussion
    if discussion is None:
        return ["configure hypothesis_discussion: no model proposes the hypothesis"]
    if discussion.mode == "single_llm":
        providers = [discussion.single_llm.provider]
    else:
        providers = [p.provider for p in discussion.multi_llm_debate.participants if p.provider]
    blockers = backend_blockers(providers)
    policy = cfg.phase_policy
    if policy.relax_structures and options.mlip_config is None:
        blockers.append("relax_structures needs an MLIP: pass --mlip-config")
    if policy.active_learning:
        blockers.append(
            "active_learning inside an investigation needs a dispatching runner; "
            "run an active-learning config with `matsim-agents run` instead"
        )
    return blockers


def _investigation_preflight(path: Path, options: RunOptions) -> Preflight:
    cfg = _investigation_config(path)
    discussion = cfg.hypothesis_discussion
    return Preflight(
        {
            "objective": cfg.objective,
            "discussion": discussion.mode if discussion else None,
            "relax_structures": cfg.phase_policy.relax_structures,
        },
        _investigation_blockers(cfg, options),
    )


def _investigation_execute(path: Path, run: ScientificRunDirectory, options: RunOptions) -> Outcome:
    from matsim_agents.workflows.investigation import run_investigation
    from matsim_agents.workflows.phase_exploration import run_phase_exploration

    cfg = _investigation_config(path)
    mlip = _mlip_override(options)

    def phase_runner(composition, policy, output_dir):
        kwargs = {"random_seed": INVESTIGATION_SEED}
        if mlip is not None:
            kwargs["mlip"] = mlip
        return run_phase_exploration(
            composition, policy=policy, output_dir=output_dir, exploration_kwargs=kwargs
        )

    result = run_investigation(cfg, phase_runner=phase_runner)
    return Outcome(
        _WORKFLOW_STATE[result.status.value],
        result.run_directory,
        {
            "hypothesis": result.hypothesis.hypothesis,
            "compositions": result.hypothesis.proposed_compositions,
            "explorations": len(result.explorations),
            "random_seed": INVESTIGATION_SEED,
        },
    )


# --- registry ---------------------------------------------------------------------------

HANDLERS: dict[str, Handler] = {
    "application": Handler(_application_preflight, _application_execute),
    "al": Handler(_al_preflight, _al_execute),
    "neb": Handler(_neb_preflight, _neb_execute),
    "external": Handler(_external_preflight, _external_execute),
    "relax": Handler(_relax_preflight, _relax_execute),
    "debate": Handler(_debate_preflight, _debate_execute),
    "llm-check": Handler(_llm_check_preflight, _llm_check_execute),
    "characterize": Handler(_characterize_preflight, _characterize_execute),
    "explore": Handler(_explore_preflight, _explore_execute, _explore_resume),
    "alloy-spec": Handler(_alloy_preflight, _alloy_execute),
    "investigation": Handler(_investigation_preflight, _investigation_execute),
}

# Files that configure something else; `run` explains what to use instead.
AUXILIARY: dict[str, str] = {
    "mlip": "an MLIP model config; reference it from a relaxation, exploration or AL config",
    "policy": "an application ranking policy; reference it from an application config",
    "mcp": "an MCP server config; use it with `matsim-agents objective --mcp-config`",
    "supercell-campaign": "a campaign config; run it through its application.yaml",
    "hierarchy-campaign": "a campaign config; run it through its application.yaml",
    "manifest": "a data manifest referenced by a campaign config",
    "finetune-eval": (
        "an evaluation protocol; pass it as --evaluation-config to "
        "`matsim-agents tools run finetune-eval`"
    ),
    "selection-example": (
        "a synthetic multi-property selection input for matsim_agents.science.acquisition"
    ),
}


def detect_kind(path: str | Path) -> str:
    """The kind of a config file, from its top-level keys."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    raw = load_yaml(path)
    keys = set(raw)
    if "paradigm" in keys and "workflow" in keys:
        return "application"
    if isinstance(raw.get("workflow"), str) and "config" in keys:
        return "workflow"
    rules: list[tuple[Callable[[set[str]], bool], str]] = [
        (lambda k: "phase_policy" in k, "investigation"),
        (lambda k: "external_tools" in k, "external"),
        (lambda k: k == {"evaluation"}, "finetune-eval"),
        (lambda k: {"selection", "candidates", "reference_alloy"} <= k, "selection-example"),
        (lambda k: {"hypothesis", "rounds"} <= k, "debate"),
        (lambda k: {"provider", "model"} <= k, "llm-check"),
        (lambda k: "contract_id" in k, "characterize"),
        (lambda k: {"mode", "structure_path"} <= k, "relax"),
        (lambda k: {"parent", "defect"} <= k, "neb"),
        (lambda k: {"random_seed", "md", "acquisition"} <= k, "al"),
        (lambda k: {"formula", "mlip"} <= k, "explore"),
        (lambda k: {"crystal_structure", "counts"} <= k, "alloy-spec"),
        (lambda k: k == {"mlip"}, "mlip"),
        (lambda k: "policy_version" in k, "policy"),
        (lambda k: k == {"mcp"}, "mcp"),
        (lambda k: "parent_supercell" in k, "supercell-campaign"),
        (lambda k: {"campaign", "target", "data"} <= k, "hierarchy-campaign"),
        (lambda k: {"library", "files"} <= k, "manifest"),
    ]
    for matches, kind in rules:
        if matches(keys):
            return kind
    raise UnrunnableInput(f"{path}: not a recognized run configuration (keys: {sorted(keys)})")


def _resolve(path: Path) -> tuple[str, Path]:
    """Follow a workflow wrapper to the config it names."""
    kind = detect_kind(path)
    if kind == "workflow":
        inner = (path.parent / str(load_yaml(path)["config"])).resolve()
        return _resolve(inner)
    return kind, path


def _handler(kind: str, path: Path) -> Handler:
    if kind in AUXILIARY:
        raise UnrunnableInput(f"{path} is {AUXILIARY[kind]}; it is not run directly")
    if kind not in HANDLERS:
        raise UnrunnableInput(f"{path}: no runner for kind {kind!r}")
    return HANDLERS[kind]


def _preflight(handler: Handler, target: Path, options: RunOptions) -> Preflight:
    """Run a preflight; a config that cannot load for unset ${VAR}s is blocked, not invalid."""
    try:
        return handler.preflight(target, options)
    except ValueError:
        missing = unset_variables(load_yaml(target))
        if not missing:
            raise
        blockers = [
            f"environment variable {name} is not set" + (f": {hint}" if hint else "")
            for name, hint in missing.items()
        ]
        return Preflight({}, blockers)


def validate(path: str | Path, options: RunOptions | None = None) -> tuple[str, Preflight]:
    """Validate statically and report blockers; no side effects. Raises on invalid input."""
    kind, target = _resolve(Path(path))
    return kind, _preflight(_handler(kind, target), target, options or RunOptions())


def run(
    path: str | Path, *, runs_dir: str | Path = "runs", options: RunOptions | None = None
) -> tuple[Path, RunRecord]:
    """Execute a config file; blocked, failed and interrupted runs are recorded, not lost."""
    source = Path(path)
    kind, target = _resolve(source)
    handler = _handler(kind, target)
    options = options or RunOptions()
    try:
        pre = _preflight(handler, target, options)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        run_dir, record = start_run(runs_dir, kind=kind, source=source)
        record.state = "failed"
        record.failure_reason = f"invalid or incomplete configuration: {error}"
        write_record(run_dir, record)
        return run_dir.path, record
    run_dir, record = start_run(runs_dir, kind=kind, source=source, resolved_config=pre.summary)
    record.summary = pre.summary
    if pre.blockers:
        record.state, record.blockers = "blocked", pre.blockers
        write_record(run_dir, record)
        return run_dir.path, record
    try:
        outcome = handler.execute(target, run_dir, options)
    except KeyboardInterrupt:
        record.state, record.resumable = "interrupted", handler.resume is not None
        write_record(run_dir, record)
        raise
    except Exception as error:  # noqa: BLE001 - recorded with its type, then reported
        record.state, record.failure_reason = "failed", f"{type(error).__name__}: {error}"
        write_record(run_dir, record)
        return run_dir.path, record
    _apply(record, outcome, handler)
    write_record(run_dir, record)
    return run_dir.path, record


def _apply(record: RunRecord, outcome: Outcome, handler: Handler) -> None:
    record.state = outcome.state
    record.native_run_directory = outcome.native_run_directory
    record.summary = {**record.summary, **outcome.summary}
    record.failure_reason = outcome.failure_reason
    record.resumable = handler.resume is not None and outcome.state in {"partial", "failed"}


def resume(reference: str | Path, *, runs_dir: str | Path | None = None) -> tuple[Path, RunRecord]:
    """Continue a resumable run, or re-run a blocked one once its blockers are resolved."""
    roots = [runs_dir] if runs_dir else None
    directory = find_run(reference, roots)
    record = load_record(directory)
    if record.state == "complete":
        return directory, record
    if record.state == "blocked" and record.source:
        return run(record.source, runs_dir=directory.parent)
    handler = HANDLERS.get(record.kind)
    if handler is None or handler.resume is None:
        raise ValueError(f"{record.kind} runs cannot be resumed; start a new run instead")
    run_dir = ScientificRunDirectory.open(directory)
    try:
        outcome = handler.resume(record)
    except Exception as error:  # noqa: BLE001
        record.state, record.failure_reason = "failed", f"{type(error).__name__}: {error}"
    else:
        _apply(record, outcome, handler)
    write_record(run_dir, record)
    return directory, record


def exit_code(record: RunRecord) -> int:
    """0 finished, 1 failed or interrupted, 3 blocked on a missing resource or approval."""
    return {"complete": 0, "partial": 0, "blocked": 3}.get(record.state, 1)


__all__ = [
    "HANDLERS",
    "Outcome",
    "Preflight",
    "RunOptions",
    "UnrunnableInput",
    "backend_blockers",
    "detect_kind",
    "exit_code",
    "load_workflow_config",
    "relaxation_blockers",
    "resume",
    "run",
    "validate",
]
