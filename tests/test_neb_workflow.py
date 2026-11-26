"""NEB workflow integration with a fake launcher (no QE needed)."""

import json
from pathlib import Path

import pytest

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.neb.endpoints import generate_vacancy_hop, vacancy_neighbors
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.neb import (
    NOT_ASSESSED,
    NebWorkflowError,
    ingest_neb_results,
    prepare_neb_workflow,
    run_neb_workflow,
)

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))
ENERGIES = [-100.0, -99.6, -99.2, -99.5, -99.8, -100.1]
ERRORS = [0.0, 0.03, 0.02, 0.04, 0.03, 0.0]


def _pair():
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config, A)[0]
    return generate_vacancy_hop(initial, config, PARENT, mover, A)


def _cfg(tmp_path, **over):
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir(exist_ok=True)
    for name in ("Mo.upf", "Ta.upf", "W.upf"):
        (pseudo / name).write_text("x")
    base = {
        "n_images": 6,
        "species": ("Mo", "Ta", "W"),
        "pseudo_dir": str(pseudo),
        "pseudopotentials": {"Mo": "Mo.upf", "Ta": "Ta.upf", "W": "W.upf"},
        "ecutwfc_ry": 50.0,
        "ecutrho_ry": 400.0,
        "occupations": "smearing",
        "degauss_ry": 0.02,
        "kpts": (2, 2, 2),
        "nspin": 1,
    }
    base.update(over)
    return QENebConfig(**base)


def _fake(energies=ENERGIES, errors=ERRORS, converged=True, exit_code=0, limit=False):
    def launch(argv, *, cwd, stdout, timeout_sec, env):
        top = max(energies)
        text = (
            "\n     ------------------------------ iteration   7 "
            "------------------------------\n\n"
            f"\n     activation energy (->) = {top - energies[0]:10.6f} eV\n"
            f"     activation energy (<-) = {top - energies[-1]:10.6f} eV\n\n"
            "     image        energy (eV)        error (eV/A)        frozen\n\n"
        )
        for i, (e, err) in enumerate(zip(energies, errors, strict=True), start=1):
            flag = "T" if i in (1, len(energies)) else "F"
            text += f"     {i:5d}    {e:15.7f}          {err:10.6f}            {flag}\n"
        if converged:
            text += "\n     neb: convergence achieved in   7 iterations\n"
        elif limit:
            text += "\n     neb: reached the maximum number of steps\n"
        stdout.write(text)
        Path(cwd, "neb.path").write_text("RESTART INFORMATION\n7\n")
        rel = [e - energies[0] for e in energies]
        Path(cwd, "neb.dat").write_text(
            "".join(
                f"  {k / 5:16.10f}  {r:16.10f}  {x:16.10f}\n"
                for k, (r, x) in enumerate(zip(rel, errors, strict=True))
            )
        )
        return exit_code

    return launch


def test_dry_run_prepares_everything_and_runs_nothing(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not launch")

    outcome = run_neb_workflow(_pair(), _cfg(tmp_path), tmp_path / "w", launcher=forbidden)
    plan = outcome.plan
    assert plan.status == "planned" and plan.dry_run and outcome.result is None
    assert plan.pathway_state == "endpoint_relaxation_pending"
    assert (tmp_path / "w" / "pathway" / "pathway.json").is_file()
    assert Path(plan.input_path).is_file()
    assert not (tmp_path / "w" / "qe" / "neb.stage1.out").exists()
    saved = json.loads((tmp_path / "w" / "neb-workflow.json").read_text())
    assert saved["pathway_id"] == plan.pathway_id and saved["config_hash"] == plan.config_hash


def test_invalid_pathway_is_rejected_without_qe_input(tmp_path):
    plan = prepare_neb_workflow(_pair(), _cfg(tmp_path, n_images=4), tmp_path / "w", max_step=0.01)
    assert plan.status == "rejected" and plan.reasons
    assert plan.input_path is None
    assert not list((tmp_path / "w").glob("qe/neb.stage*.in"))


def test_wrong_species_and_existing_directory_raise(tmp_path):
    with pytest.raises(NebWorkflowError, match="species"):
        two = _cfg(tmp_path, species=("Mo", "W"), pseudopotentials={"Mo": "Mo.upf", "W": "W.upf"})
        prepare_neb_workflow(_pair(), two, tmp_path / "a")
    prepare_neb_workflow(_pair(), _cfg(tmp_path), tmp_path / "b")
    with pytest.raises(NebWorkflowError, match="already holds"):
        prepare_neb_workflow(_pair(), _cfg(tmp_path), tmp_path / "b")


def test_dispatch_refuses_unrelaxed_endpoints_by_default(tmp_path):
    with pytest.raises(NebWorkflowError, match="not relaxed"):
        run_neb_workflow(_pair(), _cfg(tmp_path), tmp_path / "w", dispatch=True, launcher=_fake())
    assert not (tmp_path / "w" / "qe" / "neb.stage1.out").exists()


def test_mocked_dispatch_ingests_result_and_records_provenance(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    outcome = run_neb_workflow(
        _pair(),
        _cfg(tmp_path),
        tmp_path / "w",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=ledger,
    )
    result = outcome.result
    assert result.state == "converged_unqualified" and not result.endpoints_qualified
    assert result.forward_barrier_eV == pytest.approx(0.8)
    assert result.reverse_barrier_eV == pytest.approx(0.9)
    assert not outcome.plan.dry_run
    for item in NOT_ASSESSED:
        assert item in outcome.qualification_blockers
    assert any("endpoints not numerically qualified" in b for b in outcome.qualification_blockers)
    [record] = ledger.evidence()
    assert record.kind == "migration_barrier" and record.outcome == "measured"
    assert {v.name for v in record.values} == {"forward_barrier", "reverse_barrier"}
    assert record.evidence_level == "low_fidelity_dft"
    assert any(k.endswith("pathway.json") for k in record.input_sha256)
    assert any(k.endswith("neb.stage1.out") for k in record.output_artifact_sha256)
    assert record.calculation_id.startswith(outcome.plan.pathway_id)
    assert json.loads((tmp_path / "w" / "neb-result.json").read_text())["evidence_id"] == (
        record.evidence_id
    )
    assert ledger.verify_artifacts() == []


def test_unconverged_run_is_not_barrier_evidence(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    outcome = run_neb_workflow(
        _pair(),
        _cfg(tmp_path),
        tmp_path / "w",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(converged=False, limit=True),
        ledger=ledger,
    )
    assert outcome.result.state == "unconverged"
    [record] = ledger.evidence()
    assert record.outcome == "not_converged" and not record.values and record.failure_reason


def test_failed_process_records_failure_without_energies(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    outcome = run_neb_workflow(
        _pair(),
        _cfg(tmp_path),
        tmp_path / "w",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(exit_code=1, converged=False),
        ledger=ledger,
    )
    assert outcome.result.state in ("failed", "incomplete")
    assert outcome.result.forward_barrier_eV is None
    [record] = ledger.evidence()
    assert record.outcome == "failed_to_compute" and not record.values


def test_ingest_requires_a_run_stage_and_intact_pathway(tmp_path):
    prepare_neb_workflow(_pair(), _cfg(tmp_path), tmp_path / "w")
    with pytest.raises(NebWorkflowError, match="no stage has been run"):
        ingest_neb_results(tmp_path / "w")
    with pytest.raises(NebWorkflowError, match="no neb-workflow.json"):
        ingest_neb_results(tmp_path / "missing")
    (tmp_path / "w" / "pathway" / "images.extxyz").write_text("tampered\n")
    with pytest.raises(NebWorkflowError):
        ingest_neb_results(tmp_path / "w")


def test_supplied_images_replace_the_linear_guess(tmp_path):
    from matsim_agents.science.neb.images import generate_linear_images

    pair = _pair()
    images = [
        image.copy()
        for image in generate_linear_images(pair.initial, pair.final, 6, min_distance=1.0).images
    ]
    images[2].positions[0, 0] += 0.01  # an optimized path differs from the linear guess
    outcome = run_neb_workflow(
        pair, _cfg(tmp_path), tmp_path / "w", images=images, launcher=lambda *a, **k: 0
    )
    assert outcome.plan.status == "planned"
    saved = json.loads((tmp_path / "w" / "pathway" / "pathway.json").read_text())
    assert saved["parameters"]["method"] == "supplied"

    rejected = prepare_neb_workflow(pair, _cfg(tmp_path), tmp_path / "w2", images=images[:4])
    assert rejected.status == "rejected"


def test_constrained_supplied_images_are_rejected(tmp_path):
    from ase.constraints import FixAtoms

    from matsim_agents.science.neb.images import generate_linear_images

    pair = _pair()
    images = [
        image.copy()
        for image in generate_linear_images(pair.initial, pair.final, 6, min_distance=1.0).images
    ]
    images[2].set_constraint(FixAtoms(indices=[0]))
    plan = prepare_neb_workflow(pair, _cfg(tmp_path), tmp_path / "w", images=images)
    assert plan.status == "rejected"
    assert any("constrained" in reason for reason in plan.reasons)
    assert plan.input_path is None


def test_mlip_relaxed_endpoints_do_not_name_the_qe_method(tmp_path):
    pair = _pair()
    transition = pair.transition.model_copy(
        update={
            "initial": pair.transition.initial.model_copy(update={"method": "mlip:model-a"}),
            "final": pair.transition.final.model_copy(update={"method": "mlip:model-a"}),
        }
    )
    pair = pair.__class__(
        initial=pair.initial, final=pair.final, transition=transition, final_config=None
    )
    plan = prepare_neb_workflow(pair, _cfg(tmp_path), tmp_path / "w")
    assert plan.method_id.startswith("qe-neb-")
