#!/usr/bin/env python3
"""Qualify characterization contracts with real UMA and VASP calculations.

1. MLIP (UMA): relax MgO rocksalt, and rank rocksalt against zincblende.
2. VASP (PBE): compute Si, C, and 2H-SiC references with ``relaxation``
   contracts at converged DFT, then ask whether 3C-SiC is on the Si-C hull.
3. Protocol checks without calculations: incomplete references give
   ``inconclusive`` and a method mismatch gives ``invalid_protocol``.
4. Resume: the stability contract run again performs no calculation.

Software checks decide the exit code. The scientific values are reported as
computed, never asserted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ase.build import bulk
from ase.io import write

from matsim_agents.campaign.characterization import (
    CharacterizationContract,
    load_contract,
    run_characterization,
)
from matsim_agents.discovery.stability import ReferenceEnergySet, ReferencePhaseEntry
from matsim_agents.execution.contracts import CharacterizationStatus, EvidenceLevel
from matsim_agents.workflows.relaxation import run_relaxation

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "characterization"
SIGNATURE = "vasp-pbe-paw64-encut520-ismear0-sigma0.05-kspacing0.2"


class CountingRelax:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, cfg):
        self.calls.append((Path(cfg.structure_path).name, cfg.mode.value))
        return run_relaxation(cfg)


def _observable(report, name, structure=None):
    for value in report.observables:
        if value.name == name and (structure is None or value.structure == structure):
            return value.value
    return None


def _summary(report) -> dict:
    return {
        "status": report.status.value,
        "conclusion": report.conclusion,
        "missing": report.missing,
        "observables": [
            {
                "name": v.name,
                "structure": v.structure,
                "value": v.value,
                "unit": v.unit,
                "evidence_level": v.evidence_level.value,
            }
            for v in report.observables
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vasp-launcher", required=True)
    parser.add_argument("--potcar-dir", required=True)
    parser.add_argument("--skip-mlip", action="store_true")
    parser.add_argument("--skip-dft", action="store_true", help="Run only the MLIP recipes.")
    args = parser.parse_args()
    root = args.output_root.resolve()
    structures = root / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)

    mlip = {
        "mlip_backend": "uma",
        "uma_model_name": "uma-s-1p1",
        "uma_task": "omat",
        "mlp_device": "cuda",
    }

    # 1. MLIP recipes on the shipped examples.
    if not args.skip_mlip:
        for name in ("relaxation", "energy_ranking"):
            data = load_contract(EXAMPLES / f"{name}.yaml").model_dump()
            data["output_root"] = root / "mlip"
            data["methods"]["mlip"] = mlip
            contract = CharacterizationContract.model_validate(data)
            relax = CountingRelax()
            report = run_characterization(contract, relax=relax)
            results[f"mlip-{name}"] = _summary(report)
            # A resubmission reuses completed calculations: judge the evidence.
            check(
                f"mlip {name}: answered from MLIP relaxations only",
                report.status
                in {
                    CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
                    CharacterizationStatus.INCONCLUSIVE,
                }
                and bool(report.evidence)
                and all(
                    item.evidence_level == EvidenceLevel.MLIP_RELAXATION for item in report.evidence
                )
                and all(mode == "mlip" for _s, mode in relax.calls),
                f"status={report.status.value} calls={relax.calls}",
            )

    if args.skip_dft:
        return _finish(root, results, checks)

    # 2. VASP references and the stability question.
    phases = {
        "Si-diamond": bulk("Si", "diamond", a=5.47),
        "C-diamond": bulk("C", "diamond", a=3.57),
        "SiC-2H": bulk("SiC", "wurtzite", a=3.08, c=5.05),
        "SiC-3C": bulk("SiC", "zincblende", a=4.38),
    }
    paths = {}
    for phase, atoms in phases.items():
        paths[phase] = structures / f"{phase}.vasp"
        write(paths[phase], atoms, format="vasp", direct=True, sort=True)
    dft = {
        "backend": "vasp",
        "launcher": args.vasp_launcher,
        "potcar_dir": args.potcar_dir,
        "timeout_sec": 1800,
        "settings": {"encut_ev": 520.0, "ismear": 0, "sigma": 0.05, "kspacing": 0.2},
    }
    common = {
        "methods": {"dft": dft, "dft_method_signature": SIGNATURE},
        "tolerances": {"force_tolerance_eV_per_A": 0.01, "max_relaxation_steps": 100},
        "dft_approved": True,
        "output_root": str(root / "dft"),
    }
    converged_dft = [
        {"name": name, "unit": unit, "minimum_evidence_level": "converged_dft"}
        for name, unit in (
            ("relaxed_structure", "structure"),
            ("energy_per_atom", "eV/atom"),
            ("residual_max_force", "eV/A"),
        )
    ]
    energy: dict[str, float | None] = {}
    for phase, formula in (("Si-diamond", "Si"), ("C-diamond", "C"), ("SiC-2H", "SiC")):
        relax = CountingRelax()
        report = run_characterization(
            CharacterizationContract.model_validate(
                {
                    "contract_id": f"reference-{phase}",
                    "objective": f"Converged PBE energy of {phase} for the Si-C hull.",
                    "formula": formula,
                    "structures": [str(paths[phase])],
                    "recipe": "relaxation",
                    "observables": converged_dft,
                    **common,
                }
            ),
            relax=relax,
        )
        results[f"reference-{phase}"] = _summary(report)
        energy[phase] = _observable(report, "energy_per_atom")
        check(
            f"reference {phase}: converged DFT energy",
            report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
            and energy[phase] is not None
            and all(item.evidence_level == EvidenceLevel.CONVERGED_DFT for item in report.evidence)
            and all(mode == "dft" for _s, mode in relax.calls),
            f"status={report.status.value} E={energy[phase]} missing={report.missing}",
        )
    if any(value is None for value in energy.values()):
        return _finish(root, results, checks)

    e_si, e_c = energy["Si-diamond"], energy["C-diamond"]
    formation_2h = energy["SiC-2H"] - (e_si + e_c) / 2
    references = ReferenceEnergySet(
        identifier="si-c-vasp-pbe-qualification",
        method_signature=SIGNATURE,
        backend="vasp",
        elemental_energies_eV_per_atom={"Si": e_si, "C": e_c},
        phase_entries=[
            ReferencePhaseEntry(
                phase_id="SiC-2H",
                formula="SiC",
                formation_energy_eV_per_atom=formation_2h,
                method_signature=SIGNATURE,
                backend="vasp",
                structure_path=str(paths["SiC-2H"]),
                energy_per_atom_eV=energy["SiC-2H"],
                source="characterization:reference-SiC-2H",
            )
        ],
    )
    complete = root / "references-si-c.json"
    complete.write_text(references.model_dump_json(indent=2) + "\n")
    elemental_only = root / "references-si-c-elemental-only.json"
    elemental_only.write_text(
        references.model_copy(update={"phase_entries": []}).model_dump_json(indent=2) + "\n"
    )

    def stability(contract_id: str, reference_file: Path, signature: str = SIGNATURE):
        return CharacterizationContract.model_validate(
            {
                "contract_id": contract_id,
                "objective": "Is 3C-SiC on the Si-C convex hull of the computed PBE phases?",
                "formula": "SiC",
                "structures": [str(paths["SiC-3C"])],
                "recipe": "thermodynamic_stability",
                "reference_energies": str(reference_file),
                **{**common, "methods": {"dft": dft, "dft_method_signature": signature}},
            }
        )

    # 3. Protocol checks: no calculation may run.
    relax = CountingRelax()
    report = run_characterization(stability("sic-3c-incomplete", elemental_only), relax=relax)
    results["stability-incomplete-references"] = _summary(report)
    check(
        "incomplete references: inconclusive without DFT",
        report.status == CharacterizationStatus.INCONCLUSIVE and not relax.calls,
        f"missing={report.missing}",
    )
    report = run_characterization(
        stability("sic-3c-mismatch", complete, signature="qe-pbe-other"), relax=relax
    )
    results["stability-method-mismatch"] = _summary(report)
    check(
        "method mismatch: invalid_protocol without DFT",
        report.status == CharacterizationStatus.INVALID_PROTOCOL and not relax.calls,
        f"missing={report.missing}",
    )

    # The stability answer, then a resume that must not recalculate.
    contract = stability("sic-3c-stability", complete)
    relax = CountingRelax()
    report = run_characterization(contract, relax=relax)
    results["stability"] = _summary(report)
    check(
        "3C-SiC stability: answered from converged DFT",
        report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
        and relax.calls in ([], [("SiC-3C.vasp", "dft")]),
        f"status={report.status.value} conclusion={report.conclusion} missing={report.missing}",
    )
    again = CountingRelax()
    resumed = run_characterization(contract, relax=again)
    check(
        "resume: no calculation repeated, same answer",
        not again.calls and resumed.observables == report.observables,
        f"calls={again.calls}",
    )
    results["computed"] = {
        "E_Si_eV_per_atom": e_si,
        "E_C_eV_per_atom": e_c,
        "E_SiC_2H_eV_per_atom": energy["SiC-2H"],
        "formation_SiC_2H_eV_per_atom": formation_2h,
        "formation_SiC_3C_eV_per_atom": _observable(report, "formation_energy_per_atom"),
        "e_above_hull_SiC_3C_eV_per_atom": _observable(report, "energy_above_hull"),
    }
    return _finish(root, results, checks)


def _finish(root: Path, results: dict, checks: list[dict]) -> int:
    failed = [c for c in checks if not c["passed"]]
    summary = {
        "passed": not failed,
        "checks": checks,
        "results": results,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (root / "qualification-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["results"].get("computed", {}), indent=2))
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
