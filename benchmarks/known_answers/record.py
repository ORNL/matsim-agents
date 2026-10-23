"""Record the known-answer suite's DFT runs for the offline replay tier.

    python benchmarks/known_answers/record.py [WORK_DIR] [--pw-x PATH]

WORK_DIR defaults to the repository's git-ignored runs/known-answer-record;
each invocation records into a new subdirectory of it.

1. Relaxes the reference phases (diamond Si, diamond C, 2H-SiC) with the
   suite's DFT method and writes ``references.json`` from their energies into
   a copy of the suite.
2. Runs every case live on that copy through the characterization pipeline.
3. Only if every case passes, replaces the suite's ``references.json`` and
   ``replay/<case_id>/<step_id>/`` (each relaxation's ``pw.in``/``pw.out``);
   otherwise exits 1 and leaves the suite untouched.

Run it on a node with pw.x (serial runs of a few seconds to a minute each).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from matsim_agents.campaign.characterization import load_contract, run_characterization
from matsim_agents.campaign.known_answers import load_suite, run_known_answers
from matsim_agents.discovery.stability import (
    ElementalReferenceEntry,
    ReferenceEnergySet,
    ReferencePhaseEntry,
)

SUITE = Path(__file__).resolve().parent
REFERENCE_PHASES = {
    "Si": ("Si-diamond", "Si"),
    "C": ("C-diamond", "C"),
    "SiC-2H": ("SiC-2H", "SiC"),
}


def _pw_version(pw_x: str) -> str:
    out = subprocess.run(
        [pw_x], input="", capture_output=True, text=True, timeout=60, check=False
    ).stdout
    return next((line.strip() for line in out.splitlines() if "Program PWSCF" in line), "unknown")


def _step(path: Path) -> str:
    return next(p for p in reversed(path.parts) if re.fullmatch(r"relax-\w+-\d{3}", p))


def _relax_reference(name: str, formula: str, template: dict, work: Path, pw_x: str) -> dict:
    structure = SUITE / "structures" / f"{name}.vasp"
    contract = {
        **{k: template[k] for k in ("methods", "tolerances")},
        "contract_id": f"known-answer-reference-{name}",
        "objective": f"Reference energy of {name}.",
        "formula": formula,
        "recipe": "relaxation",
        "structures": [str(structure)],
        "observables": [
            {"name": n, "unit": u, "minimum_evidence_level": "converged_dft"}
            for n, u in (
                ("relaxed_structure", "structure"),
                ("energy_per_atom", "eV/atom"),
                ("residual_max_force", "eV/A"),
            )
        ],
        "budget": {"max_dft_calculations": 1},
        "dft_approved": True,
        "output_root": str(work / "references"),
    }
    contract["methods"]["dft"]["launcher"] = [pw_x, "-in", "{input}"]
    path = work / f"reference-{name}.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False))
    report = run_characterization(load_contract(path))
    if report.status != "sufficiently_characterized":
        raise SystemExit(f"reference {name} did not converge: {report.missing}")
    energy = next(v.value for v in report.observables if v.name == "energy_per_atom")
    return {
        "energy_per_atom": float(energy),
        "structure": structure,
        "evidence_id": report.evidence[0].evidence_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "work_dir", type=Path, nargs="?", default=SUITE.parents[1] / "runs" / "known-answer-record"
    )
    parser.add_argument("--pw-x", default="pw.x")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="record-", dir=args.work_dir.resolve()))
    print(f"Recording in {work}")

    template = yaml.safe_load((SUITE / "contracts" / "sic-3c-on-hull.yaml").read_text())
    signature = template["methods"]["dft_method_signature"]
    version = _pw_version(args.pw_x)
    refs = {
        key: _relax_reference(name, formula, json.loads(json.dumps(template)), work, args.pw_x)
        for key, (name, formula) in REFERENCE_PHASES.items()
    }

    def provenance(ref: dict) -> dict[str, str]:
        return {
            "code": version,
            "ledger_evidence_id": ref["evidence_id"],
            "recorded_by": "record.py",
        }

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    e_si, e_c = refs["Si"]["energy_per_atom"], refs["C"]["energy_per_atom"]
    e_2h = refs["SiC-2H"]["energy_per_atom"] - (e_si + e_c) / 2
    references = ReferenceEnergySet(
        identifier="known-answer-si-c",
        method_signature=signature,
        backend="qe",
        elemental_energies_eV_per_atom={"Si": e_si, "C": e_c},
        elemental_entries={
            el: ElementalReferenceEntry(
                element=el,
                phase_id=f"{el}-diamond",
                reference_formula=el,
                energy_eV_per_atom=refs[el]["energy_per_atom"],
                method_signature=signature,
                backend="qe",
                structure_path=f"structures/{el}-diamond.vasp",
                structure_hash=digest(refs[el]["structure"]),
                source="computed",
                provenance=provenance(refs[el]),
            )
            for el in ("Si", "C")
        },
        phase_entries=[
            ReferencePhaseEntry(
                phase_id="SiC-2H",
                formula="SiC",
                formation_energy_eV_per_atom=e_2h,
                method_signature=signature,
                backend="qe",
                structure_path="structures/SiC-2H.vasp",
                structure_hash=digest(refs["SiC-2H"]["structure"]),
                energy_per_atom_eV=refs["SiC-2H"]["energy_per_atom"],
                source="computed",
                provenance=provenance(refs["SiC-2H"]),
            )
        ],
    )
    staged = work / "suite"
    shutil.copytree(SUITE, staged, ignore=shutil.ignore_patterns("replay", "__pycache__"))
    (staged / "references.json").write_text(references.model_dump_json(indent=2) + "\n")

    suite = load_suite(staged)
    result = run_known_answers(
        staged, tier="live", output_dir=work / "live", launcher=[args.pw_x, "-in", "{input}"]
    )
    for verdict in result.verdicts:
        print(f"{verdict.case_id}: {'PASS' if verdict.passed else 'FAIL'} {verdict.reasons}")
    if not result.passed:
        sys.exit(f"Not all cases passed; {SUITE} is unchanged. Runs: {result.run_dir}")

    recordings = {}
    for case in suite.cases:
        steps: dict[str, Path] = {}
        for pw_out in sorted((Path(result.run_dir) / case.case_id).glob("**/relax-*/**/pw.out")):
            step = _step(pw_out.parent)
            if step in steps:
                sys.exit(f"{case.case_id}: two recordings for {step}; {SUITE} is unchanged")
            steps[step] = pw_out
        if not steps:
            sys.exit(f"{case.case_id}: no pw.x runs recorded; {SUITE} is unchanged")
        recordings[case.case_id] = steps
    for case_id, steps in recordings.items():
        replay = SUITE / "replay" / case_id
        shutil.rmtree(replay, ignore_errors=True)
        for step, pw_out in steps.items():
            (replay / step).mkdir(parents=True)
            shutil.copyfile(pw_out, replay / step / "pw.out")
            shutil.copyfile(pw_out.with_name("pw.in"), replay / step / "pw.in")
    shutil.copyfile(staged / "references.json", SUITE / "references.json")


if __name__ == "__main__":
    main()
