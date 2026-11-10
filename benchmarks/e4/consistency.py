"""Prespecified cross-facility comparison for E4 consistency records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

REQUIRED = {
    "facility",
    "repetition",
    "contract_sha256",
    "structure_sha256",
    "model_checkpoint_sha256",
    "dft_method_sha256",
    "scientific_conclusion",
    "mlip_converged",
    "dft_converged",
    "evidence_admissible",
    "mlip_energy_eV_per_atom",
    "dft_energy_eV_per_atom",
    "max_force_eV_per_A",
    "geometry_rmsd_to_canonical_A",
}
EXPECTED_FACILITIES = {"aurora", "frontier", "perlmutter"}


def compare_consistency_records(
    records: Iterable[dict[str, Any]],
    *,
    energy_atol_eV_per_atom: float = 0.05,
    force_atol_eV_per_A: float = 0.05,
    geometry_rmsd_atol_A: float = 0.05,
    repetitions_per_facility: int = 3,
) -> dict[str, Any]:
    rows = list(records)
    errors = [
        f"record {i} missing {sorted(REQUIRED - set(row))}"
        for i, row in enumerate(rows)
        if REQUIRED - set(row)
    ]
    if errors:
        return {"status": "invalid", "errors": errors}
    by_facility = {
        facility: [r for r in rows if r["facility"] == facility] for facility in EXPECTED_FACILITIES
    }
    extras = sorted({r["facility"] for r in rows} - EXPECTED_FACILITIES)
    if extras:
        errors.append(f"unexpected facilities: {extras}")
    for facility, group in by_facility.items():
        if len(group) != repetitions_per_facility:
            errors.append(
                f"{facility}: expected {repetitions_per_facility} repetitions, found {len(group)}"
            )
    if errors:
        return {"status": "incomplete", "errors": errors}

    same_contract = len({r["contract_sha256"] for r in rows}) == 1
    same_structure = len({r["structure_sha256"] for r in rows}) == 1
    same_model = (
        None not in {r["model_checkpoint_sha256"] for r in rows}
        and len({r["model_checkpoint_sha256"] for r in rows}) == 1
    )
    same_dft_method = (
        None not in {r["dft_method_sha256"] for r in rows}
        and len({r["dft_method_sha256"] for r in rows}) == 1
    )
    conclusions = {r["scientific_conclusion"] for r in rows}
    valid = all(
        r["mlip_converged"] and r["dft_converged"] and r["evidence_admissible"] for r in rows
    )
    errors = []
    if not same_contract:
        errors.append("scientific contract hashes differ")
    if not same_structure:
        errors.append("input structure hashes differ")
    if len(conclusions) != 1:
        errors.append("scientific conclusions differ")
    if not valid:
        errors.append("one or more outputs failed convergence/admissibility")
    if not same_model:
        errors.append(
            "model checkpoint differs or is unidentified; MLIP energy equivalence not assessed"
        )
    if not same_dft_method:
        errors.append("DFT method differs or is unidentified; DFT energy equivalence not assessed")

    def spread(key: str) -> float | None:
        values = [float(r[key]) for r in rows]
        if any(value != value or abs(value) == float("inf") for value in values):
            return None
        return max(values) - min(values)

    mlip_spread = spread("mlip_energy_eV_per_atom") if same_model else None
    dft_spread = spread("dft_energy_eV_per_atom") if same_dft_method else None
    force_spread = spread("max_force_eV_per_A")
    geometry_max = max(float(r["geometry_rmsd_to_canonical_A"]) for r in rows)
    if mlip_spread is not None and mlip_spread > energy_atol_eV_per_atom:
        errors.append("MLIP energy range exceeds tolerance")
    if dft_spread is not None and dft_spread > energy_atol_eV_per_atom:
        errors.append("DFT energy range exceeds tolerance")
    if force_spread is not None and force_spread > force_atol_eV_per_A:
        errors.append("maximum-force range exceeds tolerance")
    if geometry_max > geometry_rmsd_atol_A:
        errors.append("geometry RMSD exceeds tolerance")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "facility_counts": {key: len(value) for key, value in by_facility.items()},
        "same_contract": same_contract,
        "same_structure": same_structure,
        "same_model_checkpoint": same_model,
        "same_dft_method": same_dft_method,
        "same_scientific_conclusion": len(conclusions) == 1,
        "all_outputs_converged_and_admissible": valid,
        "mlip_energy_range_eV_per_atom": mlip_spread,
        "dft_energy_range_eV_per_atom": dft_spread,
        "max_force_range_eV_per_A": force_spread,
        "maximum_geometry_rmsd_to_canonical_A": geometry_max,
        "tolerances": {
            "energy_eV_per_atom": energy_atol_eV_per_atom,
            "max_force_eV_per_A": force_atol_eV_per_A,
            "geometry_rmsd_A": geometry_rmsd_atol_A,
        },
    }
