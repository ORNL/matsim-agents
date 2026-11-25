"""Elastic stiffness from stress-strain finite differences.

The workflow strains the cell by +/- amplitudes in each of the six Voigt strain
components, optionally relaxes the ions at fixed cell (internal relaxation),
reads the stress, and forms ``C_ij = d sigma_i / d e_j`` by central differences.
It is backend-agnostic: ``stress_fn`` and ``relax_fn`` are injected, so the same
code serves QE or VASP (production) and MLIPs that implement stress (screening).

Conventions: Voigt order xx, yy, zz, yz, xz, xy; engineering shear strain
(``e_4 = 2 eps_yz``); stress positive in tension; GPa. Strains are applied as
``cell' = cell (I + eps)``.

Scope and limits: small-strain, zero-temperature stiffness. Elastic constants are
not yield strength, toughness, ductility, or creep. Uncertainty reported here is
numerical (spread across strain amplitudes), not DFT or MLIP systematic error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import GPa as ASE_GPA
from pydantic import BaseModel, Field, model_validator

from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.science.properties.calculators import (
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)

VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
StressFn = Callable[[Atoms], tuple[Sequence[float], bool]]
RelaxFn = Callable[[Atoms], tuple[Atoms, bool]]
SYMMETRIZATION_NOTE = (
    "C was symmetrized as (C + C^T)/2 because stress-strain derivatives are only symmetric to "
    "numerical accuracy; the largest relative asymmetry is reported in convergence."
)
MECHANICAL_LIMITATION = (
    "Zero-temperature small-strain stiffness: not yield strength, toughness, ductility, or creep."
)


class ElasticSettings(BaseModel):
    """Numerical settings; recorded with every result, not scientific thresholds."""

    amplitudes: list[float] = Field(
        default_factory=lambda: [0.005, 0.01],
        description="Strain magnitudes; at least two, to measure nonlinearity.",
    )
    relax_ions: bool = Field(
        True, description="Relax internal coordinates at fixed strained cell (relaxed-ion)."
    )
    max_residual_stress_GPa: float = Field(0.1, gt=0)
    max_amplitude_spread: float = Field(
        0.05, gt=0, description="Largest relative spread of C across amplitudes."
    )
    max_asymmetry: float = Field(0.02, gt=0, description="Largest relative |C - C^T|.")

    @model_validator(mode="after")
    def _valid(self) -> ElasticSettings:
        if len(self.amplitudes) < 2 or len(set(self.amplitudes)) != len(self.amplitudes):
            raise ValueError("at least two distinct strain amplitudes are required")
        if any(not 0 < a < 0.05 for a in self.amplitudes):
            raise ValueError("strain amplitudes must be in (0, 0.05): small-strain regime")
        return self


class ElasticResult(BaseModel):
    """Stiffness, derived moduli, and the checks behind their trustworthiness."""

    settings: ElasticSettings
    n_atoms: int
    stiffness_GPa: list[list[float]]
    stiffness_std_GPa: list[list[float]]
    residual_stress_GPa: list[float]
    eigenvalues_GPa: list[float]
    mechanically_stable: bool
    moduli: dict[str, float] = Field(default_factory=dict)
    moduli_std: dict[str, float] = Field(default_factory=dict)
    converged: bool
    convergence: dict[str, Any]
    problems: list[str] = Field(default_factory=list)
    calculations: int
    limitations: list[str] = Field(default_factory=list)


def calculator_supports_stress(calc: Any) -> bool:
    """Whether an ASE calculator declares stress; an MLIP without it cannot give C."""
    return "stress" in getattr(calc, "implemented_properties", ())


def ase_stress_fn(calc_factory: Callable[[], Any]) -> StressFn:
    """Stress function from an ASE calculator; refuses calculators without stress."""

    def stress(atoms: Atoms) -> tuple[Sequence[float], bool]:
        calc = calc_factory()
        if not calculator_supports_stress(calc):
            raise ValueError("this calculator does not implement stress: elasticity unsupported")
        probe = atoms.copy()
        probe.calc = calc
        return probe.get_stress(voigt=True) / ASE_GPA, True

    return stress


def strain_matrix(voigt: Sequence[float]) -> np.ndarray:
    e = np.asarray(voigt, float)
    return np.array(
        [
            [e[0], e[5] / 2, e[4] / 2],
            [e[5] / 2, e[1], e[3] / 2],
            [e[4] / 2, e[3] / 2, e[2]],
        ]
    )


def apply_strain(atoms: Atoms, voigt: Sequence[float]) -> Atoms:
    """A copy with the cell deformed by ``I + eps``; atoms keep their fractional positions."""
    strained = atoms.copy()
    deformation = np.eye(3) + strain_matrix(voigt)
    strained.set_cell(np.asarray(atoms.cell) @ deformation.T, scale_atoms=True)
    return strained


def strain_set(amplitudes: Sequence[float]) -> list[tuple[int, float]]:
    """(Voigt component, signed amplitude) for every strained calculation."""
    return [(j, s * a) for j in range(6) for a in amplitudes for s in (1.0, -1.0)]


def plan_calculations(
    n_atoms: int,
    settings: ElasticSettings,
    *,
    parent_sites: int | None = None,
    node_hours_per_calculation: float | None = None,
    remaining_node_hours: float | None = None,
) -> dict[str, Any]:
    """Count and cost the strained calculations; never substitutes a smaller cell."""
    n = len(strain_set(settings.amplitudes)) + 1
    blockers = []
    if parent_sites is not None and n_atoms != parent_sites:
        blockers.append(
            f"structure has {n_atoms} atoms, not the {parent_sites}-site parent: a smaller "
            "supercell would change the disorder statistics and is not substituted"
        )
    cost = None if node_hours_per_calculation is None else n * node_hours_per_calculation
    if cost is None:
        blockers.append("no measured node-hours per calculation: benchmark one first")
    elif remaining_node_hours is not None and cost > remaining_node_hours:
        blockers.append(f"{cost:.1f} node-hours exceed the remaining {remaining_node_hours:.1f}")
    return {
        "calculations": n,
        "estimated_node_hours": cost,
        "blockers": blockers,
        "status": "RESOURCE_BLOCKED" if blockers else "READY",
    }


def isotropic_moduli(c: np.ndarray) -> dict[str, float]:
    """Voigt, Reuss, and Hill K and G, Hill E and nu, and the universal anisotropy index."""
    s = np.linalg.inv(c)
    d, o = (0, 1, 2), ((0, 1), (0, 2), (1, 2))
    k_v = (sum(c[i, i] for i in d) + 2 * sum(c[i, j] for i, j in o)) / 9
    g_v = (
        sum(c[i, i] for i in d) - sum(c[i, j] for i, j in o) + 3 * sum(c[i, i] for i in (3, 4, 5))
    ) / 15
    k_r = 1 / (sum(s[i, i] for i in d) + 2 * sum(s[i, j] for i, j in o))
    g_r = 15 / (
        4 * sum(s[i, i] for i in d)
        - 4 * sum(s[i, j] for i, j in o)
        + 3 * sum(s[i, i] for i in (3, 4, 5))
    )
    k, g = (k_v + k_r) / 2, (g_v + g_r) / 2
    return {
        "bulk_modulus_voigt": k_v,
        "bulk_modulus_reuss": k_r,
        "shear_modulus_voigt": g_v,
        "shear_modulus_reuss": g_r,
        "bulk_modulus": k,
        "shear_modulus": g,
        "youngs_modulus": 9 * k * g / (3 * k + g),
        "poissons_ratio": (3 * k - 2 * g) / (2 * (3 * k + g)),
        "elastic_anisotropy": 5 * g_v / g_r + k_v / k_r - 6,
    }


def _stiffness_estimates(
    atoms: Atoms,
    stress_fn: StressFn,
    relax_fn: RelaxFn | None,
    settings: ElasticSettings,
    problems: list[str],
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Residual stress, one 6x6 estimate per amplitude, calculation count, all converged."""
    all_ok = True
    count = 0

    def stress_of(structure: Atoms) -> np.ndarray:
        nonlocal all_ok, count
        count += 1
        if relax_fn is not None and settings.relax_ions:
            structure, relaxed = relax_fn(structure)
            all_ok &= bool(relaxed)
        values, converged = stress_fn(structure)
        all_ok &= bool(converged)
        sigma = np.asarray(values, float)
        if sigma.shape != (6,) or not np.all(np.isfinite(sigma)):
            raise ValueError("stress_fn must return six finite Voigt components")
        return sigma

    sigma0 = stress_of(atoms)
    estimates = np.zeros((len(settings.amplitudes), 6, 6))
    for a_index, amplitude in enumerate(settings.amplitudes):
        for j in range(6):
            vector = np.zeros(6)
            vector[j] = amplitude
            plus = stress_of(apply_strain(atoms, vector))
            minus = stress_of(apply_strain(atoms, -vector))
            estimates[a_index, :, j] = (plus - minus) / (2 * amplitude)
    if not all_ok:
        problems.append("a stress or relaxation calculation did not converge")
    return sigma0, estimates, count, all_ok


def compute_elastic_tensor(
    atoms: Atoms,
    stress_fn: StressFn,
    settings: ElasticSettings | None = None,
    *,
    relax_fn: RelaxFn | None = None,
    parent_sites: int | None = None,
) -> ElasticResult:
    """Stiffness by central differences at several amplitudes; atoms must be relaxed."""
    cfg = settings or ElasticSettings()
    if parent_sites is not None and len(atoms) != parent_sites:
        raise ValueError(
            f"{len(atoms)} atoms is not the {parent_sites}-site parent; no smaller supercell "
            "is substituted"
        )
    problems: list[str] = []
    sigma0, estimates, count, all_ok = _stiffness_estimates(
        atoms, stress_fn, relax_fn, cfg, problems
    )
    mean = estimates.mean(axis=0)
    std = estimates.std(axis=0, ddof=1)
    scale = float(np.max(np.abs(mean))) or 1.0
    asymmetry = float(np.max(np.abs(mean - mean.T)) / scale)
    c = (mean + mean.T) / 2
    spread = float(np.max(std) / scale)
    residual = float(np.max(np.abs(sigma0)))
    eigenvalues = np.linalg.eigvalsh(c)
    stable = bool(eigenvalues[0] > 0)
    if residual > cfg.max_residual_stress_GPa:
        problems.append(
            f"residual stress {residual:.3g} GPa exceeds {cfg.max_residual_stress_GPa} GPa: "
            "relax the cell first"
        )
    if spread > cfg.max_amplitude_spread:
        problems.append(f"C varies {spread:.3g} (relative) across amplitudes: not linear")
    if asymmetry > cfg.max_asymmetry:
        problems.append(f"C is asymmetric by {asymmetry:.3g} (relative)")
    if not (relax_fn is not None and cfg.relax_ions):
        problems.append("clamped-ion constants: internal relaxation was not performed")
    moduli: dict[str, float] = {}
    moduli_std: dict[str, float] = {}
    if stable:
        moduli = isotropic_moduli(c)
        per_amplitude = [
            isotropic_moduli((e + e.T) / 2) for e in estimates if np.linalg.eigvalsh(e + e.T)[0] > 0
        ]
        for name in moduli:
            values = [m[name] for m in per_amplitude]
            moduli_std[name] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        if any(math.isnan(v) for v in moduli_std.values()):
            problems.append("moduli are not defined at every amplitude: uncertainty unknown")
    else:
        problems.append(
            f"mechanically unstable: smallest eigenvalue {eigenvalues[0]:.3g} GPa is not positive"
        )
    blocking = [p for p in problems if not p.startswith("mechanically unstable")]
    return ElasticResult(
        settings=cfg,
        n_atoms=len(atoms),
        stiffness_GPa=c.tolist(),
        stiffness_std_GPa=std.tolist(),
        residual_stress_GPa=sigma0.tolist(),
        eigenvalues_GPa=eigenvalues.tolist(),
        mechanically_stable=stable,
        moduli=moduli,
        moduli_std={k: v for k, v in moduli_std.items() if not math.isnan(v)},
        converged=all_ok and not blocking,
        convergence={
            "max_residual_stress_GPa": residual,
            "max_amplitude_spread": spread,
            "max_asymmetry": asymmetry,
            "relaxed_ions": relax_fn is not None and cfg.relax_ions,
            "symmetrization": SYMMETRIZATION_NOTE,
        },
        problems=problems,
        calculations=count,
        limitations=[MECHANICAL_LIMITATION],
    )


def elastic_property_evidence(
    result: ElasticResult,
    *,
    structure_id: str,
    composition: str,
    evidence_ids: list[str],
    level: EvidenceLevel,
    backend: str,
    method: str = "stress_strain",
    conditions: PropertyConditions | None = None,
    qualification: PropertyQualification = PropertyQualification.UNVALIDATED,
) -> list[PropertyEvidence]:
    """PropertyEvidence for the tensor and each derived property; never invents a value."""
    cond = conditions or PropertyConditions()
    dft = level in {EvidenceLevel.LOW_FIDELITY_DFT, EvidenceLevel.CONVERGED_DFT}
    if not result.converged:
        status = PropertyStatus.UNCONVERGED
    else:
        status = PropertyStatus.CALCULATED if dft else PropertyStatus.SURROGATE_PREDICTED
    common: dict[str, Any] = dict(
        structure_id=structure_id,
        composition=composition,
        conditions=cond,
        method=method,
        backend=backend,
        evidence_level=level,
        converged=result.converged if status != PropertyStatus.UNCONVERGED else False,
        convergence=result.convergence,
        evidence_ids=evidence_ids,
        limitations=result.limitations,
        approximation=None,
    )
    if status != PropertyStatus.UNCONVERGED:
        common["qualification"] = qualification
    reason = "; ".join(result.problems) or None
    out: list[PropertyEvidence] = []

    def add(name: str, value: Any, uncertainty: Any, unit: str) -> None:
        extra: dict[str, Any] = {}
        if status == PropertyStatus.UNCONVERGED:
            extra["reason"] = reason or "calculation not converged"
        else:
            extra["uncertainty"] = uncertainty
        out.append(
            PropertyEvidence(
                property=name, status=status, value=value, unit=unit, **common, **extra
            )
        )

    add("elastic_stiffness_tensor", result.stiffness_GPa, result.stiffness_std_GPa, "GPa")
    add(
        "elastic_stability_min_eigenvalue",
        float(result.eigenvalues_GPa[0]),
        float(max(max(row) for row in result.stiffness_std_GPa)),
        "GPa",
    )
    units = {"poissons_ratio": "1", "elastic_anisotropy": "1"}
    for name in (
        "bulk_modulus",
        "shear_modulus",
        "youngs_modulus",
        "poissons_ratio",
        "elastic_anisotropy",
    ):
        if result.mechanically_stable and name in result.moduli and name in result.moduli_std:
            add(name, result.moduli[name], result.moduli_std[name], units.get(name, "GPa"))
        else:
            out.append(
                PropertyEvidence(
                    structure_id=structure_id,
                    composition=composition,
                    property=name,
                    status=PropertyStatus.UNSUPPORTED,
                    unit=units.get(name, "GPa"),
                    conditions=cond,
                    reason=(
                        "mechanically unstable: isotropic moduli are not defined"
                        if not result.mechanically_stable
                        else "uncertainty across amplitudes is unavailable"
                    ),
                )
            )
    return out


__all__ = [
    "ElasticResult",
    "ElasticSettings",
    "apply_strain",
    "ase_stress_fn",
    "calculator_supports_stress",
    "compute_elastic_tensor",
    "elastic_property_evidence",
    "isotropic_moduli",
    "plan_calculations",
    "strain_set",
]
