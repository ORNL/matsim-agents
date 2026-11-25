"""Parse and classify converged collinear magnetic results from QE output.

The classification is descriptive: ``afm_like`` means opposite-sign local moments with small
net moment under the stated tolerances, never proof of an antiferromagnetic ground state, and a
near-zero total magnetization alone is never read as antiferromagnetism.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms

from matsim_agents.science.magnetism.config import MagneticConfiguration

MagneticClassification = Literal[
    "nonmagnetic",
    "fm_like",
    "afm_like",
    "ferrimagnetic_like",
    "mixed_unclassified",
    "insufficient_evidence",
    "not_converged",
]
MomentMethod = Literal["qe_integrated_site", "projwfc_lowdin", "external"]

_TOTAL_RE = re.compile(r"^\s*total magnetization\s*=\s*(-?\d+\.\d+)\s+Bohr mag/cell", re.M)
_ABS_RE = re.compile(r"^\s*absolute magnetization\s*=\s*(-?\d+\.\d+)\s+Bohr mag/cell", re.M)
_SITE_RE = re.compile(r"^\s*atom:\s*(\d+)\s+charge:\s*(-?\d+\.\d+)\s+magn:\s*(-?\d+\.\d+)", re.M)
_CONVERGED_RE = re.compile(r"convergence has been achieved in\s+\d+\s+iterations")
_NOT_CONVERGED = "convergence NOT achieved"


@dataclass(frozen=True)
class ClassificationTolerances:
    """All thresholds in muB; the defaults are conventions, not physical derivations."""

    nonmagnetic_moment: float = 0.1  # all |m| below this -> nonmagnetic
    site_moment: float = 0.1  # sites below this carry no sign information
    afm_net_fraction: float = 0.1  # |net| / sum|m| at or below this -> afm_like
    fm_sign_fraction: float = 0.95  # fraction of |m| in one sign for fm_like


@dataclass(frozen=True)
class MagneticResult:
    converged: bool
    total_magnetization_muB: float | None
    absolute_magnetization_muB: float | None
    local_moments_muB: tuple[float, ...]
    atom_ids: tuple[int, ...]
    moment_method: MomentMethod
    notes: tuple[str, ...] = ()
    spin_polarized: bool = field(default=True)


def parse_qe_magnetic_output(
    text: str, atoms: Atoms, *, spin_polarized: bool = True
) -> MagneticResult:
    """Extract the final magnetization block of a pw.x output.

    ``atoms`` supplies the stable ``atom_id`` of each QE site (QE lists sites in input order).
    """
    notes: list[str] = []
    converged = bool(_CONVERGED_RE.search(text)) and _NOT_CONVERGED not in text
    if not converged:
        notes.append("SCF convergence was not confirmed")
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    ids = tuple(int(i) for i in atoms.get_array("atom_id"))

    totals = _TOTAL_RE.findall(text)
    absolutes = _ABS_RE.findall(text)
    total = float(totals[-1]) if totals else None
    absolute = float(absolutes[-1]) if absolutes else None

    local: tuple[float, ...] = ()
    sites = _SITE_RE.findall(text)
    if sites:
        block = _last_block(sites, len(ids))
        if block is None:
            notes.append(f"final site block does not list all {len(ids)} atoms")
        else:
            local = tuple(float(m) for _, _, m in block)
    elif spin_polarized:
        notes.append("no per-site magnetic moments in output (verbosity or code version)")
    return MagneticResult(
        converged=converged,
        total_magnetization_muB=total,
        absolute_magnetization_muB=absolute,
        local_moments_muB=local,
        atom_ids=ids if local else (),
        moment_method="qe_integrated_site",
        notes=tuple(notes),
        spin_polarized=spin_polarized,
    )


def _last_block(sites: list[tuple[str, str, str]], nat: int) -> list[tuple[str, str, str]] | None:
    """The last run of ``nat`` consecutive sites numbered 1..nat."""
    best = None
    start = 0
    for index, (number, _, _) in enumerate(sites):
        if int(number) == 1:
            start = index
        if int(number) == nat and index - start + 1 == nat:
            best = sites[start : index + 1]
    return best


def parse_qe_magnetic_file(path: str | Path, atoms: Atoms, **kw) -> MagneticResult:
    return parse_qe_magnetic_output(Path(path).read_text(errors="replace"), atoms, **kw)


def result_from_moments(
    moments_muB: Sequence[float],
    atom_ids: Sequence[int],
    *,
    method: MomentMethod = "external",
    converged: bool = True,
) -> MagneticResult:
    """Wrap moments obtained elsewhere, e.g. a projwfc Lowdin analysis."""
    if len(moments_muB) != len(atom_ids):
        raise ValueError("moments and atom_ids must have the same length")
    values = tuple(float(m) for m in moments_muB)
    return MagneticResult(
        converged=converged,
        total_magnetization_muB=float(sum(values)),
        absolute_magnetization_muB=float(sum(abs(v) for v in values)),
        local_moments_muB=values,
        atom_ids=tuple(int(i) for i in atom_ids),
        moment_method=method,
    )


def classify_moments(
    result: MagneticResult, tol: ClassificationTolerances | None = None
) -> MagneticClassification:
    tol = tol or ClassificationTolerances()
    if not result.converged:
        return "not_converged"
    if not result.local_moments_muB:
        return "insufficient_evidence"
    m = np.asarray(result.local_moments_muB, dtype=float)
    if np.all(np.abs(m) < tol.nonmagnetic_moment):
        return "nonmagnetic"
    sig = m[np.abs(m) >= tol.site_moment]
    if sig.size == 0:
        return "nonmagnetic"
    pos = float(np.abs(sig[sig > 0]).sum())
    neg = float(np.abs(sig[sig < 0]).sum())
    total_abs = pos + neg
    if max(pos, neg) / total_abs >= tol.fm_sign_fraction:
        return "fm_like"
    net = abs(pos - neg) / total_abs
    if pos > 0 and neg > 0:
        return "afm_like" if net <= tol.afm_net_fraction else "ferrimagnetic_like"
    return "mixed_unclassified"  # pragma: no cover - covered by the branches above


@dataclass(frozen=True)
class StateComparison:
    classification: MagneticClassification
    requested_mode: str
    relation: Literal[
        "same_arrangement",
        "same_up_to_global_reversal",
        "collapsed_to_nonmagnetic",
        "different_arrangement",
        "indeterminate",
    ]
    notes: tuple[str, ...] = ()


def compare_with_request(
    config: MagneticConfiguration,
    result: MagneticResult,
    tol: ClassificationTolerances | None = None,
) -> StateComparison:
    """Compare converged sign pattern with the requested initial pattern."""
    tol = tol or ClassificationTolerances()
    cls = classify_moments(result, tol)
    if cls in ("not_converged", "insufficient_evidence"):
        return StateComparison(cls, config.mode, "indeterminate", result.notes)
    if not config.spin_polarized:
        relation = "same_arrangement" if cls == "nonmagnetic" else "different_arrangement"
        return StateComparison(cls, config.mode, relation)
    if cls == "nonmagnetic":
        return StateComparison(cls, config.mode, "collapsed_to_nonmagnetic")
    requested = dict(zip(config.atom_ids, config.initial_moments_muB or (), strict=False))
    got = dict(zip(result.atom_ids, result.local_moments_muB, strict=True))
    if set(requested) != set(got):
        return StateComparison(cls, config.mode, "indeterminate", ("atom_ids differ",))
    a = np.array([np.sign(requested[i]) for i in sorted(got)])
    b = np.array([np.sign(got[i]) if abs(got[i]) >= tol.site_moment else 0.0 for i in sorted(got)])
    if np.array_equal(a, b):
        return StateComparison(cls, config.mode, "same_arrangement")
    if np.array_equal(a, -b):
        return StateComparison(cls, config.mode, "same_up_to_global_reversal")
    return StateComparison(cls, config.mode, "different_arrangement")


def spin_reversed_equivalent(
    a: MagneticResult, b: MagneticResult, *, atol_muB: float = 0.05
) -> bool:
    """True when ``b`` equals ``a`` or ``-a`` per atom_id within ``atol_muB``."""
    if set(a.atom_ids) != set(b.atom_ids) or not a.atom_ids:
        return False
    ma = dict(zip(a.atom_ids, a.local_moments_muB, strict=True))
    mb = dict(zip(b.atom_ids, b.local_moments_muB, strict=True))
    ids = sorted(ma)
    x = np.array([ma[i] for i in ids])
    y = np.array([mb[i] for i in ids])
    return bool(np.allclose(x, y, atol=atol_muB) or np.allclose(x, -y, atol=atol_muB))
