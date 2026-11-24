"""Pathway validation and qualification states for NEB transitions.

``assess_pathway`` never raises for a bad pathway: it records every failed check and returns
``rejected``. States, in order of evidence:

``candidate``                  nothing checked yet (no images supplied)
``endpoint_relaxation_pending`` geometry is consistent but an endpoint is only ``generated``
``geometry_validated``         geometry is consistent and endpoints are relaxed, not yet qualified
``numerically_qualified``      geometry consistent, both endpoints numerically qualified with one
                               method identity and recorded provenance
``rejected``                   at least one check failed

Unrelaxed endpoints can never reach ``numerically_qualified``. This module judges setup quality
only. It is not a statement about barriers, mechanisms or physical correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from pydantic import BaseModel, ConfigDict

from matsim_agents.discovery.neb_contracts import MIN_IMAGES, NebTransition
from matsim_agents.discovery.neb_images import DEFAULT_MAX_STEP, DEFAULT_MIN_DISTANCE
from matsim_agents.discovery.neb_paths import correspond_atoms, minimum_image
from matsim_agents.discovery.point_defects import geometry_fingerprint

PathwayState = Literal[
    "candidate",
    "geometry_validated",
    "endpoint_relaxation_pending",
    "numerically_qualified",
    "rejected",
]


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""


class PathwayAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: PathwayState
    checks: tuple[CheckResult, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)


def _guard(name: str, fn) -> CheckResult:
    try:
        detail = fn()
    except ValueError as exc:
        return CheckResult(name=name, passed=False, detail=str(exc))
    return CheckResult(name=name, passed=True, detail=detail or "")


def _same_endpoint(name: str, atoms: Atoms, endpoint) -> CheckResult:
    def check() -> str:
        if "atom_id" not in atoms.arrays:
            raise ValueError(f"{name} structure has no atom_id array")
        if tuple(int(i) for i in atoms.get_array("atom_id")) != endpoint.atom_ids:
            raise ValueError(f"{name} atom identities differ from the endpoint record")
        if tuple(atoms.get_chemical_symbols()) != endpoint.species:
            raise ValueError(f"{name} species differ from the endpoint record")
        if geometry_fingerprint(atoms) != endpoint.geometry_fingerprint:
            raise ValueError(f"{name} geometry differs from the endpoint fingerprint")
        return ""

    return _guard(f"{name}_matches_record", check)


def _image_checks(
    images: Sequence[Atoms],
    initial: Atoms,
    final: Atoms,
    transition: NebTransition,
    min_distance: float,
    max_step: float,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    def identities() -> str:
        for name, atoms in ("initial", initial), ("final", final):
            if "atom_id" not in atoms.arrays:
                raise ValueError(f"{name} structure has no atom_id array")
        return ""

    checks.append(_guard("endpoint_identities", identities))
    if not checks[-1].passed:
        return checks

    def count() -> str:
        if len(images) < MIN_IMAGES:
            raise ValueError(f"{len(images)} images; at least {MIN_IMAGES} are required")
        calc = transition.calculation
        if calc is not None and calc.n_images != len(images):
            raise ValueError(f"{len(images)} images but the calculation spec has {calc.n_images}")
        return f"{len(images)} images"

    checks.append(_guard("image_count", count))
    if not images:
        return checks

    def consistency() -> str:
        ids = initial.get_array("atom_id")
        for k, image in enumerate(images):
            if "atom_id" not in image.arrays or not np.array_equal(image.get_array("atom_id"), ids):
                raise ValueError(f"image {k} identities differ from the initial structure")
            if image.get_chemical_symbols() != initial.get_chemical_symbols():
                raise ValueError(f"image {k} species differ from the initial structure")
            if not np.allclose(image.cell.array, initial.cell.array, rtol=0.0, atol=1e-6):
                raise ValueError(f"image {k} cell differs from the initial structure")
            if not np.isfinite(image.positions).all():
                raise ValueError(f"image {k} has non-finite coordinates")
        return ""

    checks.append(_guard("image_identity_species_cell", consistency))

    def endpoints_fixed() -> str:
        if not np.array_equal(images[0].positions, initial.positions):
            raise ValueError("first image is not the initial structure")
        order = {int(a): k for k, a in enumerate(final.get_array("atom_id"))}
        index = [order[int(a)] for a in initial.get_array("atom_id")]
        offset = (images[-1].positions - final.positions[index]) @ np.linalg.inv(initial.cell.array)
        if not np.allclose(offset, np.round(offset), atol=1e-6):
            raise ValueError("last image is not the final structure modulo lattice vectors")
        return ""

    checks.append(_guard("endpoints_fixed", endpoints_fixed))

    def continuity() -> str:
        worst = 0.0
        for k in range(len(images) - 1):
            step = images[k + 1].positions - images[k].positions
            _, lengths, _ = minimum_image(step, initial.cell.array)
            raw = np.linalg.norm(step, axis=1)
            if not np.allclose(lengths, raw, atol=1e-6):
                raise ValueError(f"images {k} and {k + 1} are discontinuous (periodic jump)")
            worst = max(worst, float(raw.max()))
        if worst > max_step:
            raise ValueError(f"largest per-image step {worst:.3f} A exceeds {max_step:.3f} A")
        return f"largest step {worst:.3f} A"

    checks.append(_guard("continuity", continuity))

    def overlaps() -> str:
        for k, image in enumerate(images):
            close = neighbor_list("d", image, min_distance)
            if len(close):
                raise ValueError(f"image {k} has a pair closer than {min_distance:.3f} A")
        return ""

    checks.append(_guard("no_overlaps", overlaps))
    return checks


def assess_pathway(
    transition: NebTransition,
    initial: Atoms,
    final: Atoms,
    images: Sequence[Atoms] | None = None,
    *,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    max_step: float = DEFAULT_MAX_STEP,
) -> PathwayAssessment:
    """Check a transition and its images, returning the highest justified state.

    Without ``images`` nothing geometric is verified and the state is ``candidate``.
    """
    if images is None:
        return PathwayAssessment(
            state="candidate", reasons=("no images supplied; nothing has been validated",)
        )
    checks: list[CheckResult] = [
        _same_endpoint("initial", initial, transition.initial),
        _same_endpoint("final", final, transition.final),
    ]

    def correspondence() -> str:
        if correspond_atoms(initial, final) != transition.correspondence:
            raise ValueError("correspondence does not match the structures")
        return ""

    checks.append(_guard("correspondence", correspondence))
    checks.extend(_image_checks(images, initial, final, transition, min_distance, max_step))

    methods = {e.method for e in (transition.initial, transition.final) if e.method}

    def method_consistency() -> str:
        if len(methods) > 1:
            raise ValueError(f"endpoints use different methods: {sorted(methods)}")
        calc = transition.calculation
        if calc is not None and methods and methods != {calc.method_id}:
            raise ValueError(f"calculation method {calc.method_id!r} differs from the endpoints'")
        return ""

    checks.append(_guard("method_consistency", method_consistency))

    statuses = {transition.initial.status, transition.final.status}
    if "generated" in statuses:
        # Both endpoints must be relaxed, or none of them is comparable; a mixed pair is flagged.
        mixed = statuses != {"generated"}
        checks.append(
            CheckResult(
                name="endpoint_relaxation_consistent",
                passed=not mixed,
                detail="one endpoint relaxed and the other not" if mixed else "",
            )
        )

    failed = [c for c in checks if not c.passed]
    if failed:
        return PathwayAssessment(
            state="rejected",
            checks=tuple(checks),
            reasons=tuple(f"{c.name}: {c.detail}" for c in failed),
        )
    if "generated" in statuses:
        return PathwayAssessment(
            state="endpoint_relaxation_pending",
            checks=tuple(checks),
            reasons=("endpoints are generated, not relaxed; barriers would not be meaningful",),
        )
    if statuses == {"numerically_qualified"}:
        missing = [e.role for e in (transition.initial, transition.final) if not e.provenance]
        if missing:
            return PathwayAssessment(
                state="geometry_validated",
                checks=tuple(checks),
                reasons=(f"qualified endpoints lack provenance: {missing}",),
            )
        return PathwayAssessment(state="numerically_qualified", checks=tuple(checks))
    return PathwayAssessment(
        state="geometry_validated",
        checks=tuple(checks),
        reasons=("endpoints are relaxed but not both numerically qualified",),
    )


__all__ = [
    "CheckResult",
    "PathwayAssessment",
    "PathwayState",
    "assess_pathway",
]
