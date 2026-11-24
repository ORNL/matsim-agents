"""Map a QE NEB output parse onto the discovery ``NebResult`` contract.

Lives in ``workflows`` so the DFT backend does not import discovery types. It never yields
``qualified``: qualification needs relaxed endpoints and sensitivity checks the parser lacks.
"""

from __future__ import annotations

from matsim_agents.backends.dft.qe_neb_parse import NebOutputParse
from matsim_agents.discovery.neb_contracts import MIN_IMAGES, NebResult, ResultState

_STATE: dict[str, ResultState] = {
    "converged": "converged_unqualified",
    "iteration_limit": "unconverged",
    "electronic_failure": "failed",
    "process_failure": "failed",
    "invalid_numbers": "failed",
    "incomplete_output": "incomplete",
    "missing_images": "incomplete",
    "inconsistent_output": "incomplete",
}


def to_neb_result(
    parse: NebOutputParse,
    *,
    method_id: str,
    n_images: int,
    endpoints_qualified: bool = False,
    provenance: dict[str, str] | None = None,
) -> NebResult:
    """Map a parse onto ``NebResult``; never ``qualified``, never energies for failed runs."""
    state = _STATE[parse.outcome]
    carries = state in ("unconverged", "converged_unqualified") and parse.has_energies
    if carries and len(parse.energies_eV) != n_images:
        raise ValueError(f"parse has {len(parse.energies_eV)} images, expected {n_images}")
    meta = {"parser_outcome": parse.outcome, **(provenance or {})}
    if parse.iterations_completed is not None:
        meta["iterations_completed"] = str(parse.iterations_completed)
    return NebResult(
        state=state,
        method_id=method_id,
        n_images=max(n_images, MIN_IMAGES),
        relative_energies_eV=parse.relative_energies_eV if carries else (),
        forward_barrier_eV=parse.forward_barrier_eV if carries else None,
        reverse_barrier_eV=parse.reverse_barrier_eV if carries else None,
        endpoints_qualified=bool(endpoints_qualified and carries),
        climbing_image_used=parse.climbing_image_used,
        warnings=parse.warnings,
        provenance=meta,
    )


__all__ = ["to_neb_result"]
