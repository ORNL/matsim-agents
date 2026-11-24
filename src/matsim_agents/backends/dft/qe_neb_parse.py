"""Parse ``neb.x`` output into an outcome, energies and barriers.

Formats follow the QE sources (``NEB/src/path_formats.f90``, ``path_io_routines.f90``,
``path_base.f90``):

* stdout holds, per iteration, ``activation energy (->) / (<-) = x eV`` and a table
  ``image  energy (eV)  error (eV/A)  frozen`` (absolute energies, error in eV/Angstrom);
* ``neb: convergence achieved in N iterations`` or ``neb: reached the maximum number of steps``
  ends the optimisation;
* ``prefix.dat`` has three columns per image: normalised reaction coordinate, energy relative to
  the first image in eV, and the error in eV/Angstrom.

Rules kept here:

* Nothing is invented. A missing, truncated or non-finite table gives no energies or barriers.
* Exit code 0 is not convergence; the convergence message must be present, and for a converged
  run the interior image errors must also respect the requested threshold.
* The barrier is ``max(E_i) - E_end`` over the *sampled* images. It is an estimate until a
  saddle is refined; it is withheld when the maximum sits at an endpoint.
* ``to_neb_result`` never yields ``qualified``: qualification needs evidence this parser does
  not have (relaxed endpoints, sensitivity checks), and is decided by the workflow.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from matsim_agents.backends.dft.qe_neb_run import NebRunRecord
from matsim_agents.discovery.neb_contracts import MIN_IMAGES, NebResult, ResultState

Outcome = Literal[
    "converged",
    "iteration_limit",
    "electronic_failure",
    "process_failure",
    "incomplete_output",
    "missing_images",
    "invalid_numbers",
    "inconsistent_output",
]

# Reported activation energies are printed with F10.6, table energies with F15.7.
CROSS_CHECK_TOL_EV = 1e-4
_THRESHOLD_SLACK = 1e-6

_ITERATION = re.compile(r"-{5,}\s*iteration\s+(\d+)\s*-{5,}")
_ACT_FORWARD = re.compile(r"activation energy \(->\)\s*=\s*(\S+)\s*eV")
_ACT_REVERSE = re.compile(r"activation energy \(<-\)\s*=\s*(\S+)\s*eV")
_TABLE_HEADER = re.compile(r"image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen")
_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+([TF])\s*$")
_CI_AUTO = re.compile(r"climbing image\s*=\s*(\d+)")
_CI_MANUAL = re.compile(r"list of climbing images\s*:\s*(.*)$")
_CONVERGED = re.compile(r"neb: convergence achieved in\s+(\d+)\s+iterations")
_MAX_STEPS = re.compile(r"neb: reached the maximum number of\s+steps")
_ELECTRONIC = re.compile(r"convergence NOT achieved", re.IGNORECASE)
_ROUTINE_ERROR = re.compile(r"Error in routine\s+\S+")


class NebOutputParse(BaseModel):
    """What the output of one ``neb.x`` run says, with explicit uncertainty."""

    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    iterations_completed: int | None = None
    image_indices: tuple[int, ...] = ()
    energies_eV: tuple[float, ...] = ()
    relative_energies_eV: tuple[float, ...] = ()
    errors_eV_per_A: tuple[float, ...] = ()
    frozen: tuple[bool, ...] = ()
    max_interior_error_eV_per_A: float | None = None
    path_thr_ev_ang: float | None = None
    forward_barrier_eV: float | None = None
    reverse_barrier_eV: float | None = None
    highest_image_index: int | None = None
    endpoint_energy_difference_eV: float | None = None
    climbing_image_used: bool = False
    climbing_images: tuple[int, ...] = ()
    reported_forward_eV: float | None = None
    reported_reverse_eV: float | None = None
    dat_cross_checked: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def has_energies(self) -> bool:
        return bool(self.energies_eV)


def _finite(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_neb_dat(
    path: str | Path,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Read ``prefix.dat``: reaction coordinate, relative energy (eV), error (eV/A).

    Raises ``ValueError`` for malformed, non-finite or empty content.
    """
    coordinate: list[float] = []
    energy: list[float] = []
    error: list[float] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        values = [_finite(f) for f in fields]
        if len(fields) != 3 or any(v is None for v in values):
            raise ValueError(f"{path}:{number}: expected three finite numbers, got {line!r}")
        coordinate.append(values[0])  # type: ignore[arg-type]
        energy.append(values[1])  # type: ignore[arg-type]
        error.append(values[2])  # type: ignore[arg-type]
    if not coordinate:
        raise ValueError(f"{path} has no data")
    return tuple(coordinate), tuple(energy), tuple(error)


def _last_activation_line(lines: list[str]) -> int | None:
    starts = [i for i, line in enumerate(lines) if _ACT_FORWARD.search(line)]
    return starts[-1] if starts else None


def _process_failed(record: NebRunRecord | None, exit_code: int | None) -> str | None:
    if record is not None:
        if record.status in ("not_started", "timed_out", "exited_nonzero"):
            return f"process {record.status} (exit code {record.exit_code})"
        return None
    if exit_code not in (None, 0):
        return f"process exited with code {exit_code}"
    return None


def parse_neb_output(
    source: str | Path,
    *,
    record: NebRunRecord | None = None,
    exit_code: int | None = None,
    expected_images: int | None = None,
    path_thr_ev_ang: float | None = None,
    dat_path: str | Path | None = None,
) -> NebOutputParse:
    """Classify a ``neb.x`` run from its stdout text (or a path to it).

    ``record`` (from ``run_neb``) or ``exit_code`` tell whether the process itself failed;
    ``expected_images`` and ``path_thr_ev_ang`` enable the image-count and threshold checks.
    """
    text = Path(source).read_text(errors="replace") if isinstance(source, Path) else source
    lines = text.splitlines()
    warnings: list[str] = []
    failure = _process_failed(record, exit_code)

    iteration_marks = [
        (i, int(m.group(1))) for i, line in enumerate(lines) if (m := _ITERATION.search(line))
    ]
    conv = next((m for line in lines if (m := _CONVERGED.search(line))), None)
    limit = any(_MAX_STEPS.search(line) for line in lines)
    electronic = any(_ELECTRONIC.search(line) for line in lines)
    routine = next((m.group(0) for line in lines if (m := _ROUTINE_ERROR.search(line))), None)
    if routine:
        warnings.append(f"output reports: {routine}")

    def done(outcome: Outcome, **fields) -> NebOutputParse:
        return NebOutputParse(outcome=outcome, warnings=tuple(warnings), **fields)

    start = _last_activation_line(lines)
    if start is None:
        if electronic:
            warnings.append("electronic convergence failed before any NEB table was written")
            return done("electronic_failure")
        if failure or routine:
            if failure:
                warnings.append(failure)
            return done("process_failure")
        warnings.append("no NEB energy table found in the output")
        return done("incomplete_output")

    forward = _ACT_FORWARD.search(lines[start])
    reported_forward = _finite(forward.group(1)) if forward else None
    reported_reverse = None
    header = None
    for i in range(start + 1, min(start + 6, len(lines))):
        reverse = _ACT_REVERSE.search(lines[i])
        if reverse:
            reported_reverse = _finite(reverse.group(1))
        if _TABLE_HEADER.search(lines[i]):
            header = i
            break
    rows: list[tuple[int, float | None, float | None, bool]] = []
    if header is not None:
        for line in lines[header + 1 :]:
            if not line.strip():
                if rows:
                    break
                continue
            match = _ROW.match(line)
            if not match:
                if rows:
                    break
                continue
            rows.append(
                (
                    int(match.group(1)),
                    _finite(match.group(2)),
                    _finite(match.group(3)),
                    match.group(4) == "T",
                )
            )
    if header is None or not rows:
        warnings.append("the last NEB iteration has no readable image table")
        return done("incomplete_output")

    last_iteration = max((n for i, n in iteration_marks if i < start), default=None)
    if iteration_marks and iteration_marks[-1][0] > start:
        warnings.append(f"run stopped during iteration {iteration_marks[-1][1]}; table is stale")

    indices = tuple(r[0] for r in rows)
    common = {"iterations_completed": last_iteration, "path_thr_ev_ang": path_thr_ev_ang}
    if any(r[1] is None or r[2] is None for r in rows):
        warnings.append("non-finite or unreadable number in the image table")
        return done("invalid_numbers", image_indices=indices, **common)
    if indices != tuple(range(1, len(rows) + 1)):
        warnings.append(f"image indices are not 1..N contiguous: {list(indices)}")
        return done("missing_images", image_indices=indices, **common)
    if expected_images is not None and len(rows) != expected_images:
        warnings.append(f"expected {expected_images} images, table has {len(rows)}")
        return done("missing_images", image_indices=indices, **common)

    energies = tuple(r[1] for r in rows)  # type: ignore[misc]
    errors = tuple(r[2] for r in rows)  # type: ignore[misc]
    frozen = tuple(r[3] for r in rows)
    relative = tuple(e - energies[0] for e in energies)
    interior = errors[1:-1]
    max_interior = max(interior) if interior else None

    ci_hits = [int(m.group(1)) for line in lines if (m := _CI_AUTO.search(line))]
    ci_auto = ci_hits[-1] if ci_hits else None
    ci_manual: list[int] = []
    for line in lines:
        manual = _CI_MANUAL.search(line)
        if manual:
            ci_manual = [int(t) for t in re.findall(r"\d+", manual.group(1))]
    climbing = tuple(sorted(set(ci_manual + ([ci_auto] if ci_auto is not None else []))))

    highest = max(range(len(energies)), key=energies.__getitem__) + 1
    forward_barrier: float | None = None
    reverse_barrier: float | None = None
    if highest in (1, len(energies)):
        warnings.append(
            "highest sampled image is an endpoint; no barrier is reported (path or endpoints "
            "need inspection)"
        )
    else:
        top = energies[highest - 1]
        forward_barrier = top - energies[0]
        reverse_barrier = top - energies[-1]
        if (
            reported_forward is not None
            and abs(reported_forward - forward_barrier) > CROSS_CHECK_TOL_EV
        ):
            warnings.append(
                f"reported forward activation energy {reported_forward} differs from the table "
                f"value {forward_barrier:.6f}"
            )
        if (
            reported_reverse is not None
            and abs(reported_reverse - reverse_barrier) > CROSS_CHECK_TOL_EV
        ):
            warnings.append(
                f"reported reverse activation energy {reported_reverse} differs from the table "
                f"value {reverse_barrier:.6f}"
            )
        warnings.append("barrier is the sampled maximum, an estimate and not a refined saddle")

    cross_checked = False
    if dat_path is not None:
        try:
            dat_energy = parse_neb_dat(dat_path)[1]
        except (OSError, ValueError) as exc:
            warnings.append(f"prefix.dat unusable: {exc}")
        else:
            if len(dat_energy) != len(energies):
                warnings.append("prefix.dat image count differs from the output table")
            elif any(abs(a - b) > 1e-4 for a, b in zip(dat_energy, relative, strict=True)):
                warnings.append("prefix.dat energies differ from the output table")
            else:
                cross_checked = True

    fields = {
        **common,
        "image_indices": indices,
        "energies_eV": energies,
        "relative_energies_eV": relative,
        "errors_eV_per_A": errors,
        "frozen": frozen,
        "max_interior_error_eV_per_A": max_interior,
        "forward_barrier_eV": forward_barrier,
        "reverse_barrier_eV": reverse_barrier,
        "highest_image_index": highest,
        "endpoint_energy_difference_eV": energies[-1] - energies[0],
        "climbing_image_used": bool(climbing),
        "climbing_images": climbing,
        "reported_forward_eV": reported_forward,
        "reported_reverse_eV": reported_reverse,
        "dat_cross_checked": cross_checked,
    }

    if conv:
        fields["iterations_completed"] = int(conv.group(1))
        if failure:
            warnings.append(f"convergence message present but {failure}")
            return done("process_failure", **fields)
        if (
            path_thr_ev_ang is not None
            and max_interior is not None
            and max_interior > path_thr_ev_ang + _THRESHOLD_SLACK
        ):
            warnings.append(
                f"convergence message present but the largest interior error {max_interior} "
                f"exceeds path_thr {path_thr_ev_ang}"
            )
            return done("inconsistent_output", **fields)
        return done("converged", **fields)
    if limit:
        return done("iteration_limit", **fields)
    if electronic:
        warnings.append("electronic convergence failed during the run")
        return done("electronic_failure", **fields)
    if failure:
        warnings.append(failure)
        return done("process_failure", **fields)
    warnings.append("no convergence or iteration-limit message; the output is incomplete")
    return done("incomplete_output", **fields)


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


__all__ = [
    "NebOutputParse",
    "Outcome",
    "parse_neb_dat",
    "parse_neb_output",
    "to_neb_result",
]
