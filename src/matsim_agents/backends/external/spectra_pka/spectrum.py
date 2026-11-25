"""Neutron spectrum provider: ingestion, validation, unit conversion, flux-conserving rebinning.

A spectrum is stored as group-integrated flux (n/cm2/s per group) on ascending energy
boundaries in eV, or as a single line (monoenergetic). Normalised spectra (``absolute`` false)
carry shape only; no absolute rate may be derived from them. SPECTRA-PKA itself needs the flux
on the energy grid of the PKA library, so ``rebin`` maps any valid spectrum onto that grid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

ENERGY_TO_EV = {"eV": 1.0, "keV": 1.0e3, "MeV": 1.0e6}
FLUX_UNIT_TO_CM2_S = {"n/cm2/s": 1.0, "n/m2/s": 1.0e-4}
ValueKind = Literal["integrated", "per_energy", "per_lethargy"]
SpectrumSource = Literal["monoenergetic", "reference", "file", "external_transport"]
CONTIGUITY_RTOL = 1.0e-9
CONSERVATION_RTOL = 1.0e-9


class SpectrumError(ValueError):
    """The spectrum is invalid, inconsistent, or cannot be converted without guessing."""


@dataclass(frozen=True)
class NeutronSpectrum:
    identity: str
    source: str
    group_flux: tuple[float, ...]  # n/cm2/s per group; shape-only (sum 1) when not absolute
    absolute: bool
    boundaries_eV: tuple[float, ...] = ()
    line_energy_eV: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_line(self) -> bool:
        return self.line_energy_eV is not None

    @property
    def total_flux(self) -> float:
        return float(sum(self.group_flux))

    def payload(self) -> dict[str, Any]:
        """Scientific content only (no paths, no timestamps); input to cache keys."""
        return {
            "boundaries_eV": list(self.boundaries_eV),
            "group_flux": list(self.group_flux),
            "line_energy_eV": self.line_energy_eV,
            "absolute": self.absolute,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def normalized(self) -> NeutronSpectrum:
        """Shape-only copy summing to one. Absolute rates are no longer derivable from it."""
        total = self.total_flux
        if total <= 0.0:
            raise SpectrumError("cannot normalise a spectrum with zero total flux")
        return _replace(self, group_flux=tuple(f / total for f in self.group_flux), absolute=False)

    def scaled_to(self, total_flux_cm2_s: float) -> NeutronSpectrum:
        """Give a shape-only or absolute spectrum an explicit absolute total flux."""
        if not math.isfinite(total_flux_cm2_s) or total_flux_cm2_s <= 0.0:
            raise SpectrumError("total flux must be positive and finite")
        base = self.normalized() if self.absolute else self
        total = base.total_flux
        flux = tuple(f / total * total_flux_cm2_s for f in base.group_flux)
        return _replace(self, group_flux=flux, absolute=True)


def _replace(spec: NeutronSpectrum, **changes: Any) -> NeutronSpectrum:
    data = {
        "identity": spec.identity,
        "source": spec.source,
        "group_flux": spec.group_flux,
        "absolute": spec.absolute,
        "boundaries_eV": spec.boundaries_eV,
        "line_energy_eV": spec.line_energy_eV,
        "detail": dict(spec.detail),
    }
    data.update(changes)
    return NeutronSpectrum(**data)


def validate_spectrum(spec: NeutronSpectrum) -> list[str]:
    """Return a list of problems; empty means valid."""
    problems: list[str] = []
    flux = np.asarray(spec.group_flux, dtype=float)
    if flux.size == 0:
        problems.append("no flux values")
    elif not np.all(np.isfinite(flux)):
        problems.append("flux contains NaN or infinity")
    elif np.any(flux < 0.0):
        problems.append("flux contains negative values")
    elif float(flux.sum()) <= 0.0:
        problems.append("total flux is zero")
    if spec.is_line:
        energy = spec.line_energy_eV
        assert energy is not None
        if not math.isfinite(energy) or energy <= 0.0:
            problems.append("line energy must be positive and finite")
        if spec.boundaries_eV:
            problems.append("a monoenergetic spectrum has no group boundaries")
        if flux.size != 1:
            problems.append("a monoenergetic spectrum has exactly one flux value")
        return problems
    edges = np.asarray(spec.boundaries_eV, dtype=float)
    if edges.size != flux.size + 1:
        problems.append(f"{edges.size} boundaries do not match {flux.size} flux values (+1)")
    elif not np.all(np.isfinite(edges)) or edges[0] <= 0.0:
        problems.append("boundaries must be finite and positive")
    elif np.any(np.diff(edges) <= 0.0):
        problems.append("boundaries are not strictly increasing")
    if not spec.absolute and flux.size and abs(float(flux.sum()) - 1.0) > 1e-9:
        problems.append("a normalised spectrum must sum to 1")
    return problems


def check_spectrum(spec: NeutronSpectrum) -> NeutronSpectrum:
    problems = validate_spectrum(spec)
    if problems:
        raise SpectrumError(f"spectrum {spec.identity!r}: " + "; ".join(problems))
    return spec


def monoenergetic(
    energy: float,
    *,
    energy_units: str = "MeV",
    total_flux: float | None = None,
    flux_units: str = "n/cm2/s",
    identity: str | None = None,
) -> NeutronSpectrum:
    """A line source. Without ``total_flux`` the spectrum is shape-only (no absolute rates)."""
    e_ev = energy * _factor(ENERGY_TO_EV, energy_units, "energy unit")
    absolute = total_flux is not None
    flux = 1.0
    if total_flux is not None:
        flux = total_flux * _factor(FLUX_UNIT_TO_CM2_S, flux_units, "flux unit")
    return check_spectrum(
        NeutronSpectrum(
            identity=identity or f"monoenergetic-{e_ev:.6g}eV",
            source="monoenergetic",
            group_flux=(float(flux),),
            absolute=absolute,
            line_energy_eV=float(e_ev),
        )
    )


def from_groups(
    boundaries: Sequence[float],
    values: Sequence[float],
    *,
    energy_units: str = "eV",
    flux_units: str = "n/cm2/s",
    value_kind: ValueKind = "integrated",
    absolute: bool = True,
    identity: str,
    source: str = "file",
    detail: dict[str, Any] | None = None,
) -> NeutronSpectrum:
    """Convert a group spectrum to group-integrated n/cm2/s on eV boundaries and validate it."""
    scale_e = _factor(ENERGY_TO_EV, energy_units, "energy unit")
    scale_f = _factor(FLUX_UNIT_TO_CM2_S, flux_units, "flux unit") if absolute else 1.0
    edges = np.asarray(boundaries, dtype=float) * scale_e
    vals = np.asarray(values, dtype=float)
    if edges.size != vals.size + 1:
        raise SpectrumError(
            f"spectrum {identity!r}: {edges.size} boundaries for {vals.size} values (need +1)"
        )
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0) or edges[0] <= 0.0:
        raise SpectrumError(f"spectrum {identity!r}: boundaries must be positive and increasing")
    if value_kind == "per_energy":
        vals = vals * np.diff(edges) / scale_e  # value is per input energy unit
    elif value_kind == "per_lethargy":
        vals = vals * np.log(edges[1:] / edges[:-1])
    elif value_kind != "integrated":
        raise SpectrumError(f"unknown value kind {value_kind!r}")
    vals = vals * scale_f
    total = float(vals.sum())
    if not absolute and np.all(np.isfinite(vals)) and np.all(vals >= 0.0) and total > 0.0:
        vals = vals / total  # shape-only input is stored summing to one
    return check_spectrum(
        NeutronSpectrum(
            identity=identity,
            source=source,
            group_flux=tuple(float(v) for v in vals),
            absolute=absolute,
            boundaries_eV=tuple(float(e) for e in edges),
            detail=dict(detail or {}),
        )
    )


def _factor(table: dict[str, float], unit: str, what: str) -> float:
    if unit not in table:
        raise SpectrumError(f"unsupported {what} {unit!r}; choose one of {sorted(table)}")
    return table[unit]


def rebin(
    spec: NeutronSpectrum,
    target_boundaries_eV: Sequence[float],
    *,
    interpolation: Literal["energy", "lethargy"] = "energy",
    max_lost_fraction: float = 0.0,
) -> tuple[NeutronSpectrum, float]:
    """Flux-conserving map onto ``target_boundaries_eV``.

    Flux is spread uniformly in energy (or in lethargy) inside each source group and summed
    over the overlap with each target group. Returns the new spectrum and the fraction of
    the total flux that fell outside the target range (that flux is dropped, never rescaled
    in, and the call fails if it exceeds ``max_lost_fraction``). A line spectrum goes into the
    target group containing its energy.
    """
    check_spectrum(spec)
    tgt = np.asarray(target_boundaries_eV, dtype=float)
    if tgt.ndim != 1 or tgt.size < 2 or not np.all(np.isfinite(tgt)) or tgt[0] <= 0.0:
        raise SpectrumError("target boundaries must be a finite, positive 1-D grid")
    if np.any(np.diff(tgt) <= 0.0):
        raise SpectrumError("target boundaries must be strictly increasing")
    out = np.zeros(tgt.size - 1)
    total = spec.total_flux
    if spec.is_line:
        energy = spec.line_energy_eV
        assert energy is not None
        idx = int(np.searchsorted(tgt, energy, side="right")) - 1
        lost = 1.0
        if 0 <= idx < out.size:
            out[idx] = spec.group_flux[0]
            lost = 0.0
    else:
        src = np.asarray(spec.boundaries_eV, dtype=float)
        flux = np.asarray(spec.group_flux, dtype=float)
        coord_src, coord_tgt = (
            (np.log(src), np.log(tgt))
            if interpolation == "lethargy"
            else (
                src,
                tgt,
            )
        )
        for i in range(flux.size):
            lo, hi = coord_src[i], coord_src[i + 1]
            overlap = np.clip(
                np.minimum(coord_tgt[1:], hi) - np.maximum(coord_tgt[:-1], lo), 0, None
            )
            out += flux[i] * overlap / (hi - lo)
        lost = max(0.0, 1.0 - float(out.sum()) / total) if total > 0 else 0.0
        if abs(lost) < CONSERVATION_RTOL:
            lost = 0.0
    if lost > 0.0 and not spec.absolute:
        raise SpectrumError(
            f"spectrum {spec.identity!r}: a shape-only spectrum cannot be rebinned with flux loss"
        )
    if lost > max_lost_fraction:
        raise SpectrumError(
            f"spectrum {spec.identity!r}: {lost:.3e} of the flux lies outside the library "
            f"energy range [{tgt[0]:.6g}, {tgt[-1]:.6g}] eV (allowed {max_lost_fraction:.3e})"
        )
    detail = dict(spec.detail)
    detail["rebinned_from_sha256"] = spec.sha256()
    detail["rebin_interpolation"] = interpolation
    detail["rebin_lost_flux_fraction"] = lost
    result = NeutronSpectrum(
        identity=spec.identity,
        source=spec.source,
        group_flux=tuple(float(v) for v in out),
        absolute=spec.absolute,
        boundaries_eV=tuple(float(e) for e in tgt),
        detail=detail,
    )
    return check_spectrum(result), lost


def _read_csv(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = [r for r in csv.reader(handle) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise SpectrumError(f"{path}: empty file")
    header = [c.strip().lower() for c in rows[0]]
    if header[:3] != ["e_low", "e_high", "flux"]:
        raise SpectrumError(f"{path}: header must start with e_low,e_high,flux (got {header[:3]})")
    lows: list[float] = []
    highs: list[float] = []
    vals: list[float] = []
    for n, row in enumerate(rows[1:], start=2):
        try:
            lo, hi, v = (float(x) for x in row[:3])
        except ValueError:
            raise SpectrumError(f"{path}: line {n}: non-numeric entry") from None
        lows.append(lo)
        highs.append(hi)
        vals.append(v)
    for i in range(len(lows) - 1):
        if not math.isclose(highs[i], lows[i + 1], rel_tol=CONTIGUITY_RTOL, abs_tol=0.0):
            raise SpectrumError(
                f"{path}: group {i + 1} ends at {highs[i]} but the next starts at {lows[i + 1]}; "
                "gaps are ambiguous (zero flux or missing data), so list explicit zero groups"
            )
    return [*lows, highs[-1]], vals


def load_spectrum_file(
    path: str | Path,
    *,
    fmt: Literal["csv", "json", "auto"] = "auto",
    identity: str | None = None,
    energy_units: str = "eV",
    flux_units: str = "n/cm2/s",
    value_kind: ValueKind = "integrated",
    absolute: bool = True,
) -> NeutronSpectrum:
    """Read a CSV (``e_low,e_high,flux``) or JSON group spectrum.

    JSON keys: ``energy_boundaries``, ``flux`` and optionally ``energy_units``, ``flux_units``,
    ``value_kind``, ``normalization`` ("absolute" or "normalized"), ``identity``. Keys in the
    file override the arguments. External transport tallies use ``load_transport_tally``.
    """
    source = Path(path)
    if not source.is_file():
        raise SpectrumError(f"spectrum file not found: {source}")
    kind = source.suffix.lower().lstrip(".") if fmt == "auto" else fmt
    meta: dict[str, Any] = {"file": source.name, "file_sha256": _file_sha256(source)}
    if kind == "csv":
        boundaries, values = _read_csv(source)
    elif kind == "json":
        try:
            doc = json.loads(source.read_text())
            boundaries, values = doc["energy_boundaries"], doc["flux"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SpectrumError(f"{source}: {exc!r}") from None
        energy_units = doc.get("energy_units", energy_units)
        flux_units = doc.get("flux_units", flux_units)
        value_kind = doc.get("value_kind", value_kind)
        absolute = doc.get("normalization", "absolute" if absolute else "normalized") == "absolute"
        identity = doc.get("identity", identity)
    else:
        raise SpectrumError(f"{source}: unsupported spectrum format {kind!r}")
    return from_groups(
        boundaries,
        values,
        energy_units=energy_units,
        flux_units=flux_units,
        value_kind=value_kind,
        absolute=absolute,
        identity=identity or source.stem,
        source="file",
        detail=meta,
    )


def load_transport_tally(
    path: str | Path,
    *,
    source_rate_per_s: float | None = None,
    volume_cm3: float | None = None,
    identity: str | None = None,
) -> NeutronSpectrum:
    """Ingest an externally computed flux tally (for example exported from OpenMC) as JSON.

    Keys: ``energy_boundaries`` (with ``energy_units``) and ``flux_per_source_particle`` (track
    length per source particle in cm, per group). The result is absolute only when both
    ``source_rate_per_s`` and ``volume_cm3`` are given; otherwise it is shape-only. No transport
    is run here.
    """
    source = Path(path)
    try:
        doc = json.loads(source.read_text())
        boundaries = doc["energy_boundaries"]
        tally = doc["flux_per_source_particle"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SpectrumError(f"{source}: {exc!r}") from None
    absolute = source_rate_per_s is not None and volume_cm3 is not None
    values = np.asarray(tally, dtype=float)
    if absolute:
        assert source_rate_per_s is not None and volume_cm3 is not None
        if source_rate_per_s <= 0 or volume_cm3 <= 0:
            raise SpectrumError("source rate and volume must be positive")
        values = values * source_rate_per_s / volume_cm3
    return from_groups(
        boundaries,
        values,
        energy_units=doc.get("energy_units", "eV"),
        absolute=absolute,
        identity=identity or doc.get("identity", source.stem),
        source="external_transport",
        detail={
            "file": source.name,
            "file_sha256": _file_sha256(source),
            "source_rate_per_s": source_rate_per_s,
            "volume_cm3": volume_cm3,
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "NeutronSpectrum",
    "SpectrumError",
    "check_spectrum",
    "from_groups",
    "load_spectrum_file",
    "load_transport_tally",
    "monoenergetic",
    "rebin",
    "validate_spectrum",
]
