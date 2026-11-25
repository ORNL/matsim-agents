"""Elemental to isotopic composition. Elemental and isotopic fractions are not interchangeable.

Atom fractions only. Natural abundances come from the ``mendeleev`` package (NIST/IUPAC
values, percent); the package name and version are recorded in every resolution. Explicit
per-element overrides and a full isotope inventory (for example from a depletion code)
replace natural abundance, and the resolution says which was used for each element.
"""

from __future__ import annotations

import importlib.metadata
import math
import re
from dataclasses import dataclass
from functools import cache

ABUNDANCE_TOLERANCE = 5.0e-3  # natural abundances must sum to 1 within this
FRACTION_TOLERANCE = 1.0e-6
_ELEMENT = re.compile(r"^[A-Z][a-z]?$")
_NUCLIDE = re.compile(r"^([A-Z][a-z]?)-?(\d{1,3})$")


class IsotopeError(ValueError):
    """The composition cannot be resolved to isotopes without guessing."""


@dataclass(frozen=True)
class IsotopeResolution:
    """Atom fractions of each isotope in the material (they sum to 1)."""

    isotope_fractions: dict[tuple[str, int], float]
    element_fractions: dict[str, float]
    element_basis: dict[str, str]  # element -> natural | override | inventory
    abundance_source: str

    def isotopic_composition(self) -> dict[str, dict[int, float]]:
        """Isotope fractions within each element (the IrradiationScenario convention)."""
        out: dict[str, dict[int, float]] = {}
        for (element, mass), fraction in sorted(self.isotope_fractions.items()):
            out.setdefault(element, {})[mass] = fraction / self.element_fractions[element]
        return out

    def payload(self) -> dict[str, object]:
        """Scientific content for cache keys."""
        return {
            "isotopes": {f"{e}{a}": f for (e, a), f in sorted(self.isotope_fractions.items())},
            "basis": dict(sorted(self.element_basis.items())),
            "abundance_source": self.abundance_source,
        }


def abundance_source() -> str:
    try:
        version = importlib.metadata.version("mendeleev")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
    return f"mendeleev {version} (NIST/IUPAC natural abundances)"


@cache
def natural_abundance(element: str) -> dict[int, float]:
    """Natural isotope fractions of ``element`` summing to one (from percent in the source)."""
    if not _ELEMENT.match(element):
        raise IsotopeError(f"{element!r} is not an element symbol")
    try:
        from mendeleev import element as _element
    except ImportError:
        raise IsotopeError(
            "natural abundances need the 'mendeleev' package (pip install 'matsim-agents"
            "[spectra-pka]'); alternatively give explicit isotopic overrides"
        ) from None
    try:
        isotopes = _element(element).isotopes
    except Exception as exc:
        raise IsotopeError(f"unknown element {element!r}: {exc}") from None
    percent = {i.mass_number: float(i.abundance) for i in isotopes if i.abundance}
    total = sum(percent.values())
    if not percent or abs(total - 100.0) > 100.0 * ABUNDANCE_TOLERANCE:
        raise IsotopeError(
            f"{element} has no usable natural abundance (sum {total:.4g} %); "
            "give explicit isotopic fractions"
        )
    return {mass: p / total for mass, p in sorted(percent.items())}


def parse_nuclide(label: str) -> tuple[str, int]:
    match = _NUCLIDE.match(label.strip())
    if not match:
        raise IsotopeError(f"{label!r} is not a nuclide label such as W184 or W-184")
    return match.group(1), int(match.group(2))


def _check_fractions(name: str, fractions: dict[object, float]) -> None:
    for key, value in fractions.items():
        if not math.isfinite(value) or value < 0.0:
            raise IsotopeError(f"{name}: fraction of {key} must be finite and non-negative")
    if abs(sum(fractions.values()) - 1.0) > FRACTION_TOLERANCE:
        raise IsotopeError(f"{name}: fractions sum to {sum(fractions.values())}, not 1")


def resolve_isotopes(
    composition: dict[str, float],
    *,
    overrides: dict[str, dict[int, float]] | None = None,
    inventory: dict[str, float] | None = None,
) -> IsotopeResolution:
    """Resolve atom fractions to isotopes.

    ``composition``: element -> atom fraction (sums to 1), covering every element that is not
    in ``inventory``. ``overrides``: element -> {mass number: fraction within the element}
    (sums to 1). ``inventory``: nuclide label (``W184``) -> atom fraction in the material; the
    elements it names take their elemental fractions from it, so it must not also appear in
    ``composition``. Everything not overridden or in the inventory is natural abundance.
    """
    overrides = overrides or {}
    inventory = inventory or {}
    inv: dict[tuple[str, int], float] = {}
    for label, fraction in inventory.items():
        nuclide = parse_nuclide(label)
        if nuclide in inv:
            raise IsotopeError(f"inventory lists {label} twice")
        inv[nuclide] = float(fraction)
    inv_elements = {e for e, _ in inv}
    clash = inv_elements & set(composition)
    if clash:
        raise IsotopeError(
            f"{sorted(clash)} appear in both the composition and the isotope inventory; "
            "the inventory defines their elemental fractions"
        )
    stray = set(overrides) - set(composition)
    if stray:
        raise IsotopeError(f"overrides for elements not in the composition: {sorted(stray)}")
    for element, fraction in composition.items():
        if not _ELEMENT.match(element):
            raise IsotopeError(f"{element!r} is not an element symbol")
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise IsotopeError(f"composition fraction of {element} must be positive")
    total = sum(composition.values()) + sum(inv.values())
    if abs(total - 1.0) > FRACTION_TOLERANCE:
        raise IsotopeError(f"atom fractions sum to {total}, not 1")
    for fraction in inv.values():
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise IsotopeError("inventory fractions must be positive")

    isotopes: dict[tuple[str, int], float] = {}
    basis: dict[str, str] = {}
    for element, fraction in composition.items():
        if element in overrides:
            within = {int(a): float(f) for a, f in overrides[element].items()}
            _check_fractions(f"override for {element}", within)
            basis[element] = "override"
        else:
            within = natural_abundance(element)
            basis[element] = "natural"
        for mass, share in within.items():
            if share > 0.0:
                isotopes[(element, mass)] = fraction * share
    for nuclide, fraction in inv.items():
        isotopes[nuclide] = fraction
    for element in inv_elements:
        basis[element] = "inventory"
    elements: dict[str, float] = {}
    for (element, _), fraction in isotopes.items():
        elements[element] = elements.get(element, 0.0) + fraction
    uses_natural = "natural" in basis.values()
    return IsotopeResolution(
        isotope_fractions=dict(sorted(isotopes.items())),
        element_fractions=dict(sorted(elements.items())),
        element_basis=dict(sorted(basis.items())),
        abundance_source=abundance_source() if uses_natural else "not used",
    )


__all__ = [
    "IsotopeError",
    "IsotopeResolution",
    "abundance_source",
    "natural_abundance",
    "parse_nuclide",
    "resolve_isotopes",
]
