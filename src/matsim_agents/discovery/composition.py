"""Lightweight chemical-composition parsing.

We deliberately avoid pulling in pymatgen for the parsing step so this
module works in minimal environments (e.g. when the user is just chatting
about hypotheses). pymatgen is used downstream by the phase explorer when
available.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from pydantic import BaseModel, Field

# Two-letter symbols listed first so the regex is greedy on them.
_ELEMENT_SYMBOLS = (
    "He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba Hf Ta Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "H B C N O F P S K V Y I W U Y"
).split()
_ELEMENT_SET = set(_ELEMENT_SYMBOLS)
_ELEMENT_PATTERN = "|".join(sorted(set(_ELEMENT_SYMBOLS), key=len, reverse=True))
_FORMULA_TOKEN_RE = re.compile(rf"({_ELEMENT_PATTERN})(\d*)")
# A formula must contain at least two tokens (or one token followed by digit)
# to be meaningfully identified inside free text.
_FORMULA_RE = re.compile(rf"\b((?:(?:{_ELEMENT_PATTERN})\d*){{1,}})\b")


class Composition(BaseModel):
    """A normalized chemical composition."""

    formula: str = Field(..., description="Hill-ordered reduced formula, e.g. 'Li2MnO3'.")
    elements: dict[str, int] = Field(..., description="Element -> stoichiometric integer count.")

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def total_atoms(self) -> int:
        return sum(self.elements.values())


def _hill_order(elements: dict[str, int]) -> list[str]:
    keys = list(elements.keys())
    if "C" in keys:
        rest = sorted(k for k in keys if k not in {"C", "H"})
        return ["C"] + (["H"] if "H" in keys else []) + rest
    return sorted(keys)


def _reduce(elements: dict[str, int]) -> dict[str, int]:
    from math import gcd
    from functools import reduce as _r

    counts = list(elements.values())
    if not counts:
        return elements
    g = _r(gcd, counts)
    if g <= 1:
        return elements
    return {k: v // g for k, v in elements.items()}


def parse_composition(text: str) -> Composition | None:
    """Parse a single formula string into a :class:`Composition`.

    Returns ``None`` if the string does not look like a chemical formula
    (e.g. contains unknown element symbols or no element tokens).
    """
    text = text.strip()
    if not text:
        return None
    tokens = _FORMULA_TOKEN_RE.findall(text)
    # Reject if leftover characters do not consist of digits/whitespace.
    rebuilt = "".join(sym + cnt for sym, cnt in tokens)
    if rebuilt != text.replace(" ", ""):
        return None
    counts: Counter[str] = Counter()
    for symbol, n in tokens:
        if symbol not in _ELEMENT_SET:
            return None
        counts[symbol] += int(n) if n else 1
    if not counts:
        return None
    elements = _reduce(dict(counts))
    formula = "".join(
        f"{el}{elements[el] if elements[el] > 1 else ''}" for el in _hill_order(elements)
    )
    return Composition(formula=formula, elements=elements)


def extract_compositions(text: str) -> list[Composition]:
    """Extract all plausible chemical compositions from a free-text blob.

    Used by the chat agent to detect when a user proposes a new material.
    Single bare element tokens are ignored to avoid false positives like
    "C" (carbon vs. the letter).
    """
    found: dict[str, Composition] = {}
    for match in _FORMULA_RE.finditer(text):
        candidate = match.group(1)
        comp = parse_composition(candidate)
        if comp is None:
            continue
        # Filter out single-element single-atom hits that are very likely just
        # English words capitalized at sentence start.
        if comp.num_elements == 1 and comp.total_atoms == 1:
            continue
        found.setdefault(comp.formula, comp)
    return list(found.values())


def unique(compositions: Iterable[Composition]) -> list[Composition]:
    """De-duplicate compositions by formula, preserving insertion order."""
    seen: dict[str, Composition] = {}
    for c in compositions:
        seen.setdefault(c.formula, c)
    return list(seen.values())
