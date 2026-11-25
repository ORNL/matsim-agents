"""Parse SPECTRA-PKA result files (``<stub>.out`` and ``<stub>.indexes``).

The layout was first read from the upstream Fortran output routines and then checked against
a real run (upstream commit fff9e3f, TENDL-2017 W182 matrix, upstream PWR test spectrum). The
parser rejects what it does not recognise instead of guessing:

* a block starts at a line beginning with ``###``;
* its ``#`` lines carry the title and column header, its other lines are numeric rows;
* rows must have exactly the column count implied by the block kind and by ``has_error`` /
  ``has_tdam`` (choices the caller made in the input), otherwise ``ParseError`` is raised.

Zero-PKA bins are absent from the file and stay absent here. Daughters may be ``unknown``;
they are kept. Energies are MeV, as printed.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from matsim_agents.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    ContractError,
    IrradiationScenario,
    PKASourceProvenance,
    PKASpectrum,
    RecoilSpecies,
    SpeciesSpectrum,
)

_EXPONENT_WITHOUT_E = re.compile(r"^([+-]?\d*\.?\d+)([+-]\d{3})$")
_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")
# Real upstream form (SPECTRA-PKA fff9e3f): "# (n,np) proton matrix [ H1 ] from [ W182 ]"; the
# bracketed daughter is the recoiling nuclide. The "<el> [ daughter ] from [ parent ]" form is
# kept for older outputs.
_CHANNEL = re.compile(
    r"^#\s*(?:\((?P<rxn>[^)]*)\)\s*[A-Za-z]+\s+matrix|(?P<pka>[A-Za-z]+))\s*"
    r"\[\s*(?P<dau>[A-Za-z]+)(?P<dnum>\d*)\s*\]\s*"
    r"from\s*\[\s*(?P<par>[A-Za-z]+)(?P<pnum>\d+)\s*\]",
)
_INDEX_FILE = re.compile(r"(?P<path>[^\s\]]+\.asc)\s*$", re.I)
_ISOTOPE_MATRIX = re.compile(r"recoil matrix of\s+(?P<el>[A-Za-z]+)(?P<num>\d+)\b", re.I)
_ELEMENT_MATRIX = re.compile(r"elemental recoil matrix of\s+(?P<el>[A-Za-z]+)\b", re.I)
_MT_MATRIX = re.compile(r"total recoil matrix of mt\s+(?P<mt>\d+)", re.I)
_TOTAL_MATRIX = re.compile(r"total recoil matrix", re.I)
_INDEX_HEADER = re.compile(r"^\s*###\s*index\s+(?P<n>\d+)", re.I)
_AVERAGE = re.compile(r"AVERAGE PKA ENERGY\s*=\s*(?P<v>\S+)", re.I)
_TOTAL_FLUX = re.compile(r"TOTAL FLUX\s*=\s*(?P<f>\S+)\s+FLUENCE\s*=\s*(?P<u>\S+)", re.I)

MAX_BYTES = 512 * 1024 * 1024


class ParseError(ValueError):
    """The file does not match a layout this parser recognises."""


class BlockKind(StrEnum):
    INPUT_SPECTRUM = "input_spectrum"
    CHANNEL = "channel"
    GLOBAL_ISOTOPE = "global_isotope"
    GLOBAL_ELEMENT = "global_element"
    GLOBAL_TOTAL = "global_total"
    GLOBAL_MT = "global_mt"


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    title: str
    index: int | None
    header: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    footer: dict[str, float] = field(default_factory=dict)
    # identity, when the title carries one
    recoil_element: str | None = None
    recoil_mass_number: int | None = None
    parent_element: str | None = None
    parent_mass_number: int | None = None
    daughter: str | None = None
    reaction: str | None = None
    mt: int | None = None


@dataclass(frozen=True)
class IndexEntry:
    index: int
    description: str
    pka_file: str | None


@dataclass(frozen=True)
class ParsedOutput:
    blocks: tuple[Block, ...]
    indexes: tuple[IndexEntry, ...]
    output_sha256: str
    indexes_sha256: str | None
    has_error: bool
    has_tdam: bool

    def of_kind(self, kind: BlockKind) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.kind is kind)


def parse_number(token: str) -> float:
    """Fortran ES/E output, including the three-digit exponent form that drops the ``E``."""
    text = token.strip()
    match = _EXPONENT_WITHOUT_E.match(text)
    if match:
        text = f"{match.group(1)}E{match.group(2)}"
    if not _FLOAT.match(text):
        raise ParseError(f"not a number: {token!r}")
    value = float(text.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ParseError(f"non-finite number: {token!r}")
    return value


def _element(symbol: str) -> str:
    if symbol.lower() == "unknown":
        return "unknown"
    return symbol[:1].upper() + symbol[1:].lower()


def _split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.lstrip().startswith("###"):
            current = [line]
            blocks.append(current)
        elif current is not None:
            current.append(line)
        elif line.strip():
            raise ParseError(f"content before the first block: {line.strip()[:60]!r}")
    return blocks


def _classify(comments: list[str]) -> tuple[BlockKind, dict]:
    joined = "\n".join(comments)
    if "INPUT SPECTRUM" in joined.upper():
        return BlockKind.INPUT_SPECTRUM, {}
    for line in comments:
        m = _CHANNEL.match(line.strip())
        if m:
            ident = {
                "recoil_element": _element(m["pka"] or m["dau"]),
                "daughter": _element(m["dau"]) + m["dnum"],
                "parent_element": _element(m["par"]),
                "parent_mass_number": int(m["pnum"]),
            }
            if m["rxn"] is not None:
                ident["reaction"] = f"({m['rxn']})"
                if m["dnum"]:
                    ident["recoil_mass_number"] = int(m["dnum"])
            return BlockKind.CHANNEL, ident
    for line in comments:
        m = _ELEMENT_MATRIX.search(line)
        if m:
            return BlockKind.GLOBAL_ELEMENT, {"recoil_element": _element(m["el"])}
        m = _MT_MATRIX.search(line)
        if m:
            return BlockKind.GLOBAL_MT, {"mt": int(m["mt"])}
        if _TOTAL_MATRIX.search(line):
            return BlockKind.GLOBAL_TOTAL, {}
        m = _ISOTOPE_MATRIX.search(line)
        if m:
            return BlockKind.GLOBAL_ISOTOPE, {
                "recoil_element": _element(m["el"]),
                "recoil_mass_number": int(m["num"]),
            }
    raise ParseError(
        "unrecognised block title: " + " | ".join(c.strip()[:70] for c in comments[:3])
    )


def _expected_columns(kind: BlockKind, has_error: bool, has_tdam: bool) -> int:
    if kind is BlockKind.INPUT_SPECTRUM:
        return 3
    if kind is BlockKind.CHANNEL:
        return 4 + (1 if has_error else 0) + (4 if has_tdam else 0)
    return 5


def _parse_block(raw: list[str], *, has_error: bool, has_tdam: bool) -> Block:
    head = raw[0]
    m = _INDEX_HEADER.match(head)
    index = int(m["n"]) if m else None
    comments: list[str] = []
    rows: list[tuple[float, ...]] = []
    footer: dict[str, float] = {}
    for line in raw[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comments.append(stripped)
            fm = _AVERAGE.search(stripped)
            if fm:
                footer["average_pka_energy_ev"] = parse_number(fm["v"])
            tf = _TOTAL_FLUX.search(stripped)
            if tf:
                footer["total_flux"] = parse_number(tf["f"])
                footer["fluence"] = parse_number(tf["u"])
            continue
        rows.append(tuple(parse_number(t) for t in stripped.split()))
    kind, ident = _classify(comments)
    width = _expected_columns(kind, has_error, has_tdam)
    for row in rows:
        if len(row) != width:
            raise ParseError(
                f"{kind.value} block {comments[0][:50]!r}: expected {width} columns, got "
                f"{len(row)}; check has_error/has_tdam against the input used"
            )
    return Block(
        kind=kind,
        title=comments[0] if comments else head.strip(),
        index=index,
        header=tuple(comments),
        rows=tuple(rows),
        footer=footer,
        **ident,
    )


def parse_indexes(text: str) -> tuple[IndexEntry, ...]:
    entries: list[IndexEntry] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise ParseError(f"indexes line {number}: expected '<index> <description>'")
        description = parts[1].strip()
        path: str | None = None
        # Upstream writes the file name straight after "]" with no space.
        fm = _INDEX_FILE.search(description)
        if fm:
            path = fm["path"]
            description = description[: fm.start()].strip()
        entries.append(IndexEntry(int(parts[0]), description, path))
    return tuple(entries)


def parse_output_text(
    text: str, *, has_error: bool = False, has_tdam: bool = False, indexes_text: str | None = None
) -> ParsedOutput:
    if not text.strip():
        raise ParseError("the output file is empty")
    blocks = [
        _parse_block(raw, has_error=has_error, has_tdam=has_tdam)
        for raw in _split_blocks(text.splitlines())
    ]
    if not blocks:
        raise ParseError("no blocks found (no line starts with '###')")
    return ParsedOutput(
        blocks=tuple(blocks),
        indexes=parse_indexes(indexes_text) if indexes_text is not None else (),
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        indexes_sha256=(
            hashlib.sha256(indexes_text.encode()).hexdigest() if indexes_text is not None else None
        ),
        has_error=has_error,
        has_tdam=has_tdam,
    )


def parse_output_file(
    out_path: Path,
    indexes_path: Path | None = None,
    *,
    has_error: bool = False,
    has_tdam: bool = False,
) -> ParsedOutput:
    """Read and parse. The raw files are only read, never rewritten."""
    for p in (out_path, indexes_path):
        if p is not None and p.is_file() and p.stat().st_size > MAX_BYTES:
            raise ParseError(f"{p} is larger than {MAX_BYTES} bytes")
    try:
        text = out_path.read_text()
        indexes_text = indexes_path.read_text() if indexes_path and indexes_path.is_file() else None
    except (OSError, UnicodeDecodeError) as exc:
        raise ParseError(f"cannot read output: {exc}") from None
    return parse_output_text(
        text, has_error=has_error, has_tdam=has_tdam, indexes_text=indexes_text
    )


def _spectrum_from_block(block: Block) -> SpeciesSpectrum:
    if not block.rows:
        raise ParseError(f"block {block.title!r} has no rows")
    return SpeciesSpectrum(
        species=RecoilSpecies(
            element=block.recoil_element or "unknown", mass_number=block.recoil_mass_number
        ),
        bin_low_mev=tuple(r[0] for r in block.rows),
        bin_high_mev=tuple(r[1] for r in block.rows),
        rates=tuple(r[2] for r in block.rows),
    )


def to_pka_spectrum(
    parsed: ParsedOutput,
    scenario: IrradiationScenario,
    provenance: PKASourceProvenance,
    *,
    resolution: str = "isotope",
) -> PKASpectrum:
    """Species-resolved spectrum from the per-target-atom global blocks.

    ``resolution='isotope'`` uses the isotope matrices, ``'element'`` the elemental ones. Each
    block is the recoil of one species; its parent is not stated there, so ``parent`` stays
    unset and sites are matched by the recoiling element (an approximation for products of
    capture reactions, to be refined from the channel blocks).
    """
    kind = BlockKind.GLOBAL_ISOTOPE if resolution == "isotope" else BlockKind.GLOBAL_ELEMENT
    blocks = parsed.of_kind(kind)
    if not blocks:
        raise ParseError(
            f"no {kind.value} blocks in the output; was do_global_sums enabled? "
            "Nothing is invented in their place."
        )
    try:
        return PKASpectrum(
            scenario=scenario,
            spectra=tuple(_spectrum_from_block(b) for b in blocks),
            rate_unit="PKA s^-1 per target atom",
            rate_basis=RATE_BASIS_PER_ATOM_PER_S,
            provenance=provenance,
        )
    except (ContractError, ValueError) as exc:
        raise ParseError(f"output cannot be turned into a spectrum: {exc}") from None


__all__ = [
    "Block",
    "BlockKind",
    "IndexEntry",
    "ParseError",
    "ParsedOutput",
    "parse_indexes",
    "parse_number",
    "parse_output_file",
    "parse_output_text",
    "to_pka_spectrum",
]
