"""Read LAMMPS thermo output and the final per-atom dump written by ``inputs.render_script``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read


class LammpsParseError(ValueError):
    pass


@dataclass(frozen=True)
class ThermoBlock:
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    def column(self, name: str) -> tuple[float, ...]:
        try:
            i = self.columns.index(name)
        except ValueError:
            raise LammpsParseError(f"no thermo column {name!r} in {self.columns}") from None
        return tuple(r[i] for r in self.rows)

    def last(self, name: str) -> float:
        values = self.column(name)
        if not values:
            raise LammpsParseError("thermo block has no rows")
        return values[-1]


def parse_log(text: str) -> list[ThermoBlock]:
    """Every thermo block (one per ``run``/``minimize``), from its header to ``Loop time``."""
    blocks: list[ThermoBlock] = []
    columns: tuple[str, ...] | None = None
    rows: list[tuple[float, ...]] = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "Step" and len(parts) > 1:
            columns, rows = tuple(parts), []
            continue
        if columns is None:
            continue
        if line.startswith("Loop time"):
            blocks.append(ThermoBlock(columns, tuple(rows)))
            columns = None
            continue
        if len(parts) == len(columns):
            try:
                rows.append(tuple(float(p) for p in parts))
            except ValueError:
                continue
    return blocks


def parse_log_file(path: Path) -> list[ThermoBlock]:
    if not path.is_file():
        raise LammpsParseError(f"log not found: {path}")
    blocks = parse_log(path.read_text(errors="replace"))
    if not blocks:
        raise LammpsParseError(f"no completed thermo block in {path}")
    return blocks


def read_final_atoms(dump: Path) -> Atoms:
    """Final positions and forces (eV/Angstrom) from the sorted dump with an element column."""
    if not dump.is_file():
        raise LammpsParseError(f"dump not found: {dump}")
    return ase_read(dump, format="lammps-dump-text", index=-1)


__all__ = ["LammpsParseError", "ThermoBlock", "parse_log", "parse_log_file", "read_final_atoms"]
