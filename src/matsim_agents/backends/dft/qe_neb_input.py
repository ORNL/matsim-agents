"""Write a single-file ``neb.x`` input (``neb.x -inp neb.in``) from a ``QENebConfig`` and images.

Layout (per the Quantum ESPRESSO NEB documentation)::

    BEGIN
    BEGIN_PATH_INPUT
    &PATH ... /
    [CLIMBING_IMAGES i, j]            manual climbing images only
    END_PATH_INPUT
    BEGIN_ENGINE_INPUT
    &CONTROL / &SYSTEM / &ELECTRONS / &IONS
    ATOMIC_SPECIES                    must precede the positions: pw.x aborts otherwise
    BEGIN_POSITIONS
    FIRST_IMAGE / ATOMIC_POSITIONS angstrom ...
    INTERMEDIATE_IMAGE / ATOMIC_POSITIONS angstrom ...   (repeated)
    LAST_IMAGE / ATOMIC_POSITIONS angstrom ...
    END_POSITIONS
    CELL_PARAMETERS angstrom / K_POINTS automatic
    END_ENGINE_INPUT
    END

All images share species, atom order and cell. A climbing image is never requested from the
first iteration: a ``from_scratch`` input must use ``ci_scheme='no-CI'``; climbing images are
enabled by a later ``restart`` stage (see the staged workflow).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_relax import _ATOMIC_MASS, _fortran, _namelist

_CELL_TOLERANCE = 1e-8


@dataclass(frozen=True)
class NebInput:
    path: str
    sha256: str
    n_images: int
    config_hash: str


def _check_images(config: QENebConfig, images: Sequence[Atoms]) -> None:
    if len(images) != config.n_images:
        raise ValueError(f"{len(images)} images supplied but the config asks for {config.n_images}")
    first = images[0]
    config.check_structure(first)
    symbols = first.get_chemical_symbols()
    ids = first.arrays.get("atom_id")
    for k, image in enumerate(images):
        if image.get_chemical_symbols() != symbols:
            raise ValueError(f"image {k} species or atom order differ from the first image")
        if (ids is None) != ("atom_id" not in image.arrays) or (
            ids is not None and not np.array_equal(image.arrays["atom_id"], ids)
        ):
            raise ValueError(f"image {k} atom identities differ from the first image")
        if not np.allclose(image.cell.array, first.cell.array, rtol=0.0, atol=_CELL_TOLERANCE):
            raise ValueError(f"image {k} cell differs from the first image")
        if not np.isfinite(image.positions).all() or not np.isfinite(image.cell.array).all():
            raise ValueError(f"image {k} has non-finite coordinates or cell")
    if not first.pbc.all():
        raise ValueError("neb.x inputs here assume fully periodic cells")


def _positions_card(image: Atoms) -> list[str]:
    lines = ["ATOMIC_POSITIONS angstrom"]
    for symbol, pos in zip(image.get_chemical_symbols(), image.positions, strict=True):
        lines.append(f"  {symbol}  {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
    return lines


def render_neb_input(config: QENebConfig, images: Sequence[Atoms]) -> str:
    """Return the ``neb.x`` input text; raises ``ValueError`` for any inconsistency."""
    images = list(images)
    if not images:
        raise ValueError("no images supplied")
    _check_images(config, images)
    if config.restart_mode == "from_scratch" and config.ci_scheme != "no-CI":
        raise ValueError(
            "a from_scratch NEB must use ci_scheme 'no-CI'; enable climbing images in a "
            "restart stage after the path has converged"
        )

    species = sorted(set(images[0].get_chemical_symbols()))
    nat = len(images[0])

    path = {
        "string_method": "'neb'",
        "restart_mode": f"'{config.restart_mode}'",
        "nstep_path": str(config.nstep_path),
        "num_of_images": str(config.n_images),
        "opt_scheme": f"'{config.opt_scheme}'",
        "CI_scheme": f"'{config.ci_scheme}'",
        "path_thr": _fortran(config.path_thr_ev_ang),
        "minimum_image": _fortran(config.minimum_image),
    }
    path.update({k: _fortran(v) for k, v in config.extra_path.items()})

    control = {
        "calculation": "'relax'",
        "prefix": f"'{config.prefix}'",
        "pseudo_dir": f"'{os.path.abspath(config.pseudo_dir)}'",
        "outdir": f"'{config.outdir}'",
        "tprnfor": ".true.",
        "verbosity": "'high'",
    }
    system = {
        "ibrav": "0",
        "nat": str(nat),
        "ntyp": str(len(species)),
        "ecutwfc": _fortran(config.ecutwfc_ry),
        "ecutrho": _fortran(config.ecutrho_ry),
        "occupations": f"'{config.occupations}'",
    }
    if config.occupations == "smearing":
        system["smearing"] = f"'{config.smearing}'"
        system["degauss"] = _fortran(config.degauss_ry)
    system["nspin"] = str(config.nspin)
    if config.nspin == 2:
        for index, symbol in enumerate(species, start=1):
            system[f"starting_magnetization({index})"] = _fortran(
                config.starting_magnetization[symbol]
            )
    system.update({k: _fortran(v) for k, v in config.extra_system.items()})
    electrons = {
        "conv_thr": _fortran(config.conv_thr_ry),
        "mixing_beta": _fortran(config.mixing_beta),
    }
    electrons.update({k: _fortran(v) for k, v in config.extra_electrons.items()})
    ions = {"ion_dynamics": "'bfgs'"}

    parts: list[str] = ["BEGIN", "BEGIN_PATH_INPUT", _namelist("PATH", path)]
    if config.ci_scheme == "manual":
        parts.append("CLIMBING_IMAGES")
        parts.append("  " + ", ".join(str(i) for i in config.climbing_images))
    parts.append("END_PATH_INPUT")
    parts.append("BEGIN_ENGINE_INPUT")
    parts.append(_namelist("CONTROL", control))
    parts.append(_namelist("SYSTEM", system))
    parts.append(_namelist("ELECTRONS", electrons))
    parts.append(_namelist("IONS", ions))
    parts.append("ATOMIC_SPECIES")
    for symbol in species:
        parts.append(
            f"  {symbol}  {_ATOMIC_MASS.get(symbol, 1.0):.4f}  {config.pseudopotentials[symbol]}"
        )
    parts.append("BEGIN_POSITIONS")
    for index, image in enumerate(images):
        label = (
            "FIRST_IMAGE"
            if index == 0
            else "LAST_IMAGE"
            if index == len(images) - 1
            else "INTERMEDIATE_IMAGE"
        )
        parts.append(label)
        parts.extend(_positions_card(image))
    parts.append("END_POSITIONS")

    parts.append("CELL_PARAMETERS angstrom")
    for v in images[0].cell.array:
        parts.append(f"  {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
    nk, ko = config.kpts, config.koffset
    parts.append("K_POINTS automatic")
    parts.append(f"  {nk[0]} {nk[1]} {nk[2]}  {ko[0]} {ko[1]} {ko[2]}")
    parts.append("END_ENGINE_INPUT")
    parts.append("END")
    return "\n".join(parts) + "\n"


def write_neb_input(
    config: QENebConfig,
    images: Sequence[Atoms],
    input_path: str | os.PathLike,
    *,
    overwrite: bool = False,
) -> NebInput:
    """Write ``input_path`` atomically and return its path and SHA-256.

    An existing file is kept unless ``overwrite`` is true, and is replaced only after the new
    content has been written completely.
    """
    text = render_neb_input(config, images)
    target = Path(input_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(prefix=".neb-in-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise
    return NebInput(
        path=str(target.resolve()),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        n_images=len(images),
        config_hash=config.config_hash(),
    )


__all__ = ["NebInput", "render_neb_input", "write_neb_input"]
