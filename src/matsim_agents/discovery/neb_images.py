"""Linear NEB image generation in periodic displacement space.

Images are ``initial + t * d`` with ``d`` the identity-matched minimum-image displacement of
:mod:`neb_paths` and ``t = k / (n - 1)``, so an atom that crosses a cell boundary takes the short
route and the path is continuous in unwrapped coordinates. Image 0 is the initial structure
unchanged. The last image is the final structure translated by lattice vectors into the
unwrapped frame (same periodic configuration, different coordinates), because QE ``neb.x`` does
not apply the minimum-image convention itself. Atom order, ``atom_id`` and species are those of
the initial structure.

A linear interpolation is an initial guess. It is not a minimum-energy path, a barrier or
evidence of a mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from matsim_agents.discovery.neb_contracts import MIN_IMAGES
from matsim_agents.discovery.neb_paths import PeriodicDisplacements, periodic_displacements
from matsim_agents.discovery.radiation_defects import geometry_fingerprint

DEFAULT_MIN_DISTANCE = 1.0  # angstrom; hard-overlap guard, not a physical stability criterion
DEFAULT_MAX_STEP = 1.0  # angstrom; largest allowed per-atom move between neighbouring images


@dataclass(frozen=True)
class LinearImages:
    """Generated images (index 0 is the initial endpoint) and the facts behind them."""

    images: tuple[Atoms, ...]
    displacements: PeriodicDisplacements
    fingerprints: tuple[str, ...]
    max_step: float  # largest per-atom move between neighbouring images, angstrom
    parameters: dict[str, float | int | str]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _check_image(image: Atoms, index: int, min_distance: float) -> None:
    if not np.isfinite(image.positions).all():
        raise ValueError(f"image {index} has non-finite coordinates")
    close = neighbor_list("ijd", image, min_distance)
    if len(close[0]):
        k = int(np.argmin(close[2]))
        a, b = (
            int(image.get_array("atom_id")[close[0][k]]),
            int(image.get_array("atom_id")[close[1][k]]),
        )
        raise ValueError(
            f"image {index}: atom_id {a} and {b} are {float(close[2][k]):.3f} A apart "
            f"(minimum {min_distance:.3f} A)"
        )


def generate_linear_images(
    initial: Atoms,
    final: Atoms,
    n_images: int,
    *,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    max_step: float = DEFAULT_MAX_STEP,
    allow_ambiguous: bool = False,
) -> LinearImages:
    """``n_images`` images including both endpoints (QE needs at least 4).

    Raises ``ValueError`` for too few images, broken correspondence, invalid cells, ambiguous
    displacements (unless ``allow_ambiguous``), non-finite coordinates, an overlap closer than
    ``min_distance`` in any image, or a per-atom step between neighbouring images above
    ``max_step`` (a likely wrong pairing or too few images). Inputs are not modified.
    """
    if isinstance(n_images, bool) or not isinstance(n_images, int | np.integer):
        raise ValueError(f"n_images must be an integer, got {n_images!r}")
    n_images = int(n_images)
    if n_images < MIN_IMAGES:
        raise ValueError(f"n_images must be at least {MIN_IMAGES}, got {n_images}")
    min_distance = _positive("min_distance", min_distance)
    max_step = _positive("max_step", max_step)

    displacements = periodic_displacements(initial, final, allow_ambiguous=allow_ambiguous)
    step = displacements.max_length / (n_images - 1)
    if step > max_step:
        raise ValueError(
            f"per-image step {step:.3f} A exceeds {max_step:.3f} A "
            f"(largest displacement {displacements.max_length:.3f} A over {n_images} images); "
            "use more images or check the endpoint pairing"
        )
    images: list[Atoms] = []
    for k in range(n_images):
        t = k / (n_images - 1)
        image = initial.copy()
        if k:
            image.positions = initial.positions + t * displacements.vectors
        image.info.clear()
        _check_image(image, k, min_distance)
        images.append(image)
    last = images[-1]
    # The last image must be the final structure up to lattice translations.
    offset = (last.positions - final.positions[_final_order(initial, final)]) @ np.linalg.inv(
        initial.cell.array
    )
    if not np.allclose(offset, np.round(offset), atol=1e-6):
        raise ValueError("last image is not the final structure modulo lattice vectors")
    return LinearImages(
        images=tuple(images),
        displacements=displacements,
        fingerprints=tuple(geometry_fingerprint(image) for image in images),
        max_step=step,
        parameters={
            "method": "linear_periodic",
            "n_images": n_images,
            "min_distance": min_distance,
            "max_step": max_step,
            "max_displacement": displacements.max_length,
        },
    )


def _final_order(initial: Atoms, final: Atoms) -> np.ndarray:
    final_ids = final.get_array("atom_id").tolist()
    index = {atom_id: k for k, atom_id in enumerate(final_ids)}
    return np.array([index[int(i)] for i in initial.get_array("atom_id")], dtype=int)


__all__ = [
    "DEFAULT_MAX_STEP",
    "DEFAULT_MIN_DISTANCE",
    "LinearImages",
    "generate_linear_images",
]
