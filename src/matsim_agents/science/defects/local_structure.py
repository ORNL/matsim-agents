"""Per-atom local structural descriptors that complement Wigner-Seitz occupancy analysis.

Coordination number, centrosymmetry parameter, fixed-cutoff common neighbour analysis (CNA) and
Voronoi cell volume, all under periodic boundary conditions. Every neighbour criterion is an
explicit argument with no default, because the right cutoff depends on the lattice, the
temperature and the alloy. Labels are geometric: thermal disorder can move an atom across a
cutoff, so compare against a thermally equilibrated reference (``label_change_fraction``)
rather than reading absolute fractions as defect counts. Nothing here is validated against
cascade data.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial import ConvexHull, Voronoi

_CNA_SIGNATURES = {
    "fcc": {(4, 2, 1): 12},
    "hcp": {(4, 2, 1): 6, (4, 2, 2): 6},
    "bcc": {(6, 6, 6): 8, (4, 4, 4): 6},
    "ico": {(5, 5, 5): 12},
}


def _check_cutoff(cutoff: float) -> None:
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be positive and finite")


_Image = tuple[int, tuple[int, int, int]]


def _image_neighbours(atoms: Atoms, cutoff: float) -> list[dict[_Image, None]]:
    """Neighbours of each atom as (index, periodic shift) so small cells are not double counted."""
    i, j, shift = neighbor_list("ijS", atoms, cutoff)
    out: list[dict[_Image, None]] = [{} for _ in range(len(atoms))]
    for a, b, s in zip(i, j, shift, strict=True):
        out[int(a)][(int(b), (int(s[0]), int(s[1]), int(s[2])))] = None
    return out


def coordination_numbers(atoms: Atoms, *, cutoff: float) -> np.ndarray:
    """Number of atoms within ``cutoff`` angstrom of each atom."""
    _check_cutoff(cutoff)
    i = neighbor_list("i", atoms, cutoff)
    return np.bincount(i, minlength=len(atoms))


def centrosymmetry(atoms: Atoms, *, n_neighbors: int) -> np.ndarray:
    """Centrosymmetry parameter per atom (LAMMPS definition, angstrom^2).

    Uses the ``n_neighbors`` nearest neighbours (even; 8 for bcc, 12 for fcc) and sums the
    ``n_neighbors / 2`` smallest values of ``|r_i + r_j|^2`` over all neighbour pairs. It is
    near zero in a perfect centrosymmetric lattice and grows near defects. Atoms with fewer
    neighbours than requested within the search radius get NaN.
    """
    if n_neighbors < 2 or n_neighbors % 2:
        raise ValueError("n_neighbors must be an even integer >= 2")
    if len(atoms) < n_neighbors + 1:
        raise ValueError("too few atoms for the requested number of neighbours")
    # Grow the search radius until every atom has enough neighbours or the cap is reached.
    volume_per_atom = abs(atoms.cell.volume) / len(atoms)
    radius = 1.5 * (volume_per_atom ** (1.0 / 3.0))
    for _ in range(8):
        i, _j, d = neighbor_list("ijD", atoms, radius)
        if np.bincount(i, minlength=len(atoms)).min() >= n_neighbors:
            break
        radius *= 1.3
    csp = np.full(len(atoms), np.nan)
    order = np.argsort(i, kind="stable")
    i, d = i[order], d[order]
    starts = np.searchsorted(i, np.arange(len(atoms) + 1))
    iu = np.triu_indices(n_neighbors, k=1)
    for a in range(len(atoms)):
        vecs = d[starts[a] : starts[a + 1]]
        if len(vecs) < n_neighbors:
            continue
        nearest = vecs[np.argsort(np.linalg.norm(vecs, axis=1))[:n_neighbors]]
        sums = nearest[iu[0]] + nearest[iu[1]]
        pair = np.sum(sums * sums, axis=1)
        csp[a] = np.sort(pair)[: n_neighbors // 2].sum()
    return csp


def _longest_bond_chain(edges: list[tuple[int, int]]) -> int:
    best = 0

    def walk(node: int, used: frozenset[int], length: int) -> None:
        nonlocal best
        best = max(best, length)
        for e, (u, v) in enumerate(edges):
            if e in used or node not in (u, v):
                continue
            walk(v if node == u else u, used | {e}, length + 1)

    for start in {n for e in edges for n in e}:
        walk(start, frozenset(), 0)
    return best


def cna_signatures(atoms: Atoms, *, cutoff: float) -> list[Counter[tuple[int, int, int]]]:
    """Honeycutt-Andersen (common neighbours, bonds among them, longest bond chain) per atom.

    One signature per neighbour bond within ``cutoff``; the cutoff must lie between the
    shells that define the structure of interest (the first shell for fcc, hcp and icosahedral,
    between the second and third shell for bcc).
    """
    _check_cutoff(cutoff)
    nbrs = _image_neighbours(atoms, cutoff)
    out: list[Counter[tuple[int, int, int]]] = []
    for neigh in nbrs:
        sigs: Counter[tuple[int, int, int]] = Counter()
        for b, sab in neigh:
            # Express b's neighbours in a's image frame.
            around_b = {(k, (sab[0] + sk[0], sab[1] + sk[1], sab[2] + sk[2])) for k, sk in nbrs[b]}
            common = sorted(set(neigh) & around_b)
            edges = []
            for x, (k, sk) in enumerate(common):
                for ell, sl in common[x + 1 :]:
                    rel = (sl[0] - sk[0], sl[1] - sk[1], sl[2] - sk[2])
                    if (ell, rel) in nbrs[k]:
                        edges.append((x, common.index((ell, sl))))
            sigs[(len(common), len(edges), _longest_bond_chain(edges))] += 1
        out.append(sigs)
    return out


def cna_labels(atoms: Atoms, *, cutoff: float) -> list[str]:
    """Structure label per atom: ``fcc``, ``hcp``, ``bcc``, ``ico`` or ``other``.

    ``other`` covers surfaces, defect cores, disordered regions and atoms the cutoff
    misclassifies; it is not a structural identification.
    """
    labels = []
    for sigs in cna_signatures(atoms, cutoff=cutoff):
        label = "other"
        for name, pattern in _CNA_SIGNATURES.items():
            if dict(sigs) == pattern:
                label = name
                break
        labels.append(label)
    return labels


def voronoi_volumes(atoms: Atoms) -> np.ndarray:
    """Voronoi cell volume per atom in angstrom^3, using periodic images in 27 cells.

    Returns NaN for an atom whose cell is not closed by the images, which happens when the
    periodic cell is too small for the neighbour range; check that the finite volumes sum to
    the cell volume.
    """
    n = len(atoms)
    others = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    others.remove((0, 0, 0))
    shifts = np.array([(0, 0, 0), *others])
    pos = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    images = np.concatenate([pos + s @ cell for s in shifts])
    vor = Voronoi(images)
    vols = np.full(n, np.nan)
    for a in range(n):
        region = vor.regions[vor.point_region[a]]
        if not region or -1 in region:
            continue
        vols[a] = ConvexHull(vor.vertices[region]).volume
    return vols


def label_change_fraction(labels: list[str], reference_labels: list[str]) -> float:
    """Fraction of atoms whose label differs from the same atom in a reference labelling."""
    if len(labels) != len(reference_labels) or not labels:
        raise ValueError("labels must be non-empty and the same length as reference_labels")
    return sum(a != b for a, b in zip(labels, reference_labels, strict=True)) / len(labels)
