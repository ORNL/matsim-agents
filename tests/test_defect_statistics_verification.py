"""Software verification on constructed damaged cells; not physical validation."""

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.defects.cascade_ensemble import Realization, Stratum, stratified_estimate
from matsim_agents.science.defects.ensemble import convergence_curve
from matsim_agents.science.defects.survival import cluster_sensitivity
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis

A = 3.165


def _ref():
    return bulk("W", "bcc", a=A, cubic=True).repeat(5)


def _damage(ref, rng, n_pairs, noise):
    cur = ref.copy()
    idx = rng.choice(len(ref), size=n_pairs, replace=False)
    pos = cur.positions
    for i in idx:
        pos[i] = rng.uniform(0, 1, 3) @ ref.cell.array
    keep = np.setdiff1d(np.arange(len(ref)), idx)
    pos[keep] += rng.normal(0.0, noise, (len(keep), 3))
    cur.positions = pos
    return cur


def _brute_force(ref, cur):
    """Independent analysis: full minimum-image distance matrix on the orthogonal cell."""
    L = np.diag(ref.cell.array)
    d = cur.positions[:, None, :] - ref.positions[None, :, :]
    d -= np.round(d / L) * L
    nearest = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
    occ = np.bincount(nearest, minlength=len(ref))
    return int((occ == 0).sum()), int(np.maximum(occ - 1, 0).sum())


@pytest.mark.parametrize("seed", range(8))
def test_agrees_with_independent_brute_force_on_random_damage(seed):
    rng = np.random.default_rng(seed)
    ref = _ref()
    cur = _damage(ref, rng, n_pairs=int(rng.integers(0, 12)), noise=0.05)
    ws = wigner_seitz_analysis(ref, cur)
    n_vac, n_int = _brute_force(ref, cur)
    assert len(ws.vacant_sites) == n_vac
    assert sum(max(o - 1, 0) for o in ws.occupancy) == n_int


def test_spurious_defects_appear_as_thermal_noise_approaches_half_spacing():
    rng = np.random.default_rng(0)
    ref = _ref()
    counts = []
    for noise in (0.05, 0.3, 0.8, 1.2):
        cur = _damage(ref, np.random.default_rng(1), 0, noise)
        counts.append(len(wigner_seitz_analysis(ref, cur).vacant_sites))
    assert counts[0] == 0 and counts[1] == 0
    assert counts[-1] > counts[1] and counts == sorted(counts)
    del rng


def test_cluster_cutoff_changes_the_cluster_statistics_of_a_damaged_cell():
    rng = np.random.default_rng(3)
    ref = _ref()
    cur = _damage(ref, rng, 20, 0.02)
    ws = wigner_seitz_analysis(ref, cur)
    pos = ref.positions[list(ws.vacant_sites)]
    res = cluster_sensitivity(pos, ref.cell.array, [2.0, 3.0, 5.0, 8.0])
    n = [r.n_clusters for r in res]
    assert n[0] >= n[-1] and len(set(n)) > 1
    assert res[-1].largest >= res[0].largest


def test_interval_halfwidth_shrinks_with_ensemble_size_and_covers_the_truth():
    rng = np.random.default_rng(5)
    truth = 40.0
    vals = rng.poisson(truth, 400).astype(float)
    sizes = [5, 10, 20, 40, 80, 160, 320]
    curve = convergence_curve(vals, sizes)
    half = [(e.ci_high - e.ci_low) / 2 for e in curve]
    assert half[-1] < half[0] / 4
    assert half[-1] == pytest.approx(half[0] * np.sqrt(sizes[0] / sizes[-1]), rel=0.5)
    assert curve[-1].ci_low < truth < curve[-1].ci_high


def test_stratified_interval_shrinks_with_cascades_per_stratum():
    strata = [Stratum("a", 0.7), Stratum("b", 0.3)]
    rng = np.random.default_rng(7)
    widths = []
    for n in (4, 16, 64):
        real = [
            Realization(f"{s}{i}", s, i, float(rng.normal(m, 3)))
            for s, m in (("a", 10.0), ("b", 30.0))
            for i in range(n)
        ]
        e = stratified_estimate(strata, real)
        widths.append(e.ci_high - e.ci_low)
    assert widths[0] > widths[1] > widths[2]
