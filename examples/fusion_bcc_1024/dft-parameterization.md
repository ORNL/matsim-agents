# DFT parameterization of the W campaign

This note records how the Quantum ESPRESSO parameters for the W campaign were chosen, with the
numbers behind each choice. It covers the measured evidence only. No protocol is qualified.

## Summary

| Parameter | Value for the campaign | Status | Evidence in one line |
|---|---|---|---|
| Functional and pseudopotentials | PBE, PSlibrary 1.0.0 PAW (`W.pbe-spn-kjpaw_psl.1.0.0.UPF`, Re, He) | fixed | Chosen before the scans; the scans test numerics, not this choice |
| `conv_thr` | 1e-7 Ry | qualified | 1e-6 fails on forces (0.0073 eV/A); 1e-7 and 1e-8 agree within every limit |
| k-mesh | 7x7x7 on the 54-site cell (k-spacing 0.10 1/A) | provisional | 6x6x6 passes against 7x7x7, 5x5x5 does not; no denser mesh was run |
| `ecutwfc` / `ecutrho` | 100 / 1200 Ry | not converged | 90/1080 differs from 100/1200 by 1.093 meV/atom against a 1.0 limit |
| Smearing | Marzari-Vanderbilt, `degauss` 0.010 Ry | provisional | 0.010 versus 0.005 gives a 0.0265 eV/A force difference against 0.005; scanned at a coarse mesh |

Every protocol in [qe-protocols.yaml](../../deployments/common/qe-protocols.yaml) stays `unqualified`.
The values above are stored there under `provisional_recommendation`, not in any `settings` block.

## Collecting magnetic-state results

For a campaign prepared by the current code, put each QE output in its calculation directory
as `pw.out`, then run `matsim-agents magnetic collect <campaign-directory>`. The collector
writes `magnetic_selection.json` with the converged energies and output hashes for each state,
the lowest-energy evaluated state per structure, and incomplete or rejected results. It
checks the prepared structure, input and pseudopotential hashes and rejects differing structures or
methods. A missing, incomplete or unconverged state leaves the selection provisional and the
magnetic downstream flag false. Missing spin-resolved site moments or unresolved energy ties
also prevent readiness for review. The downstream flag remains false even for complete groups:
the collector cannot verify that an output was produced from its prepared input. Rankings cover
only the requested states, not all magnetic orders.

The flag is only a magnetic-survey check, not numerical qualification of the DFT method. The
generic NEB workflow does not yet consume this flag or selected moments automatically; do not
run a downstream campaign calculation on the assumption that this report gates it. Older
manifests without the full magnetic configuration and pseudopotential hashes cannot be
collected by this command; preserve their original inputs and results rather than overwriting
them. No QE calculation is launched by collection.

## Scope and constraints

- Machine: OLCF Frontier, account LRN110, QOS `normal` (120 min limit), `pw.x` from QE 8.0dev on
  AMD GPUs, launched through
  [run-pw-gpu-frontier.sh](../../deployments/frontier/launchers/run-pw-gpu-frontier.sh).
- Cells: 54-site bcc W qualification cells (not the production supercell). They are the bulk, a
  vacancy (53 sites), a Re substitution (54) and a He interstitial (55). The cells are listed in
  `outputs/qe-convergence/point_defects/cells/manifest.json`, which is git-ignored.
- Electrons and bands for the largest runs: 756 (bulk), 742 (vacancy), 757 (Re) and 758 (He); 454
  bands for the bulk.
- No further runs were affordable. The smearing scan was not repeated at a converged mesh, and no
  run combines the final mesh with the final cutoff.

## Acceptance rule

A scan point is accepted only if it and every more demanding point agree with the last
(reference) point within all criteria. The criteria are the `criteria_default` block of
[qe-protocols.yaml](../../deployments/common/qe-protocols.yaml).

| Observable | Limit | Applies to |
|---|---|---|
| Total energy | 1.0 meV/atom | each case |
| Derived energy (vacancy formation) | 0.02 eV | `E(vacancy) - (53/54) E(bulk)` |
| Force, max component difference | 0.005 eV/A | bulk and non-defect cases |
| Force, defect cases | 0.01 eV/A | vacancy, substitution, interstitial |
| Stress | 0.1 GPa | each case |

Tables below report the largest difference over the four cases against the reference point.
`dE` is the energy difference in meV/atom, `d(Ef)` the change of the vacancy formation energy in
eV, `max dF` the largest force difference in eV/A and `dP` the pressure difference in GPa.
The formation-energy differences in the tables were recomputed from the recorded bulk and
vacancy energies with 53/54; their displayed rounded values are unchanged. The earlier 0.9815
coefficient introduced a large absolute-energy error: for the first cutoff scan reference
(80/960 Ry), it gave 13.286368 eV instead of 2.942496 eV. This correction does not change
the recorded convergence decisions. Existing on-disk reports made with the old coefficient
are historical artifacts, not corrected reports; do not use their absolute derived energies.

## Run inventory

| Matrix | Config | Jobs | Layout | Result |
|---|---|---|---|---|
| First matrix | [frontier_point_defects.yaml](../qe_convergence/frontier_point_defects.yaml) | 64 (4 cases x 16 scan points) | cutoff, k-points, smearing, `conv_thr` scans at 60/720 Ry, 3x3x3 and `degauss` 0.010 | Only `conv_thr` qualified; 12 heavy runs were repeated on whole nodes |
| Follow-up 1, dense k-meshes | [frontier_point_defects_followup_kpoints.yaml](../qe_convergence/frontier_point_defects_followup_kpoints.yaml) | 16 | 8 GCDs and 8 k-pools per job | All COMPLETED |
| Follow-up 2, higher cutoffs | [frontier_point_defects_followup_cutoff.yaml](../qe_convergence/frontier_point_defects_followup_cutoff.yaml) | 16 | 8 GCDs per job | All COMPLETED |

The first matrix could not show a plateau for k-points: its spacings gave only 2x2x2, 3x3x3 and
4x4x4, and its cutoff scan stopped at 80/960 Ry. The follow-ups were designed to fix exactly these
two limits.

## k-spacing to mesh

The mesh follows the cell edge $L$ as $n = \lceil (2\pi/L)/s \rceil$ for spacing $s$.

| Spacing (1/A) | Mesh, 54-site cell | Irreducible k-points | Mesh, 1024-site cell (25.28 A) |
|---|---|---|---|
| 0.5 | 2x2x2 | 4 | 1x1x1 |
| 0.4 | 2x2x2 | 4 | 1x1x1 |
| 0.3 | 3x3x3 | 4 | 1x1x1 |
| 0.25 | 3x3x3 | 4 | 1x1x1 |
| 0.20 | 4x4x4 | 10 | 2x2x2 |
| 0.14 | 5x5x5 | 10 | 2x2x2 |
| 0.12 | 6x6x6 | 20 | 3x3x3 |
| 0.10 | 7x7x7 | 20 | 3x3x3 |

The meshes for the 1024-site cell follow from the formula and were not run.

### Why 3x3x3 is the 1,024-site equivalent of the qualified 7x7x7

All k-point evidence in this document comes from the 54-site qualification cell. A mesh count is
not transferable between cells of different size; the k-point **spacing** (sampling density) is.
The Brillouin zone shrinks as the real-space cell grows, and the electronic states fold into it,
so the cell-independent measure is

(mesh count) x (cell repeats) = effective sampling of one conventional bcc cell.

| Cell | Conventional repeats | Mesh | Effective conventional-cell sampling | Spacing (2pi/A) |
|---|---|---|---|---|
| 54-site qualification cell (9.48 A) | 3x3x3 | 7x7x7 | 21x21x21 | 0.095 |
| 1,024-site production cell (25.28 A) | 8x8x8 | 3x3x3 | 24x24x24 | 0.083 |

3x3x3 on the 1,024-site cell is therefore a slightly *denser* sampling than the qualified
7x7x7 on the 54-site cell, not a downgrade. The limit makes this intuitive: an infinitely large
supercell needs only the Gamma point, because its Brillouin zone has collapsed to a point.

"More k x k x k is better" holds only at fixed cell size (it is how the 54-site scan itself was
run, 3x3x3 up to 7x7x7). Carrying the mesh *count* 7x7x7 to the 1,024-site cell would be an
effective 56x56x56 per conventional cell, about 12.7x more k-points than anything tested, at
roughly proportional cost per SCF, with no scan evidence that it changes any observable within
the acceptance limits. It is rejected on that basis.

**Decision:** the stored invariant is the spacing 0.10 2pi/A from the 54-site study; its
translation to the 1,024-site cell is 3x3x3 (`kpts: [3, 3, 3]` in [campaign.yaml](campaign.yaml),
checked against `kpts_for_spacing`). The caveat stands that this mesh is derived, not run at
1,024 sites, and that 7x7x7 was the densest mesh tested, so no independent plateau beyond it is
shown (see Decisions and Open items).

## Results

### Cutoff scan, first matrix (3x3x3, `degauss` 0.010, `conv_thr` 1e-8)

| `ecutwfc`/`ecutrho` (Ry) | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted |
|---|---|---|---|---|---|
| 50/650 | 19.639 | 0.0008 | 0.0012 | 0.80 | no |
| 60/720 | 3.930 | 0.0001 | 0.0011 | 0.10 | no |
| 70/840 | 1.729 | 0.0000 | 0.0008 | 0.09 | no |
| 80/960 (reference) | 0 | 0 | 0 | 0 | yes |

### Cutoff scan, follow-up 2 (3x3x3, `degauss` 0.010, `conv_thr` 1e-8)

| `ecutwfc`/`ecutrho` (Ry) | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted | Bulk wall time (8 GCDs) |
|---|---|---|---|---|---|---|
| 70/840 | 5.200 | 0.0002 | 0.0006 | 0.02 | no | 12 min |
| 80/960 | 3.471 | 0.0002 | 0.0003 | 0.10 | no | 16 min |
| 90/1080 | 1.093 | 0.0000 | 0.0007 | 0.09 | no | 20 min |
| 100/1200 (reference) | 0 | 0 | 0 | 0 | yes | 21 min |

The total energy keeps dropping by about 2 meV/atom per step and is not at a plateau. The
formation energy, forces and stress are already inside their limits from 70/840 Ry. The dE limit
alone blocks acceptance, and only for the 90/1080 point.

### k-point scan, first matrix (60/720 Ry, `degauss` 0.010, `conv_thr` 1e-8)

| k-spacing (1/A) | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted |
|---|---|---|---|---|---|
| 0.5 | 15.814 | 0.2069 | 0.1027 | 0.37 | no |
| 0.4 | 15.814 | 0.2069 | 0.1027 | 0.37 | no |
| 0.3 | 15.017 | 0.4131 | 0.0391 | 0.29 | no |
| 0.25 | 15.017 | 0.4131 | 0.0391 | 0.29 | no |
| 0.2 (reference) | 0 | 0 | 0 | 0 | yes |

Pairs with equal values map to the same mesh. This scan was degenerate and carries no plateau.

### k-point scan, follow-up 1 (60/720 Ry, `degauss` 0.010, `conv_thr` 1e-8)

| k-spacing (1/A) | Mesh | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted | Bulk wall time (8 GCDs) |
|---|---|---|---|---|---|---|---|
| 0.20 | 4x4x4 | 1.354 | 0.0910 | 0.0122 | 0.07 | no | 32 min |
| 0.14 | 5x5x5 | 1.673 | 0.1289 | 0.0156 | 0.15 | no | 29 min |
| 0.12 | 6x6x6 | 0.728 | 0.0165 | 0.0082 | 0.07 | yes | 44 min |
| 0.10 (reference) | 7x7x7 | 0 | 0 | 0 | 0 | yes | 42 min |

The formation energy is the slowest observable: it moves by 0.09 and 0.13 eV from the coarse
meshes and then by 0.0165 eV at 6x6x6, which is close to the 0.02 eV limit.

### Smearing scan, first matrix (60/720 Ry, 3x3x3, `conv_thr` 1e-8, Marzari-Vanderbilt)

| `degauss` (Ry) | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted |
|---|---|---|---|---|---|
| 0.03 | 9.087 | 0.1645 | 0.0517 | 0.30 | no |
| 0.02 | 3.223 | 0.0203 | 0.0468 | 0.20 | no |
| 0.01 | 0.616 | 0.0004 | 0.0265 | 0.09 | no |
| 0.005 (reference) | 0 | 0 | 0 | 0 | yes |

The 3x3x3 mesh is too coarse for a small `degauss`, because the result is controlled by the mesh
rather than the smearing. Forces never reach the 0.005 eV/A limit. This scan must be repeated at
7x7x7 before `degauss` is accepted.

### `conv_thr` scan, first matrix (60/720 Ry, 3x3x3, `degauss` 0.010)

| `conv_thr` (Ry) | dE (meV/atom) | d(Ef) (eV) | max dF (eV/A) | dP (GPa) | Accepted |
|---|---|---|---|---|---|
| 1e-6 | 0.000 | 0.0000 | 0.0073 | 0.00 | no |
| 1e-7 | 0.000 | 0.0000 | 0.0017 | 0.00 | yes |
| 1e-8 (reference) | 0 | 0 | 0 | 0 | yes |

## Decisions

| Parameter | Decision | Reason | Residual risk |
|---|---|---|---|
| `conv_thr` | 1e-7 Ry | Cheapest value that passes every limit | None observed |
| k-mesh | 7x7x7 (spacing 0.10) | 6x6x6 passes, but its formation energy is 0.0165 eV from the reference against a 0.02 eV limit; 7x7x7 has the same 20 irreducible k-points and about the same cost | 7x7x7 is the densest mesh tested, so a plateau beyond it is not shown |
| Cutoff | 100/1200 Ry | The largest tested; the only observable still moving is the total energy | Not shown converged; absolute energies are good to a few meV/atom, differences are well within limits |
| Smearing | MV, 0.010 Ry | Keeps the value used in all scans, so the other results stay comparable | Forces moved 0.0265 eV/A with `degauss` at a coarse mesh; unknown at 7x7x7 |

The formation energy and forces are the quantities the campaign uses, and both are inside their
limits from 70/840 Ry. [campaign.yaml](campaign.yaml) now carries this provisional recommendation
(100/1200 Ry, k-spacing 0.10 as 3x3x3 on the 1,024-site cell, MV 0.010 Ry, `conv_thr` 1e-7) for
margin, not as a proven converged protocol; the earlier 60/720 Ry, 2x2x2, `degauss` 0.02,
`conv_thr` 1e-6 pre-dated the scans.

## Cr magnetic smearing scan (54-site cell, 60/720 Ry, 5x5x5, `conv_thr` 1e-7)

Config: [frontier_cr_magnetic.yaml](../qe_convergence/frontier_cr_magnetic.yaml), cells from
[cells_cr.yaml](../qe_convergence/cells_cr.yaml). The AFM state is the two-sublattice bcc
checkerboard (`Cr1`/`Cr2`, opposite starting moments). Single cell size, one cutoff, one mesh: this is
a screening result, not a qualification.

| `degauss` (Ry) | E(AFM)-E(NM) bulk (meV/atom) | Ef NM (eV) | Ef AFM (eV) | `vac_afm` total moment (muB) |
|---|---|---|---|---|
| 0.020 | -18.3 | 3.039 | 3.134 | -2.31 |
| 0.010 | -22.6 | 2.995 | 3.153 | -2.81 |
| 0.005 (reference) | -23.0 | 2.983 | 3.131 | -3.18 |

Relative to `degauss` 0.005 the AFM vacancy formation energy changes by 0.003 and 0.022 eV, and the
NM one by 0.056 and 0.012 eV. The analysis rejects both coarser values (largest derived-energy
change 0.026 eV at 0.010): forces change by 0.011 to 0.033 eV/A (limit 0.01 for defect forces) and
stress by 0.18 GPa at 0.010 (limit 0.1 GPa), and the magnetic state differs from the reference.

Total energies of the 12 finished, converged runs (Ry), read back from
`outputs/qe-convergence/cr_magnetic/runs/smearing`:

| `degauss` (Ry) | bulk NM | bulk AFM | vacancy NM | vacancy AFM |
|---|---|---|---|---|
| 0.020 | -13403.76572 | -13403.83820 | -13155.32451 | -13155.38864 |
| 0.010 | -13403.75907 | -13403.84862 | -13155.32117 | -13155.39744 |
| 0.005 | -13403.75747 | -13403.84895 | -13155.32053 | -13155.39937 |

Bulk AFM is 0.072 to 0.091 Ry below NM (0.99 to 1.24 eV per 54-atom cell), and `vac_afm` is 0.064
to 0.079 Ry below `vac_nm`; both separations widen as the smearing narrows, so neither is
converged against smearing.

Findings:

- The antiferromagnet is 18 to 23 meV/atom below the non-magnetic state, so the non-magnetic
  protocol is not valid for Cr. The spin state moves the vacancy formation energy by 0.10 to 0.16 eV,
  far above the 0.02 eV limit. The AFM value (about 3.1 eV) is the one to use.
- The AFM vacancy cell has an odd sublattice count (27/26), so it carries a net moment that grows as
  `degauss` narrows (-2.3 to -3.2 muB) and has not plateaued. This is a cell-size effect to test at
  larger cells, not a converged moment.
- `bulk_fm` (uniform starting moment) has not converged. As of 2026-11-28 the three runs are at
  SCF iterations 52 to 53 with an estimated accuracy of 3e-7 to 6e-7 Ry against the 1e-7 target,
  falling slowly; if they hit the wall-time limit they are recorded as unusable, not converged.
  The ferromagnetic state has already collapsed: net moment about 0.00 muB with an absolute
  magnetization of 0.08 to 0.22 muB per cell, and the running energies sit within about 0.4 mRy
  of `bulk_nm` (for example -13403.7653 against -13403.7657 Ry at `degauss` 0.020). No stable
  ferromagnetic state is shown at this lattice parameter; no FM conclusion is drawn.
- Not run: other cell sizes, a spin-spiral or other AFM orderings, a
  lattice-parameter relaxation in the AFM state, and the divacancy, dumbbell, H, Os and alloy cells
  (generated, not run through DFT).

## Open items

1. Repeat the smearing scan at 7x7x7 (and 100/1200 Ry), the one scan that needs a rerun.
2. Run one combined check (7x7x7, 100/1200 Ry) against a denser mesh or cutoff to show a plateau.
3. Qualify a larger cell. The mesh for the 1024-site cell is derived from the spacing, not run.
4. [campaign.yaml](campaign.yaml) carries the provisional recommendation (100/1200 Ry, 3x3x3,
   MV 0.010, `conv_thr` 1e-7). No test pins these numbers; they change again only with a
   qualified protocol. The 3x3x3 mesh is derived from the 0.10 spacing, not run at 1,024 sites.

## Reproduce

```text
python -m matsim_agents.workflows.numerical_convergence examples/qe_convergence/frontier_point_defects_followup_kpoints.yaml plan
python -m matsim_agents.workflows.numerical_convergence examples/qe_convergence/frontier_point_defects_followup_kpoints.yaml analyse
```

`submit` and `collect` come between these two. Each config writes to its own `out_dir`, so the
three matrices do not overwrite one another.
