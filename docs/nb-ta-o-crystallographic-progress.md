# Nb-Ta-O Crystallographic Progress

**Scope:** source recovery, parent-derived NbTaO5 orderings, and a revised finite-domain inventory. No DFT calculations, VASP/QE runs, scheduler submissions, E2, or E3 were performed or authorized.

## 1. Concrete NbTaO5 Structures

The parent is the experimental zeta-Nb2O5 record [COD 1528678](https://www.crystallography.net/cod/1528678.html), from Laves, Petter, and Wulf, *Naturwissenschaften* 51 (1964), 633-634. The COD CIF reports B 1 1 2/b (No. 15; standardized here as C2/c), a=12.73 A, b=5.56 A, c=4.88 A, beta=105.05 degrees, Z=4. It has 28 sites: Nb8O20, all fully occupied. The source CIF is available from the COD record; download a local copy before generating orderings:

```bash
python scripts/prepare_nbtao5_orderings.py --parent /path/to/1528678.cif
```

The parent conventional cell is already a valid equimolar substitution cell: replacing four of its eight Nb sites gives Nb4Ta4O20, i.e. four NbTaO5 formula units in the same 28-site cell. Exhaustive enumeration gives 70 raw half-substitutions and 14 inequivalent arrangements under the parent C2/c symmetry. The script [prepare_nbtao5_orderings.py](../scripts/prepare_nbtao5_orderings.py) reproduces the enumeration. Its [manifest](../research/nb-ta-o/orderings/manifest.csv) records the parent-site indices, symmetry orbit sizes, COD ID, and source-CIF SHA-256; the 14 output CIFs are in [research/nb-ta-o/orderings](../research/nb-ta-o/orderings).

These are **scientifically motivated generated structures**, not experimentally observed NbTaO5 polymorphs. Their coordinates are copied from the Nb2O5 parent; only four Nb labels/species change to Ta. They do not model the reported hexagonal delta or orthorhombic gamma solid-solution families, and they do not represent configurational entropy or a disordered solution. Use them as a small-cell ordering sample only, not as evidence for an ordered 1:1 line compound or a complete solid-solution free energy.

## 2. Binary Oxide Records

COD results below were queried on 2026-11-03 using its spaced-formula syntax. Atom counts were parsed from CIFs. Multiple COD entries can be duplicate measurements or aliases, not separate phases. A COD formula hit is not by itself proof of a pure phase; a no-hit is not proof of nonexistence.

### Niobium Oxides

| Formula / record | Space group; cell (A, degrees) | CIF sites | Source and classification |
|---|---|---:|---|
| NbO, [COD 1010410](https://www.crystallography.net/cod/1010410.html) | Pm-3m; a=b=c=4.200 | 6 | Brauer (1940), *Ueber die Oxyde des Niobs*. Experimental record; vacancy ordering and relation to later records need review. |
| NbO, [COD 9008684](https://www.crystallography.net/cod/9008684.html) | Fm-3m; a=b=c=4.2097 | 8 | Wyckoff compilation (1963), rocksalt entry. Do not treat as a separately verified polymorph without checking its occupancy model. |
| NbO, [COD 9008782](https://www.crystallography.net/cod/9008782.html) | Pm-3m; a=b=c=4.2103 | 6 | Wyckoff compilation (1963), defective-rocksalt entry; likely overlaps the Pm-3m phase family. |
| NbO2, [COD 1520791](https://www.crystallography.net/cod/1520791.html), [1538348](https://www.crystallography.net/cod/1538348.html), [1538349](https://www.crystallography.net/cod/1538349.html), [1542078](https://www.crystallography.net/cod/1542078.html), [2106754](https://www.crystallography.net/cod/2106754.html), [9009093](https://www.crystallography.net/cod/9009093.html) | I41/a; a=13.66-13.71, c=5.964-5.987; alpha=beta=gamma=90 | 96 each | Experimental diffraction records/compilation, including Marinder (1961, 1962), Cheetham and Rao neutron diffraction (1976), and Pynn, Axe, and Thomas low-temperature distortions (1976). Treat these as one phase family pending CIF comparison, not six polymorphs. |
| NbO2, [COD 1548821](https://www.crystallography.net/cod/1548821.html) | P42/mnm; a=b=4.8463, c=3.0315 | 6 | Zhao and Kulik (2018), theoretical structure record. A predicted rutile topology, not an experimental NbO2 polymorph. |
| Nb2O5, zeta, [COD 1528678](https://www.crystallography.net/cod/1528678.html) | B 1 1 2/b (No. 15); 12.73, 5.56, 4.88; beta=105.05 | 28 | Laves, Petter, and Wulf (1964); experimental structure and the parent used for the orderings above. |
| Nb2O5, M, [COD 1528723](https://www.crystallography.net/cod/1528723.html) | I4/mmm; a=b=20.44, c=3.832 | 112 | Martin, Gruehn, and Andersson (1970); experimental record. |
| Nb2O5, N, [COD 1534156](https://www.crystallography.net/cod/1534156.html) | C2/m; a=28.51, b=3.83, c=17.48; beta=120.8 | 112 | Andersson (1967); the paper explicitly says prepared with small amounts of LiF. Exclude as a pure binary reference unless impurity effects are resolved. |
| Nb2O5, h, [COD 2107337](https://www.crystallography.net/cod/2107337.html) | P 1 2/m 1; a=21.153, b=3.8233, c=19.356; beta=119.8 | 99 | Kato (1976), structure refinement of h-Nb2O5; experimental record. |
| Nb2O5, high-temperature form, [COD 2310581](https://www.crystallography.net/cod/2310581.html) | P 1 2 1; a=21.16, b=3.822, c=19.35; beta=119.83 | 98 | Gatehouse and Wadsley (1964); experimental high-temperature form. Its relation to the h-Nb2O5 entry should be resolved before counting separate polymorphs. |

COD also lists [1531802](https://www.crystallography.net/cod/1531802.html) under a NbO-like formula, but its source paper and title identify Nb5O12F; it is excluded from pure NbO coverage.

### Tantalum Oxides

Exact COD formula searches returned no TaO, TaO2, Ta2O3, or Ta2O5 CIFs. The absence is a database result only.

| Proposed family | Best source found | Structural data and classification |
|---|---|---|
| Ta2O5, low-temperature L phase | Lehovec, *J. Less-Common Metals* 7 (1964), 397-406, [DOI](https://doi.org/10.1016/0022-5088(64)90036-0); hidden-order analysis by Lee et al. (2013), [DOI](https://doi.org/10.1103/PhysRevLett.110.235502) | Experimentally established orthorhombic low-temperature phase family. The complete experimental atom list, lattice, and CIF were not recovered in this pass; do not make a POSCAR from a formula or schematic. |
| Ta2O5, high-temperature H phase | Liu et al. (2007), [DOI](https://doi.org/10.1016/j.actamat.2006.11.031) | TEM resolves a tetragonal Ta sublattice, I41/amd, a approximately 3.86 A and c approximately 36.18 A. The 2026 article reports room-temperature stabilization by laser irradiation without additives. Experimental oxygen coordinates remain unresolved, so this is not a complete experimental CIF. The L-to-H transition is reported near 1360 C; rapid quenching or oxide stabilizers can retain H-like material. |
| Ta2O5, tetragonal-lambda H model | Lee and Hong, arXiv:2410.22017v3 (2026), [article](https://arxiv.org/abs/2410.22017), [DOI](https://doi.org/10.1038/s42005-026-02656-x) | **Theoretical generated model**, P41212, 168 atoms (24 Ta2O5), relaxed a=b about 7.6 A, c about 35.7 A. Built from orthorhombic lambda blocks and guided by the experimental Ta sublattice; DFT-relaxed coordinates are not an experimental structure. The accessible text says coordinates are in Supplementary Data 1 and 2. |
| Ta2O5, high-pressure A phase | *J. Solid State Chem.* (2026), [DOI](https://doi.org/10.1016/j.jssc.2026.126252) | Experimental high-pressure phase lead (W3O8(I)-related); full cell/coordinates were not accessible here. Optional for an ambient-pressure hull, relevant to a pressure-qualified domain. |
| TaO2 | Zhao et al. (2020), [DOI](https://doi.org/10.1063/5.0027018) | The paper reports no prior experimental crystalline oxide between Ta2O5 and TaO2 and **predicts Ta4O9**, P2/m, by cluster expansion and DFT. This does not verify TaO2. Classify TaO2 as unresolved/hypothetical; Ta4O9 is a predicted competitor. |
| TaO; Ta2O3 | COD exact-formula searches | No exact structure record identified. Keep unresolved; do not create coordinates or count these as verified phases. |

## 3. Mixed-Oxide Evidence and Missing Competitors

- Yamaguchi et al., “Formation and Transformation of Solid Solutions in the System Nb2O5-Ta2O5,” *J. Am. Ceram. Soc.* 71 (1988), [DOI](https://doi.org/10.1111/j.1151-2916.1988.tb05074.x), report a continuous hexagonal delta-Nb2O5/delta-Ta2O5 solid solution formed by heating amorphous material from simultaneous hydrolysis of Nb and Ta alkoxides. Across the series, a changes from 3.604 to 3.620 A and c from 3.923 to 3.883 A. At higher temperature, the series transforms to orthorhombic gamma-Nb2O5/beta-Ta2O5 solid solutions. This supports a mixed pentoxide composition range, not an ordered NbTaO5 line compound and not the zeta parent used above.
- Zafrir et al., “Preparation of a continuous series of solid solutions in the Ta2O5-Nb2O5 system,” *J. Solid State Chem.* 18 (1976), 377-380, [DOI](https://doi.org/10.1016/0022-4596(76)90121-3), is a second primary solid-solution lead. Exact structural coordinates were not recovered.
- A 2022 mixed-powder study, “On the Sintering Behavior of Nb2O5 and Ta2O5 Mixed Oxide Powders,” *Materials* 15, 5036, [DOI](https://doi.org/10.3390/ma15145036), reports an Nb4Ta2O15 diffraction component in a 70 mol% Nb2O5 / 30 mol% Ta2O5 feed annealed in air/hydrogen. In-situ XRD assigned 6% and 19.8% ternary phase at 890 C and 950 C; the paper also reports 20.4% and 20.8% in the ex-situ pellets. This is **experimentally reported diffraction evidence**, but no space group, refined atomic coordinates, or exact-formula COD CIF was found. Count it as an unresolved structure lead, not a verified input.
- No experimental NbTaO4 structure was found in the records checked. With O2-, NbTaO4 is a mixed-dioxide (+4 average cation valence) hypothesis, not a demonstrated Nb(V)/Ta(V) pentoxide solid solution. Classify proposed NbTaO4 orderings as **hypothetical** unless a primary structure source is supplied.
- Nb12O29 and Nb25O62 shear phases, and other Nb-O nonstoichiometric competitors, must be checked before making a broad Nb-Ta-O equilibrium-hull claim. Exact structure records were not recovered in this pass. Ta4O9 is a known computationally predicted competitor, not an experimental phase. These are domain-coverage leads, not automatic VASP tasks.

## 4. Revised Inventory and Counts

The former 20-slot floor is superseded. It counted proposed entries rather than distinct, source-verified phases. Current evidence supports these separate counts:

| Evidence class | Current count | Interpretation |
|---|---:|---|
| Exact-formula COD CIF records for NbO, NbO2, Nb2O5 | 15 records: 3, 7, and 5 respectively | Includes duplicate/related NbO2 reports, the theoretical COD 1548821 rutile entry, and the LiF-prepared Nb2O5 N entry. This is not 15 unique validated phases. |
| Preserved experimental parent CIF in this workspace | 1 | COD 1528678 zeta-Nb2O5, 28 sites. |
| Equimolar NbTaO5 parent-derived orderings | 14 | All symmetry-inequivalent 4-Ta substitutions in the 8-cation parent cell; 28 sites each. Scientifically motivated generated hypotheses. |
| Experimentally reported Ta2O5 phase families without a complete experimental CIF recovered here | 2 ambient families | Orthorhombic L and tetragonal H; the available full H coordinate model is theoretical. |
| Mixed-oxide phase diffraction lead without resolved structure | 1 | Nb4Ta2O15. |
| Proposed exact compositions still unresolved or hypothetical | Nb2O3, TaO, TaO2, Ta2O3, NbTaO4 | COD no-hits do not prove nonexistence. Do not issue candidate coordinates without a source or an explicitly labeled generated hypothesis. |

The currently generated structural campaign pool is therefore **14 NbTaO5 orderings**, not 20 unrelated slots. It is not yet a domain-complete hull: full Ta2O5 experimental input, Nb4Ta2O15 coordinates, and Nb-O/Ta-O nonstoichiometric coverage remain unresolved. Record aliases and phase families separately from unique relaxed structures in any future manifest.

## 5. Staged Convergence Design

The earlier 36-task blanket sweep should be conditional, not the default. The 28-atom zeta-parent ordering cells permit a compact mixed-cation test, while NbO2 represents a different electronic/structural class; the 112-site Nb2O5 records and 168-atom predicted H-Ta2O5 model are larger and should not be used as first-pass convergence proxies.

1. **Identity gate, zero VASP tasks:** decide the finite hull domain; acquire full source structures for selected Ta2O5 references and any included Nb4Ta2O15/shear phases; freeze PAWs and the common energy convention. Exclude unresolved rows rather than filling slots by prototype guess.
2. **Pilot, 18 tasks:** on Nb metal and one 28-atom parent-derived NbTaO5 ordering, use one baseline plus two one-factor checks each for cutoff, k-point density, electronic tolerance, and force tolerance (9 per structure). Hold geometry, magnetic state, PAW set, and non-varied settings fixed. Stop and revise settings if relative energies move by more than 5 meV/atom, the ordering changes, or forces/stress fail.
3. **Conditional extension, up to 18 more:** only after the pilot settings are stable, repeat the nine-point qualification on Ta metal and an experimental NbO2 structure to test Ta PAW and reduced-oxide behavior. Thus the previous 36 remains a maximum for these four representative classes, not an unconditional work package. A failed pilot triggers a settings review before further tasks.
4. **Ordering assessment:** a claim about the lowest parent-cell ordering requires evaluating all 14 unique structures at common converged settings. A 3-ordering screen may identify clear high-energy candidates, but cannot establish the global minimum. If screening energies cluster within 5 meV/atom, do not prune on static energy alone; relax the remaining orderings. Any broad solid-solution claim needs larger/disorder sampling and is outside this 14-cell set.
5. **Hull assessment:** first relax source-backed binary structures and approved mixed candidates. Add close polymorph pairs, the largest selected cells, and candidates within 10 meV/atom of the hull to the tighter final settings. The 14 orderings share the same parent cell, so do not qualify all 14 independently before convergence is established.

The count for a future run is not yet a single number: it is 18 pilot tasks, up to 18 conditional qualification tasks, plus the number of source-approved relaxations. The latter cannot be frozen until the Ta2O5 structure and hull domain are decided. No VASP/QE inputs or jobs have been created or launched.

## 6. Decisions That Actually Block VASP

1. Define whether the claim is a parent-cell NbTaO5 ordering comparison or an equilibrium ternary hull. A broad hull requires the missing Ta2O5 and nonstoichiometric/shear structures to be sourced or explicitly excluded by a bounded domain statement.
2. Choose whether all 14 orderings are needed for a minimum-ordering claim or whether an explicitly exploratory subset is acceptable.
3. Freeze a homogeneous, authorized PAW release (including Nb/Ta semicore choices), functional, magnetic policy, and consistent O2/reference energy convention. Existing local PAW hashes do not establish a homogeneous release.
4. Resolve inclusion of the experimental Nb4Ta2O15 diffraction lead and the predicted Ta4O9 competitor. Neither currently has an approved input structure in this workspace.

The missing TaO, TaO2, Ta2O3, Nb2O3, and NbTaO4 structures do not block a **bounded** calculation if those phases are explicitly outside its domain. They do block claims that depend on their stability or on an unrestricted equilibrium hull.