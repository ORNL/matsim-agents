# Nb-Ta-O Stage A Completion Report

> Historical inventory audit. The source-backed crystallographic update, generated NbTaO5 CIFs, revised inventory, and staged convergence protocol are in [nb-ta-o-crystallographic-progress.md](nb-ta-o-crystallographic-progress.md). The provisional 20-slot count and its 36-task sweep below are superseded by that update.

**Status: incomplete; not execution-ready.** This report audits the provisional 20-slot proposal and defines a bounded preparation protocol. It does not authorize or perform VASP/QE calculations, scheduler submissions, E2 evidence assessment, or E3 MLIP evaluation. NbTaO5 and NbTaO4 remain provisional composition choices only; no crystallographic structure candidate has been approved.

## 1. Audit Result

The original 20 is a count of intended structure slots, not a validated inventory. The bundled AFLOW generator is not a substitute for crystallographic provenance: it decorates generic prototypes and can emit mappings from unrelated prototype chemistries. A temporary VASP-format generation check confirmed that its prototype IDs, space groups, and atom counts describe generated hypotheses, not experimentally identified Nb/Ta oxide polymorphs. Those outputs were not retained as campaign structures.

COD exact-formula searches (2026-11-03) returned five Nb2O5 entries, seven NbO2 entries, four NbO entries, two Nb-Ta alloy entries, and no exact entries for NbTaO4, NbTaO5, Nb2O3, TaO, TaO2, Ta2O3, or Ta2O5. A COD no-hit is a database-coverage result, not evidence that a phase cannot exist. CIFs successfully parsed for COD 1528723 (Nb2O5, 112 atoms, I4/mmm), COD 1534156 (Nb2O5, 112 atoms, C2/m), COD 1548821 (NbO2, 6 atoms, P4_2/mnm), and COD 1523086/1537942 (NbTa, 2 atoms each, Im-3m). The N-Nb2O5 record 1534156 cites preparation with LiF and is not accepted as a pure-phase reference without review. The two Nb-Ta records are not evidence for two distinct orderings.

## 2. Proposed 20-Slot Inventory Audit

`Unfilled` means there is no approved structure or reproducible POSCAR for that slot. Database records listed as leads still require inspection of the original CIF, occupancy, chemical purity, phase conditions, and suitability for the declared comparison. Atom counts are the number of sites in the cited conventional CIF cell when parsed; a `1 f.u.` or `2 f.u.` primitive-cell proposal is not the cited CIF cell.

| Slot | Formula | Proposed phase / polymorph | Crystallographic source and identifier | Atoms | Role and disposition |
|---:|---|---|---|---:|---|
| 1 | Nb | bcc metal | ASE `bulk('Nb', 'bcc')` construction; no curated record frozen | 1 proposed primitive cell | Elemental reference; source and lattice choice need approval |
| 2 | Ta | bcc metal | ASE `bulk('Ta', 'bcc')` construction; no curated record frozen | 1 proposed primitive cell | Elemental reference; source and lattice choice need approval |
| 3 | O2 | isolated triplet molecule in a fixed periodic box | Computational molecular reference, not a crystal record | 2 atoms | Oxygen chemical-potential reference; box and energy convention need approval |
| 4 | NbTa | equiatomic bcc-derived alloy | COD 1523086 (also COD 1537942; both list Im-3m) | 2 in parsed CIF | Binary reference lead; site occupancies/order and whether this is a distinct ordered structure remain unresolved |
| 5 | NbO | cubic phase | COD 1010410, Pm-3m; CIF atom count not parsed | Unknown | Binary competitor lead; inspect vacancy ordering and CIF before acceptance |
| 6 | NbO2 | I4_1/a record | COD 1520791, I4_1/a; CIF atom count not parsed | Unknown | Binary polymorph lead; confirm phase label/conditions and relation to the second NbO2 family |
| 7 | NbO2 | rutile-type record | COD 1548821, P4_2/mnm; theoretical record by Zhao and Kulik (2018) | 6 | Binary competitor lead; theoretical CIF is not an experimental polymorph record |
| 8 | Nb2O3 | unspecified | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; remove from required inventory unless an exact phase record is supplied |
| 9 | Nb2O5 | M-Nb2O5 | COD 1528723, I4/mmm; Martin, Gruehn, and Andersson (1970) | 112 | Experimental polymorph lead; candidate pending CIF/source review |
| 10 | Nb2O5 | H-Nb2O5 | COD 2107337, P 1 2/m 1; Kato (1976) | Unknown | Experimental polymorph lead; CIF atom count and phase relation need verification |
| 11 | TaO | unspecified | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; no TaO POSCAR or polymorph claim |
| 12 | TaO2 | polymorph A | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; exact structure and phase conditions required |
| 13 | TaO2 | polymorph B | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; a second polymorph cannot be inferred from a second generic prototype |
| 14 | Ta2O3 | unspecified | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; remove from required inventory unless an exact phase record is supplied |
| 15 | Ta2O5 | polymorph A | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; obtain an inspectable crystallographic record and phase label |
| 16 | Ta2O5 | polymorph B | No exact COD entry found; no source or identifier selected | Unknown | Unfilled; select a demonstrably distinct phase family, not a duplicate CIF |
| 17 | NbTaO4 | ordering 1 | No exact COD entry; no parent structure, ordering, or identifier selected | Unknown | Unfilled; treat only as a hypothetical 1:1 mixed-dioxide composition, not an established line compound |
| 18 | NbTaO4 | ordering 2 | No exact COD entry; no parent structure, ordering, or identifier selected | Unknown | Unfilled; symmetry-distinctness and common parent lattice have not been established |
| 19 | NbTaO5 | ordering 1 | No exact COD entry; no parent structure, ordering, or identifier selected | Unknown | Unfilled; composition lies on a reported pentoxide solid-solution line, not proof of an ordered compound |
| 20 | NbTaO5 | ordering 2 | No exact COD entry; no parent structure, ordering, or identifier selected | Unknown | Unfilled; a second small-cell arrangement would sample ordering only, not represent the solid solution |

The COD leads above are traceable starting points, not approval of a VASP reference structure. Primary mixed-pentoxide literature reports Nb2O5-Ta2O5 solid solutions (Yamaguchi et al., DOI 10.1111/j.1151-2916.1988.tb05074.x; Zafrir et al., DOI 10.1016/0022-4596(76)90121-3). It does not establish ordered NbTaO5. NbTaO4 has charge-balanced average cation valence +4 if assigned O2-, so it is a mixed-dioxide hypothesis; its presence as a stable ambient ordered phase is not established by the records checked here.

## 3. Completeness and Scope Changes

**Essential before any finite-domain hull can be frozen:**

- Replace unsupported formula-only entries Nb2O3, TaO, and Ta2O3 with exact sourced phases, or remove them by an explicit domain decision. Do not let a stoichiometric formula alone satisfy a competitor slot.
- Obtain inspectable Ta-O structures, including the selected Ta2O5 phase families, and verify exact compositions, occupancies, atom counts, and phase identities.
- Decide whether NbTaO4 is retained as a hypothetical mixed-dioxide composition. If retained, select an experimentally or computationally justified parent lattice and enumerate cation orderings from that same parent; otherwise remove both slots.
- For NbTaO5, select an actual Nb2O5-Ta2O5 solid-solution parent and an explicit disorder/configuration sampling method. Two ordered approximants may be reported as a small ordering sample only; they cannot support a solid-solution thermodynamics claim.
- Define the hull claim as relative to an explicitly finite domain. A broad equilibrium claim additionally needs a source-backed review of nonstoichiometric/shear competitors, including Nb-O shear phases such as Nb12O29 and Nb25O62, and any relevant mixed Nb-Ta-O counterparts. These are not silently included in the current 20.

**Optional expansion after the finite domain is reviewed:** additional NbTaO5 configurations or special quasirandom structures; composition points other than 1:1 on the pentoxide solid-solution line; additional sourced polymorphs; oxygen-vacancy/interstitial structures; high-pressure phases; and vibrational/configurational free energies. Each expansion changes the manifest and counts.

## 4. Input Proposal and PAW Provenance

There are currently **zero structure-approved candidates**, so no candidate-specific POSCAR/INCAR/KPOINTS bundle is issued. Writing POSCARs for the unverified AFLOW decorations or assigning structures to the unfilled slots would falsely imply phase validation. Once a row is approved, the exact source CIF should be preserved and hashed, parsed composition and occupancies checked, and its POSCAR plus mesh generated from that cell. The provisional common setup below is a review proposal, not a frozen production signature:

| Item | Proposed value | Gate |
|---|---|---|
| Functional | PBE; no U in the baseline | Revisit U only with independent physical justification |
| PAW components | Nb `Nb_pv`, Ta `Ta_pv`, O `O` | Candidate set only; verify one authorized homogeneous release before use |
| ENCUT | 520 eV starting value | Qualify against the three-point cutoff campaign below |
| Electronic convergence | EDIFF = 1e-6 eV | Compare 1e-5 and 1e-7 eV |
| Relaxation | IBRION=2, ISIF=3, NSW=200, EDIFFG=-0.02 eV/A | Compare force thresholds -0.03 and -0.01 eV/A; check stress and cell stability |
| Spin | ISPIN=2 for transition-metal oxides; triplet O2 | Freeze per-site MAGMOM and magnetic patterns only after structures are selected |
| K points | Gamma-centered mesh generated at KSPACING=0.20 A^-1 | Compare 0.25 and 0.15 A^-1; record actual mesh per cell |
| Static energies | Optional, not in the initial relaxation count | Define smearing and energy convention consistently before hull assembly |

Local component metadata already documented in [the numerical-results run plan](numerical-results-run-plan.md): `Nb_pv` is `PAW_PBE Nb_pv 08Apr2002`, ENMAX 208.608 eV, SHA-256 `086061c276946e9fc55bb7633d9ce78b32a420dec762592491c07c1c47ded114`; `Ta_pv` is `PAW_PBE Ta_pv 07Sep2000`, ENMAX 223.667 eV, SHA-256 `ffb05e92ac3f68ca84c26af10a80f2fbb0cf79e16fcc96fcae98089455bbbaaa`; `O` is `PAW_PBE O 08Apr2002`, ENMAX 400.000 eV, SHA-256 `8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64`. These identify local files but do not establish an authorized, homogeneous parent release; the local tree has mixed dates and no verified release marker. The POTCAR payloads are not included in this report or Git.

## 5. Bounded Convergence and Acceptance Protocol

After four representative structures are approved (one metal/reference, one reduced oxide, one pentoxide, and one mixed-cation target or closest approved analogue), use a one-factor-at-a-time qualification. Use the same frozen PAW set, cell, magnetic state, and all non-varied settings within each comparison:

- Cutoff: 400, 520, and 600 eV. Compare fixed-geometry energies at 400 and 600 eV against the 520 eV baseline.
- K-point density: KSPACING 0.25, 0.20, and 0.15 A^-1. Compare fixed-geometry energies at 0.25 and 0.15 against 0.20.
- Electronic convergence: EDIFF 1e-5, 1e-6, and 1e-7 eV. Compare SCF convergence and energy at the same geometry.
- Ionic/cell convergence: force thresholds 0.03, 0.02, and 0.01 eV/A. Compare relaxed geometry, residual stress, and relative energies.
- Magnetic starts: use explicit FM and symmetry-inequivalent AFM starts only for selected structures where partially occupied d states and cell size make them meaningful. Record every start and retain the lowest converged state per declared phase. O2 is a fixed-cell triplet reference.
- Acceptance: the final cutoff/k-point/electronic/relaxation checks must change each relevant relative phase energy by at most 5 meV/atom, preserve the tested ordering, and meet the declared force/stress bounds. If the ordering gap is at or below 5 meV/atom or changes sign, report it as unresolved and expand convergence or configuration sampling; do not select a preferred phase from the noisy ordering.

Conditional qualification count: four representatives, each with one 520 eV / 0.20 A^-1 / 1e-6 / 0.02 eV/A baseline plus two cutoff checks, two k-density checks, two electronic checks, and two force-threshold relaxations: **36 calculation tasks total**. Four baseline tasks overlap the initial-relaxation set if those representatives are among the 20 approved structures; therefore the qualification adds 32 distinct tasks under that assumption. This is a planning count only, not a job submission or execution request.

## 6. Separate Work Counts

| Work category | Current approved count | Conditional planning count | Notes |
|---|---:|---:|---|
| Unique structure identities | 0 | 20 proposed slots | The 20-slot inventory is not validated; the five successfully parsed CIF record IDs include duplicate Nb-Ta measurements and an LiF-prepared Nb2O5 record |
| Initial structural relaxations | 0 | 20 if all slots are replaced/approved | One per unique approved structure; retries are not counted |
| Optional final statics | 0 | 0 initially; up to 20 optional | Add only after the relaxation and energy convention are approved |
| Magnetic comparisons | 0 | Up to 16 additional starts under the provisional allowance | Eight reduced-oxide structure slots x two alternate starts; actual count depends on sourced phases and magnetic feasibility |
| Convergence calculations | 0 | 36 total; 32 additional if four baselines overlap initial relaxations | Four representative structures and the bounded one-factor tests above |
| Uncertain/expanded requirements | Not countable | Separate versioned change | Additional orderings/SQS, shear phases, extra polymorphs, high-pressure phases, and retries are not in the 20-slot floor |

The conditional arithmetic is not an authorization. If the inventory is reduced or expanded, recalculate all categories from the approved manifest. E2 frozen-evidence assessment and E3 MLIP evaluation remain separate future activities and have zero starts here.

## 7. Exact Decisions Needed to Close Stage A

1. Approve a bounded scientific question: finite-domain ranking of the 1:1 pentoxide composition, ordering spread on a specified parent, or a wider composition-resolved solid-solution study. These are different workloads and claims.
2. Decide whether to retain NbTaO4 as a hypothetical mixed-dioxide composition; if yes, approve a parent structure family and require sourced TaO2/NbO2 references. Do not label it an established ordered phase.
3. Approve replacements/removals for the unverified Nb2O3, TaO, and Ta2O3 slots and decide whether shear/nonstoichiometric phases are in scope.
4. Select and approve exact Nb2O5 and Ta2O5 CIF records/polymorph families, including phase conditions and impurity/occupancy checks. Resolve the H-Nb2O5 atom count and inspect the source CIFs.
5. For NbTaO5, approve a specific solid-solution parent and an explicit ordering/SQS sampling scope. Two structures do not by themselves validate the solid solution.
6. Approve the Nb-Ta binary reference representation and the O2 box, triplet, and energy convention.
7. Verify the authorized PAW release and freeze hashes/species order; approve the convergence thresholds, magnetic-start policy, smearing/static-energy convention, and whether optional statics are required.

Until these decisions are made, Stage A remains **blocked at crystallographic identity and input generation**. No VASP binary, QE binary, scheduler, E2, or E3 process was run or started for this report.