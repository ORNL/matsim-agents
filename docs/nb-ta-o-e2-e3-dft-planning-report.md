# Nb-Ta-O E2/E3 DFT Planning Report

**Status:** preparation and scientific-design review only. No VASP/QE calculation, scheduler submission, or E2/E3 evaluation was run for this report. Nothing here authorizes scientific execution. The target, phase domain, method signature, resources, and acceptance criteria remain proposals pending explicit approval.

## 1. Existing-Data and Provenance Inventory

| Source | What is present or documented | Qualification for a new VASP hull |
|---|---|---|
| Current `matsim-agents` checkout | Nb-Ta-O reference-generation code, hull infrastructure, a restricted NbO2 UMA/VASP AL template, and documentation of an earlier QE campaign. No frozen Nb-Ta-O VASP phase manifest, phase-energy set, or Nb-Ta-O VASP structure/result bundle was found. Generic VASP vendor testsuite POSCARs are unrelated. | No admissible VASP phase energies or geometries have been established. The AL YAML is not a hull manifest or E2/E3 reference dataset. |
| `../matsim-agents-dft-reference/` | Read-only inventory found VASP benchmark calculations and Nb-bearing alloy structures, including NbTaVHfZrTi and Hf-Mo-Nb-Ta-Ti boride records. Its structure metadata and VASP calculation names contain no Nb-Ta-O oxide phases or Nb-Ta-O oxide results. | The archive is adjacent and accessible, but its Nb-bearing alloy records are not Nb-Ta-O evidence. Its worktree contains unrelated modifications; it was not changed. |
| Documented historical QE campaign | Repository docs describe QE 7.6-dev/PBE single-point labels, Nb/O pseudopotentials, 80/640 Ry cutoffs, eight training and two validation frames, plus a proposed reference manifest. The exact archive `/global/cfs/cdirs/m5216/mlupopa/runs/portability/nb-ta-o--7llm-uma-al-qe-hull--653eb06--j59874625` is not mounted. The documentation reports ten completed QE children but eight native output directories; reconciliation is outstanding. | This is documentary evidence only in the current workspace. No raw energy, force, hash, input, or output was verified here. Even after authorized recovery, these QE energies and forces must remain separate from VASP labels and hulls. |
| Reference generator | `prepare_nb_ta_o_references.py` defaults to `--backend qe` and one AFLOW prototype per competing formula. `--backend vasp` is available, and curated structures can be supplied, but output is generated candidate material, not a frozen or scientifically qualified campaign manifest. | Do not treat generated structures, formula presence, or generator success as evidence of polymorph completeness or convergence. |
| Existing proposal/audit docs | The run-plan and E2/E3 audit docs describe a bounded domain and explicitly exclude QE values from E3. Those files are user-modified/untracked and were left unchanged. | Their candidate lists are planning context, not calculated VASP results or author approval. |

**Provenance rule:** preserve historical QE artifacts, methods, splits, and IDs unchanged under a QE-only namespace. A historical structure may be considered as a geometry/prototype lead after provenance review, but any E3 label or hull entry must be recomputed and qualified with the frozen VASP protocol. Do not recover values from prose, relabel QE as VASP, or combine codes because both use PBE.

## 2. Ranked Target-Composition Shortlist

The ranking is for choosing a bounded first scientific question, not a claim that any candidate phase exists or is stable.

| Rank | Candidate | Scientific basis | Main caveat |
|---|---|---|---|
| 1 | Equimolar cation composition NbTaO5, as a representative of the Nb2O5-Ta2O5 substitution series | Direct experimental literature reports continuous Nb2O5-Ta2O5 solid-solution series. The 1:1 cation composition lies on that compositional line and is a better-supported target for a mixed pentoxide question than an assumed ordered ternary compound. | This is a solid-solution composition, not proof of a distinct ordered NbTaO5 line compound. One ordered cell cannot represent configurational entropy or the full series. |
| 2 | NbTaO4, equivalent in reduced notation to (Nb0.5Ta0.5)O2 | A focused mixed-dioxide/order-versus-disorder question can be posed using NbO2/TaO2-related parent structures and symmetry-distinct cation orderings. It is also a useful contrast with the pentoxide solid-solution line. | The supplied literature leads do not establish an ordered NbTaO4 ground state. Its competing suboxide polymorphs and magnetic/electronic states need independent structure review. |
| 3 | NbO2 | It connects to the historical QE/UMA workflow and can be a tightly scoped VASP methodology check after approval. | The QE archive is inaccessible, the documented frames are single points rather than a complete relaxed phase set, and no existing VASP result was found. This is not a ternary target. |
| 4 | Nb2O5-Ta2O5 composition series beyond 1:1 | It is the strongest literature-grounded scientific family if the intended claim is substitution, ordering, or composition-dependent polymorphism rather than one formula. | It requires several compositions and multiple ordered/disorder approximants; that is a materially larger scope than one target composition. |

## 3. Recommended Target and Scientific Justification

**Recommendation for design review:** use NbTaO5 as the primary target composition, framed explicitly as the equimolar point on the reported Nb2O5-Ta2O5 solid-solution line. Retain NbTaO4 as a secondary, separate mixed-dioxide question. Do not describe either as a known ordered ternary phase unless an independently verified crystallographic source supports that exact structure and ordering.

The evidence basis is phase-equilibrium literature, not a computed energy ranking. Yamaguchi et al. report hexagonal delta solid solutions across the Nb2O5-Ta2O5 system and a higher-temperature transformation to orthorhombic gamma solid solutions. Zafrir et al. report preparation of a continuous solid-solution series in the same oxide system. These results establish a relevant composition family and polymorph leads; they do not supply VASP reference energies, a 0 K convex hull, or a crystallographic record proving ordered NbTaO5.

The primary question should therefore be bounded as: **among the predeclared structures in a finite, method-compatible Nb-Ta-O domain, where does the 1:1 Nb/Ta pentoxide composition lie relative to the domain hull, and what ordering spread is resolved by the tested approximants?** If only one NbTaO5 ordering is calculated, report its energy relative to the domain and state that configurational disorder/order is unresolved. A negative formation energy or a low energy above this finite hull is not by itself evidence for an experimentally realized ordered compound.

## 4. Proposed Finite Reference-Domain Inventory

Proposed domain name: `nb-ta-o-screen-v0.1`. It is a bounded screening domain, not an exhaustive phase diagram. Formula coverage and minimum starting-structure coverage are separate requirements.

| Subsystem | Required formulas / phase slots | Minimum distinct starting structures |
|---|---|---:|
| Elements | Nb, Ta, O2 | 3: one selected bulk reference each for Nb and Ta; one fixed-cell, spin-triplet O2 reference |
| Nb-Ta binary | NbTa | 1 validated ordering/prototype |
| Nb-O binary | NbO, NbO2, Nb2O3, Nb2O5 | 6: one for NbO; two distinct phase families for NbO2; one for Nb2O3; two distinct phase families for Nb2O5 |
| Ta-O binary | TaO, TaO2, Ta2O3, Ta2O5 | 6: one for TaO; two distinct phase families for TaO2; one for Ta2O3; two distinct phase families for Ta2O5 |
| Nb-Ta-O ternary | NbTaO4 and NbTaO5 | 3: at least two symmetry-distinct NbTaO4 orderings; one NbTaO5 pentoxide-line representative |
| **Total** | **14 reduced formulas, with polymorph/order slots as above** | **19** |

The 19 count is conditional on this exact v0.1 floor and counts distinct initial structure identities, not VASP invocations, retries, or independent thermodynamic configurations. Remove the exploratory NbTaO5 slot and the count is 18. Since NbTaO5 is recommended as the primary target, a minimal ordering comparison should use at least two distinct NbTaO5 approximants; that promotes the floor to **20**. More approximants are required before making a configurational or solid-solution claim.

The finite domain includes elemental references, both binary subsystems, and two ternary compositions so that the target cannot be assessed against unary endpoints alone. It deliberately omits other ternary stoichiometries, additional polymorphs/orderings beyond the declared minima, nonstoichiometry, oxygen vacancies/interstitials, high-pressure phases, surfaces, finite-temperature vibrational/configurational free energies, and a composition-resolved solid-solution phase diagram. Any claim must say “relative to the declared v0.1 DFT reference domain.”

## 5. Phase-Inclusion and Polymorph Rationale

1. **Curated structures first.** For Nb2O5 and Ta2O5, include multiple experimentally reported ambient-pressure polymorph families where structures can be verified. For the other required compositions, include at least one source-traceable candidate and retain its database/source ID. A database entry is a structure lead, not an energy label or proof of the lowest polymorph.
2. **AFLOW as candidate generation.** Use AFLOW prototype enumeration to fill gaps, record prototype IDs, decorations, generator version, and hashes, and deduplicate by structure matching. AFLOW must not silently satisfy a “two polymorphs” condition with two duplicates or two cells of the same family.
3. **Ternary ordering.** For NbTaO4, freeze two symmetry-distinct cation orderings before calculations. For NbTaO5, distinguish the pentoxide solid-solution composition from a line-compound assertion. If evaluating order, use multiple cation arrangements with an explicitly stated supercell and grouping plan.
4. **Suboxide and magnetic coverage.** NbO, NbO2, Nb2O3, TaO, TaO2, and Ta2O3 can have electronic or magnetic states that affect relative energies. Freeze multiple relevant spin starts for reduced phases and preserve the lowest converged state per declared structural family; the number of such starts is additional to the 19/20 structure floor.
5. **Explicit exclusions.** Do not add Nb12O29 or other shear/nonstoichiometric phases to the mandatory v0.1 set until both the phase record and structure/formula mapping are independently checked. Add them through a versioned domain change, not by silently editing an active manifest.

## 6. Proposed VASP Method Signature

The NbTaO5 pilot now has a fixed input signature, but it is not numerically convergence-qualified and does not define a runnable broad-hull manifest. The repository's VASP 6.6.1 executable/build identity is fingerprinted; job `9037133` failed before VASP started. The broader E2/E3 domain still requires approved phase structures and independent review.

| Field | Proposed setting | Freeze condition |
|---|---|---|
| Code | VASP 6.6.1; record binary/build identity and platform | Confirm licensed executable and version on the approved facility |
| Functional | PBE GGA | Use consistently for every elemental, binary, and ternary entry |
| PAW | PBE PAW datasets; candidate choices Nb_pv, Ta_pv, and O must be checked against the available licensed library | Record exact dataset labels, release, and SHA-256 for each POTCAR; no POTCAR contents in Git |
| Hubbard U | None in the baseline | If U is scientifically required, approve a single policy and re-evaluate all affected entries; never mix U and no-U energies |
| Spin | ISPIN=2 as the common initialization path; test and record magnetic starts for relevant reduced oxides | Freeze phase-specific MAGMOM starts and the rule for selecting among converged states |
| Spin-orbit | Off in the baseline | Treat as a sensitivity study only if relevant relative energies are comparable to the uncertainty budget |
| Cutoff | ENCUT set from the selected POTCARs, initially at least 1.3 times the largest ENMAX in the phase; use one common value | Determine the actual eV value by the convergence gate and freeze it for every hull entry |
| Cutoff | `ENCUT=max(600 eV,1.3*max ENMAX)=600 eV` from selected PAW metadata (`max ENMAX=400.000 eV`) | Fixed campaign choice; not independently cutoff-convergence-tested |
| Brillouin zone | Explicit Gamma-centered regular-mesh KPOINTS at target spacing 0.15 A^-1; five NbTaO5 POSCARs use 4x8x9. O2 uses separate Gamma-only 1x1x1 in a fixed 15 A cubic box | Recompute static KPOINTS from relaxed CONTCAR; record per-run dimensions and hashes. No KSPACING/KGAMMA in INCAR |
| Relaxation | IBRION=2, ISIF=3 for bulk phases, NSW=200, EDIFF=1e-6 eV, EDIFFG=-0.01 eV/A, ISYM=0 | Require electronic convergence at every ionic step, force and stress acceptance, and no unconverged final step |
| Smearing/energy | Common relax smearing proposal ISMEAR=0, SIGMA=0.05 eV; final static energy policy must be identical across periodic solids and tested for metallic references | Freeze whether final energies use a separate tetrahedron/static step; document the O2 energy convention and any correction |
| Precision | PREC=Accurate, LASPH=.TRUE., LREAL=.FALSE.; record all additional INCAR tags | Hash normalized INCAR, KPOINTS, POTCAR identities, VASP build, and relaxation protocol |
| Thermodynamic state | 0 K, nominal zero external pressure | State that finite-temperature and synthesis-kinetic effects are out of scope |

The selected local Nb/Ta/O component hashes and headers are recorded in the runtime manifest; user approval covers their private local-source use, but does not independently verify an official homogeneous release. The fixed cutoff and KPOINTS convention are selected and hashed, not convergence-tested. Freeze the O2 correction convention and phase-specific magnetic starts before any broad hull claim.

## 7. Convergence Qualification

The current fixed pilot qualification contains four baseline workflows: NbTaO5 ordering 01 (Q1), Nb bcc and Ta bcc (Q2), and fixed-cell O2 triplet (Q3). It is capped at 2.0 node-hours. Cutoff and k-point sweeps were removed by decision; these fixed choices are not claimed to be independently convergence-tested. These are acceptance checks, not achieved results:

- Increase ENCUT and density of k-points until relative phase energies change by no more than 5 meV/atom between the two tightest settings; also check that the target energy-above-hull changes by no more than 10 meV/atom.
- Require electronic SCF convergence for every ionic step and static calculation.
- Require ionic convergence and maximum final force <=0.01 eV/A; for bulk periodic cells require maximum absolute principal residual stress <=2 kbar (0.2 GPa).
- Generate the relaxation mesh from the initial POSCAR, then regenerate the explicit static mesh from the relaxed CONTCAR. Hash and record both KPOINTS files and dimensions.
- A VASP exit, `JOB DONE`, electronic convergence alone, or a walltime interruption is not a pass. Interrupted runs are not classified as physically unconverged without completed parseable results.
- Check multiple magnetic initializations for the reduced Nb/Ta oxides and compare final converged energies under the same numerical signature. Preserve all starts and failures.
 - For future reduced oxides, review relevant magnetic starts and preserve all attempts; this is outside the four-workflow NbTaO5 qualification.
- Relax bulk cells and internal coordinates at the declared pressure; keep the O2 cell fixed and verify its triplet spin state. The selected O2 box/cutoff are not independently convergence-tested.
- A static energy calculation is required after each relaxation and uses the relaxed structure with a newly computed explicit mesh.

Qualification runs, magnetic starts, stricter cutoff/mesh calculations, static steps, failures, and retries are extra invocations beyond the minimum structure count. No timing distribution is available for this Nb-Ta-O workload.

## 8. Hull Algorithm, Completeness, and Uncertainty

Use one versioned hull snapshot for each frozen domain and VASP signature. Parse composition from the actual final structure, compare it with the phase manifest, normalize energies per atom, calculate formation energies against the selected elemental references, and pass all compatible candidate phases to the existing pymatgen convex-hull implementation. Store each phase's raw total energy, formation energy, energy above hull, decomposition coefficients, final structure/hash, backend, method signature, corrections, and convergence record.

The existing `ReferenceCompletenessPolicy` checks elemental references, required formulas, binary subsystem coverage, and an optional ternary competitor. It does **not** presently enforce named phase IDs, a minimum count of distinct polymorph families, source qualification, formula-versus-structure agreement, or geometry convergence. The hull may therefore appear formula-complete while scientifically undersampled. Before calling the result domain-qualified, software must separately enforce:

1. **Formula gate:** every actual structure composition matches its declared formula and each required formula/subsystem is represented.
2. **Polymorph gate:** every predeclared phase ID and minimum family/order count is present, hash-stable, and calculated.
3. **Compatibility/convergence gate:** every energy has the same frozen VASP signature and passes electronic, ionic, force, and stress criteria.

Any failed gate means `provisional=true`, `domain_qualified=false`, with exact missing IDs and reasons. A qualified result remains relative to the declared finite domain; it is not a global phase-diagram completeness statement.

Report numerical convergence sensitivity, alternative magnetic minima, and O2-reference correction sensitivity separately. Keep raw energies immutable. Apply an O2 correction only if an approved, provenance-recorded policy is applied consistently to all oxygen-bearing entries; do not import corrected formation energies from a different code, PAW set, or database compatibility scheme. If the uncertainty interval for energy above hull overlaps zero or the domain's numerical/systematic sensitivity window, classify stability as unresolved rather than stable/unstable. PBE/PAW systematic error, polymorph incompleteness, configurational disorder, and finite-temperature effects are limitations, not quantities inferred from the hull algorithm.

**Required future software change:** extend the completeness policy/report to represent required phase IDs, polymorph families/minimum counts, formula validation, and convergence/compatibility gaps. Add focused tests for formula mismatch, missing polymorph family, SCF-converged but geometry-unconverged references, method/backend mismatch, fully qualified domain, and provisional-to-qualified transition. No such code change is made by this report.

## 9. E2 Incomplete-Evidence Packet

**Packet status today: can be assembled only as an insufficiency packet; no scientific hull answer is admissible.** A bounded E2 question could be: “Do the currently accessible records establish whether the equimolar Nb-Ta pentoxide composition is stable relative to the proposed `nb-ta-o-screen-v0.1` domain?” The correct answer from the present inventory is `unresolved / insufficient admissible evidence`, not stable or unstable.

The immutable packet should include:

- The claim, target formula, pressure/temperature scope, proposed domain version, and explicit exclusions.
- An evidence ledger stating that no Nb-Ta-O VASP energies or phase structures were found in this checkout or the inspected adjacent DFT archive.
- The QE campaign facts as documentary, unverified historical context only, including the inaccessible path, 8/2 frame counts, and unresolved 10-child/8-output-directory discrepancy.
- The list of missing VASP phase IDs, polymorph families, elemental references, method signature, convergence records, O2 convention, and independent review.
- An admissibility rule that forbids QE energy/force values and undocumented prose from satisfying VASP hull requirements.
- A fixed response schema allowing `supported`, `contradicted`, or `unresolved`, with evidence IDs and a reasoned follow-up. For this packet, only `unresolved` is supported by the evidence state.

LGEA and DEA must receive the same frozen packet and follow-up options; hidden answers and scoring labels remain separate. This is an E2 evidence-sufficiency case, not an E2 calculation or evaluation performed here.

## 10. E2 Complete-Evidence Packet (Future Gate)

A complete packet may be frozen only after the domain and method are approved and all required records exist. It must contain:

1. Domain version and SHA-256 of the complete manifest, with required and excluded formulas/phase IDs.
2. Structure sources, CIF/AFLOW IDs as applicable, structure generator versions/seeds, parsed formulas, raw and canonical hashes, and ordering/group IDs.
3. Exact VASP version/build, PAW identities/hashes, normalized INCAR/KPOINTS, O2 convention/correction, and method-signature digest.
4. All raw input/output paths and per-attempt provenance; no reconstructed or prose-only numbers.
5. For each phase: final composition, energy and units, final forces/stress, electronic/ionic convergence facts, and retry lineage.
6. Explicit formula, polymorph, and convergence-gate results plus the list of gaps; no silent omission of failed calculations.
7. Versioned hull output with formation energy, energy above hull, decomposition, numerical sensitivity/uncertainty, and domain-relative wording.
8. Two independent scientific reviews of admissibility, grouping, references, uncertainty, and claim limits.
9. Packet hash and a frozen E2 question/criterion/answer key before model access.

If any required gate is incomplete, downgrade to the incomplete packet rather than treating the hull plot as complete. No complete packet currently exists.

## 11. E3 Reuse and Leakage Controls

- QE energies and forces are never E3 train, validation, or test labels and never VASP hull entries. Preserve them in a separate historical QE catalog even if the original archive is later restored.
- A QE geometry can be considered only as a reviewed starting-structure lead. Recompute the final geometry and all needed labels under VASP; record both the source-geometry hash and new VASP structure/trajectory hashes.
- Build the E3 VASP reference set from scratch for the frozen domain. Maintain a distinct hull ledger and model-evaluation dataset so a phase label used to train a model cannot be presented as an independent prediction/evaluation result.
- Split by scientific groups, not frames: group related polymorphs, cation orderings/supercells, relaxation trajectories, duplicate structures, and same-prototype decorations as required by the task. Hash structures before assigning groups; keep each group wholly within train, validation, or test.
- Freeze validation-only thresholds and adaptation eligibility before model evaluation. Keep test labels sealed and use them once for final reporting; never use test errors to choose acquisitions, hyperparameters, or the target composition.
- Evaluate only model-system cells that pass support/runtime/dependency checks, but report every cell as supported/pass, supported/fail, insufficient evidence, unsupported, or blocked. A passing model does not replace domain-mandated VASP references.
- For every failing supported cell, compare C and D from the same initial checkpoint, candidate universe, VASP signature, information, retry policy, and attempted-label cap. Share a physical VASP result only for the same canonical structure and full signature; charge solver work once while preserving both policy requests and counterfactual costs.
- Keep scientific DFT needed to qualify the hull separate from the model-accuracy trigger. A model passing validation does not remove missing phase calculations; an incomplete hull is not a reason to adapt a passing model.

## 12. Exact Minimum Structure Count (Conditional)

For `nb-ta-o-screen-v0.1`, the minimum screening inventory is exactly **19 distinct starting structures**: 3 elemental references + 1 NbTa binary + 6 Nb-O structures + 6 Ta-O structures + 2 NbTaO4 orderings + 1 NbTaO5 representative. Excluding NbTaO5 gives 18. Requiring two NbTaO5 orderings for even a minimal target-ordering comparison gives **20**.

These are unique structure slots after source curation and before structural deduplication; duplicate source cells do not count twice. The count assumes one selected Nb and Ta elemental structure, one O2 cell, only the stated polymorph minima, and exactly the listed ternary orderings. It excludes all convergence sweeps, extra magnetic starts, disorder approximants beyond the stated minimum, final static runs if separately required, phonons, retries, and any domain expansion. If candidate structures relax to the same phase, the missing family must be replaced and the count rises. This is a conditional count, not a frozen manifest, execution count, resource estimate, or authorization.

## 13. Dependencies and Resource Estimate

**Software/data dependencies:** approved VASP 6.6.1 executable and license; access to the selected Nb/Ta/O PAW datasets; ASE and pymatgen structure/hull tooling; AFLOW prototype data and optionally pyXtal for candidate generation; structure sources with stable IDs and reuse terms; facility-specific launcher and a reviewed, versioned manifest. The repository has code/build recipes, but that does not prove all scientific or facility dependencies are available.

**Resource estimate:** the fixed four-workflow qualification is capped at 2.0 node-hours (one node, 30 minutes per workflow, serial). No VASP timing distribution exists; this is a ceiling, not a prediction. There are no qualified timings for the broader hull domain. Historical QE timings are not transferable to VASP. Record application time, scheduler elapsed time, nodes, accelerators, ionic steps, failures/retries, and setup overhead separately before estimating the broader domain. Monetary costs require a verified facility rate and billing unit.

Minimum scientific volume is the 19/20 distinct structure inventory, not a runtime estimate. Each structure may require one relaxation and may require an additional static calculation, extra spin starts, tighter-convergence qualification, or retry. All are separately counted in the ledger. No calculation or scheduler job was run or submitted for this report.

## 14. Manifests and Directory Plan

Proposed, not-created output root: `runs/nb-ta-o/e2e3/v0.1/`. Keep generated structures and raw VASP outputs out of Git unless repository policy explicitly permits selected small manifests. Store POTCAR identity and hash, not licensed POTCAR content.

Recommended immutable layout:

```text
runs/nb-ta-o/e2e3/v0.1/
  domain-manifest.json
  method-manifest.json
  structure-candidates/
  inputs/<phase-id>/<attempt-id>/
  outputs/<phase-id>/<attempt-id>/
  results/phase-results.jsonl
  results/hull-snapshot-v0001.json
  packets/e2-incomplete.json
  packets/e2-complete.json
  splits/e3-group-splits-v0001.json
  accounting/attempts.csv
  accounting/scheduler-jobs.csv
```

Each manifest entry must include schema/domain version, phase ID, declared and parsed formula, subsystem, polymorph family/order ID, source/database/prototype, generator/version/seed, source and canonical hashes, scientific group, relaxation flags, VASP signature, PAW identities/hashes, convergence criteria and outcomes, raw energy/force/stress units, attempt/retry lineage, corrections, and qualification state. The manifest must freeze before any calculation. Keep QE historical records in a distinct root/catalog; do not copy them into the VASP result tree under a VASP label.

The current generator can help prepare candidate geometry files only after review. Its QE default, one-prototype default, and formula-level completeness output must be overridden/extended for an approved VASP campaign; it must not write into an existing historical run directory.

## 15. Decisions Requiring Approval

1. Confirm the scientific objective: pentoxide solid-solution composition NbTaO5, mixed dioxide NbTaO4 ordering, or a broader Nb2-xTaxO5 composition series.
2. Approve the finite formula domain, exact phase IDs, source-structure records, polymorph-family minima, ternary orderings, and exclusions. Decide whether NbTaO5 needs at least two orderings in the minimum scope.
3. Select exact Nb/Ta/O PAW datasets, cutoff, k-point and smearing/static-energy protocols, spin-start policy, no-U baseline, O2 reference/correction, force/stress tolerances, and uncertainty thresholds.
4. Approve whether the claim is strictly a 0 K domain-relative hull, and whether any finite-temperature, disorder, or magnetic-order follow-up is in scope.
5. Review and approve the formula/polymorph/convergence completeness software changes and the tests listed in Section 8.
6. Confirm VASP license, PAW access, platform/runtime validation, output destination, account/project, bounded qualification request, node/time/accelerator caps, stop rules, and data handling.
7. Approve E2 question/criterion and independent reviewers; approve E3 supported model matrix, independent grouping/splits, validation thresholds, C/D budgets, and leakage controls.
8. Explicitly authorize each stage before it starts: qualification, reference generation, E2 assessment, E3 baseline evaluation, and any adaptive DFT. Authorization for one stage does not authorize the next.

## 16. GO / NO-GO

- **Design-preparation GO:** the candidate ranking, domain draft, method fields, evidence gaps, conditional structure count, and approval list are ready for scientific review.
- **Revised VASP execution NO-GO pending separate authorization:** the fixed method and resource ceiling are recorded, but the settings have not been tested; the prior Q1 attempt failed before VASP and cannot qualify them.
- **E2 scientific comparison NO-GO:** only an insufficiency packet is supportable; no complete, admissible VASP hull packet exists.
- **E3 VASP labels/model evaluation NO-GO:** there is no frozen VASP dataset, qualified domain, leakage-safe split, or approved model evaluation.
- **Stability claim NO-GO:** no conclusion that NbTaO5, NbTaO4, NbO2, or any other Nb-Ta-O phase is stable or unstable follows from this report.

**Bottom line:** proceed to author/scientist review of the target and v0.1 scope only. Do not run VASP/QE, submit HPC jobs, or initiate E2/E3 evaluations until the required decisions are approved and separately authorized.

## References and Provenance Notes

- O. Yamaguchi et al., “Formation and Transformation of Solid Solutions in the System Nb2O5-Ta2O5,” *Journal of the American Ceramic Society* 71(5), 1988. DOI: [10.1111/j.1151-2916.1988.tb05074.x](https://doi.org/10.1111/j.1151-2916.1988.tb05074.x). Crossref metadata and abstract identify a continuous hexagonal solid-solution series and a higher-temperature orthorhombic series. This is phase-family evidence, not a VASP energy source.
- M. Zafrir et al., “Preparation of a continuous series of solid solutions in the Ta2O5-Nb2O5 system,” *Journal of Solid State Chemistry* 18(4), 377-380 (1976). DOI: [10.1016/0022-4596(76)90121-3](https://doi.org/10.1016/0022-4596(76)90121-3). Crossref metadata confirms title, authorship, journal, year, and pages. This is a structure/phase-equilibrium lead, not a computed reference.
- Repository evidence reviewed: `docs/numerical-results-run-plan.md`, `docs/e2-existing-evidence-audit.md`, `docs/e3-vasp-primary-design.md`, `docs/reference-hulls.md`, `docs/scientific-workflows.md`, `docs/hpc-validation.md`, `docs/vasp-aurora.md`, `deployments/perlmutter/jobs/prepare_nb_ta_o_references.py`, and `src/matsim_agents/discovery/stability.py`.
