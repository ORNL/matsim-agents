# Existing-data and duplicate audit

## Public refractory datasets

The following OSTI records were inspected through their public metadata and
descriptions: [3377969](https://www.osti.gov/biblio/3377969),
[2473294](https://www.osti.gov/biblio/2473294),
[2466323](https://www.osti.gov/biblio/2466323),
[2466324](https://www.osti.gov/biblio/2466324),
[2217644](https://www.osti.gov/biblio/2217644),
[2222910](https://www.osti.gov/biblio/2222910),
[2228839](https://www.osti.gov/biblio/2228839), and
[2222906](https://www.osti.gov/biblio/2222906).

The 2023/2024 binary archives describe 128-site disordered BCC alloys, PBE,
350 eV, 2x2x2, VASP 6.3.2, and many randomized arrangements. The NbTaV ternary
archive contains 10,500 structures over 105 compositions. The three recent
binary records cover Nb-Zr, Ta-Zr, and V-Zr. These are direct overlap warnings
for the proposed 22 quaternary/quinary/six-element systems, especially shared
binary/ternary subsets and structures; the OSTI archive payloads were not
downloaded or enumerated here, so no structure-level duplicate conclusion is
claimed. OSTI 3377969 is a correction/data bundle for NbZr/TaZr/VZr, not a
structure inventory substitute.

The public records confirm high-level BCC protocol context but do not expose
the full production inputs. The local E4 A/B input deck and its six-component
PAW payload are now available and have been compared to the private component
library; this recovers local PAW labels, release metadata, checksums,
`ISMEAR`, `SIGMA`, and the two-precision settings. It does not prove that the
local A/B deck matches every published production case or resolve public
archive overlap. `EDIFFG` is absent from the recovered deck.

The Nature Scientific Data article [First-principles data for solid solution
niobium-tantalum-vanadium alloys with body-centered-cubic
structures](https://www.nature.com/articles/s41597-024-03720-3) and its linked
OSTI record ([2217644](https://www.osti.gov/dataexplorer/biblio/2217644), DOI
[10.13139/OLCF/2217644](https://doi.org/10.13139/OLCF/2217644)) were inspected.
For its Nb-Ta-V dataset, the article specifies PBE, PAW, 350 eV, a 2x2x2 mesh,
normal precision, Methfessel-Paxton smearing with 0.2 eV broadening, and a
128-site BCC cell. It states that V and Nb potentials include s and p semicore
states and Ta includes p semicore states. The article describes two-stage
relaxation (low then normal precision), with energy change below 1e-4 eV and
maximum force below 1e-2 eV/A. It does not state a numeric `ISMEAR`, all
relaxation tags, or PAW release identifiers/checksums.

The article's linked Figshare records ([part 1](https://api.figshare.com/v2/articles/25824178),
[part 2](https://api.figshare.com/v2/articles/25838581)) list 16.1 GB and
14.6 GB archives and include small README files. Those READMEs confirm that
per-case `INCAR` and `POTCAR` files are present in the Nb-Ta-V archive, but
also state an energy-change criterion below 1e-6 eV, differing from the
article's 1e-4 eV. Neither large archive was downloaded or enumerated. These
sources concern Nb-Ta-V only; they do not establish the E4 Nb-Ta-V-Hf-Zr-Ti
production deck, Hf/Zr/Ti PAWs, or structure-level overlap. Treat the settings
as corroborating protocol context, not as verified E4 production inputs.

## Aurora environment and scheduler snapshot

The current ALCF Aurora PBS documentation and read-only live queue queries were
reviewed on 2026-11-06. `debug` is 1-2 nodes/1 hour; `debug-scaling` supports up
to 256 nodes/1 hour; `capacity` supports 1-16 nodes/168 hours with a 512-node
total assignment cap; and `prod` routes to execution queues whose minimum is
256 nodes. Therefore 32-128-node E4 tiers do not have a verified long-walltime
route. Live queue enabled/started status does not establish project-specific
queue authorization.

`sbank` shows all three named allocations active on Aurora, with user rows for
`mlupopa`, positive balances, and expirations recorded in
`allocation-status.json`. This verifies account membership and aggregate
balance, not queue acceptance or whether the balance covers the still-unknown
runtime. The `-A <project>` PBS syntax is documented. No test submission or job
was issued.

No VASP module is available through `module avail vasp`, but the repository
VASP 6.6.1 build and documented `/soft/applications/vasp/vasp.6.6.1/bin/vasp_std`
path are accessible as executables. The existing native launcher uses the
repository build. Current ALCF guidance recommends a newer MPICH module to
avoid a severe GPU-aware MPI performance regression; the shared native module
helper does not explicitly select it. Treat this as a mandatory pilot
configuration decision, not grounds to replace or edit the preserved launcher.

## Local E2 and ongoing work

- The E2 NbTaVHfZrTi suite records four 128-atom near-equiatomic structures
  (`ideal`, `random1`, `random2`, `random3`) under a distinct nonmagnetic
  345 eV / 0.10 eV-smearing protocol. The frozen suite has the ideal structure
  hash and source run identity. The exact ideal source case is mounted at
  `/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials/e2-dft-v1-20261104/cases/bcc_hea_NbTaVHfZrTi_equiatomic_128_ideal`;
  its `POSCAR` SHA-256 is
  `798e572ee9dd7c5a6a40a0a1c80989a0e1c259548f22d503a69ca1a1440044d7`,
  matching the frozen case record. The archive contains `POSCAR`, `INCAR.relax`,
  `INCAR.static`, `KPOINTS`, `POTCAR`, and relaxation/static outputs. Its PAW
  components are Hf_pv 06Sep2000, Nb_pv 08Apr2002, Ta_pv 07Sep2000,
  Ti_pv 07Sep2000, V_pv 07Sep2000, and Zr_sv 04Jan2005; the archived POTCAR
  hash is `40c87ed0dc4b827194210fd185433b977e945a39544e0e3dd0f0905491151e2d`.
  This is recovered E2 provenance only, not verification of the E4 350 eV
  production PAWs or input deck. The four E2 cases are excluded from E4 reuse.
- The E2 Cantor suite concerns a 108-atom Cr-Mn-Fe-Co-Ni cell and the four
  named configurations (`ideal`, `random1`, `random2`, `random3`). It documents
  qualified statics only for ideal and random1; both ionic relaxations remain
  unqualified and random2/random3 were interrupted. The selected E4 candidates
  are 128 atoms, have differing compositions/occupations, and are not exact
  duplicates or integer supercells of the accessible 108-atom cases. Public
  archive overlap is still unresolved.
- `research/paper/e2/evidence/archive-extract.json` indexes the prior LiFePO4
  ideal, Li-vacancy, and Fe/Li-antisite runs. It records output hashes,
  VASP 6.6.0, 28-atom cell, PAW labels, and settings. The LiFePO4 reference
  POSCAR/INCAR/KPOINTS/POTCAR inputs are now mounted in
  `matsim-agents-dft-reference/dft_reference/vasp/calculations/`. The frozen
  pristine 112-atom candidate is an exact 2x2x1 periodic replication of the
  archived ideal structure and should be excluded from new pristine
  production. The selected antisite candidates do not match the accessible
  28-atom antisite/vacancy occupations under the tested 2x2x1 comparison;
  broader private/public archive uniqueness remains unresolved.
- Existing NbTaV calculations and other mounted Nb-Ta-O work are not treated as
  equivalent to these proposed alloys, but were checked as context. No source
  structure archive for the ongoing NbTaV campaign is present in the E2
  evidence tree.

## Findings and exclusions

The refreshed 16-case Stage 1 bundle has canonical identities and POSCAR/input
hashes. The selected BCC candidates were compared against the local E2
six-component cases and E4 A/B inputs; all eight are distinct from those
accessible records. The selected 128-atom FCC cases are distinct from the
accessible 108-atom E2 structures. The LiFePO4 pristine supercell is a
confirmed structural reuse of the archived ideal calculation; the three
selected antisite cases are distinct from the accessible 28-atom
antisite/vacancy cases. No exact same-size qualified or prior-incomplete
duplicate was found among the selected BCC/FCC cases or LiFePO4 antisites.

These are local-screen conclusions only. Public OSTI/Globus structure payloads
were not enumerated, and other private archives remain inaccessible. The
Stage 1 manifest therefore retains an incomplete external-duplicate status
and blocks execution; the planning inventory's placeholder hashes are not the
same artifact as the populated materialized Stage 1 manifest.