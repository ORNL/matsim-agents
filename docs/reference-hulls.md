# Competing-phase reference hulls

A convex hull needs elemental chemical potentials and method-compatible competing compounds. Elemental references alone are enough to compute formation energies, but not enough to rule out decomposition into binary or ternary phases.

## Reference manifest

Campaign reference manifests support multiple polymorphs of the same formula:

```json
{
  "schema_version": 2,
  "phases": {
    "Nb": {
      "formula": "Nb",
      "path": "/path/Nb.extxyz",
      "source": "curated"
    },
    "NbO2-rutile": {
      "formula": "NbO2",
      "path": "/path/NbO2-rutile.cif",
      "source": "experimental_structure",
      "provenance": {"database_id": "..."}
    },
    "NbO2-distorted": {
      "formula": "NbO2",
      "path": "/path/NbO2-distorted.cif",
      "source": "experimental_structure"
    }
  },
  "completeness": {
    "required_formulas": ["NbO2", "Nb2O5", "TaO2", "Ta2O5"],
    "require_binary_subsystems": true,
    "require_ternary_competitor": false
  }
}
```

Each phase is relaxed with the campaign DFT backend. Its registry entry records the phase ID, formula, structure path and hash, total and formation energies, backend, method signature, provenance, and corrections. Entries with incompatible method signatures or backends are rejected.

New campaign-generated reference sets are named `campaign-<DFT-method-signature>`;
existing sets retain their supplied identifiers. This reference-set ID is a
label, not a content hash, and is distinct from the campaign ID and run directory
name. Compatible phases can accumulate under the same reference-set ID. See
[Campaign names and identifiers](./scientific-workflows.md#campaign-names-and-identifiers)
for assignment rules, configuration controls, and a concrete Nb-Ta-O example.

For an oxygen reference correction, set `energy_correction_eV_per_atom` on the O2 entry. The correction is applied per oxygen atom to the elemental chemical potential and retained in provenance.

MLIP unary-reference relaxation artifacts use a sanitized phase ID plus its
SHA-256 digest for both log and optimized-geometry filenames. Distinct IDs
such as `Nb/A` and `Nb-A` therefore cannot overwrite each other's artifacts
after filename sanitization. The unary cache key includes the artifact schema
version so campaign searches do not reuse caches created under the older,
collision-prone naming convention.

## Completeness and provisional hulls

Every hull snapshot includes a reference-completeness report. A hull is marked `provisional: true` when it lacks:

- an elemental reference;
- a formula required by the manifest;
- coverage for a binary chemical subsystem; or
- a ternary competitor when the policy requires one.

Missing regions are also listed under `undersampled_regions`. A provisional hull can guide exploration, but it must not support an unqualified stability claim.

## Nb-Ta-O preparation

`deployments/perlmutter/jobs/prepare_nb_ta_o_references.py` generates elemental Nb, Ta, and O2 structures plus AFLOW prototype seeds for a documented set of alloy, oxide, suboxide, and mixed-oxide formulas. Use `--max-prototypes-per-formula` to retain more polymorphs and `--curated-manifest` to merge trusted structures.

AFLOW-generated structures are reference candidates, not proof that the lowest-energy polymorph has been found. Production claims should use curated structures where available and retain multiple plausible polymorphs. Formulas with no compatible AFLOW prototype remain listed as missing, keeping the resulting hull provisional.
