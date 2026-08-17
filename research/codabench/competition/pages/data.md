# Data

## What you receive

The public data provides the **test geometries** you must run your model on:

```
public_data/
  structures_metadata.csv   # structure_id,file_path
  MATS-XXXX.xyz             # one geometry per test structure
reference_data/
  elemental_energies.json   # fixed DFT elemental references (for formation energies)
```

Each structure is identified only by an opaque `MATS-XXXX` key. The mapping to
real compound identities, material classes, and defect variants is
**intentionally withheld** to keep the benchmark blind.

## Structure files

`*.xyz` files are standard extended-XYZ with the lattice and atomic species.
The chemical species are real (your potential must run on the actual
elements); only the *identity/provenance* of each structure is anonymised.

## Reference labels

Ground-truth DFT labels (energies, forces, relaxed geometries) are held
**server-side** and are never distributed. They are used only by the scoring
program to evaluate your submission.

## Material classes (aggregate scope, not per-structure)

The hidden test set draws from:

1. 2D monolayers
2. Intermetallic phases
3. BCC high-entropy alloy (equiatomic, 128-atom supercells)
4. Critical minerals
5. Catalysis slabs

with point vacancies, antisites, and interstitials as defect variants. Which
class a given `MATS-XXXX` belongs to is not disclosed.

## Starting kit

The starting kit (`starting_kit/`) contains a full participant guide
(`README.md`), the model interface description (`MODEL_INTERFACE.md`), and an
`example_submission/` showing the expected file layout.
