# Fusion PFC: external nuclear evidence

Status: the import contract is implemented and tested on SYNTHETIC records. No nuclear
code has been run by this software, and no imported value in the repository is real.

## Contract

`matsim_agents.science.irradiation.nuclear_evidence.ExternalNuclearEvidence` validates one result from
an external code (for example FISPACT-II or SPECTRA-PKA). Every record must carry:

- the application and version, the nuclear-data library, and sha256 hashes of the
  inputs and outputs, validated separately;
- the material id, the irradiation scenario id, and the exposure and cooling times
  where the family needs them;
- quantities with units fixed per family (`FAMILY_QUANTITIES`) and one-sigma
  uncertainties. `None` means not reported; `0.0` is kept as reported.

## Rules per family

| Family | Rules |
|---|---|
| `pka_dpa` | A dpa value states its convention (NRT, arc-dpa, damage energy). dpa is never converted to surviving defects; that needs a cascade model. |
| `transmutation` | Isotope-resolved quantities, an isotope inventory, and an exposure time are required. `hydrogen_isotopes(q)` and `helium_isotopes(q)` return H/D/T and He-3/He-4 separately, for one named quantity. |
| `activation` | Exposure and cooling times are required. A shutdown dose rate needs both `dose_model` and `dose_geometry`. SPECTRA-PKA is refused as an activation source. |

Scientific qualification needs a stated basis and non-synthetic data.

## Transmutation configurations

`matsim_agents.workflows.radiation.transmutation_configs.transmuted_configuration` turns per-atom
transmutation evidence into product substitutions on seeded host sites. The
caller passes the site map, the `host` element and the `products`; the fusion
application uses host W, products Re and Os, and the 1,024-site BCC alloy from
[examples/fusion_bcc_1024](../../examples/fusion_bcc_1024/application.yaml).
It accepts `atom_fraction` or `concentration_appm`.

- Counts are rounded to the nearest integer.
- Gas isotopes are counted and not placed.
- No electronic-structure calculation is run.

## Recording

`to_ledger_record` writes the append-only ledger form; append it with the
evidence ledger API.

Tests: [tests/test_fusion_nuclear_evidence.py](../../tests/test_fusion_nuclear_evidence.py),
[tests/test_fusion_transmutation.py](../../tests/test_fusion_transmutation.py).
