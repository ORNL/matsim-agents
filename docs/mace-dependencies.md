# MACE dependency contract

Dependency review date: 2026-09-02.

The supported MACE distribution is the latest published release,
`mace-torch==0.3.16`. Inspection of its PyPI wheel metadata
(`mace_torch-0.3.16-py3-none-any.whl`, SHA-256
`b80407edf6b2a1ec8523668c2a36852d20927ce1c3c56b70983a9f2dc53233ad`)
shows:

```text
Python >=3.9
torch >=1.12
e3nn ==0.4.4
```

Upstream MACE `develop` at commit
`59ad3a473ca02101a1cf02db242197e8d616dd11` identifies itself as 0.3.17,
raises its Python floor to 3.10, and still declares `e3nn==0.4.4`. Moving from
the published release to current source therefore does not resolve the
conflict.

HydraGNN main instead pins `e3nn==0.5.1`. These two exact e3nn constraints have
an empty intersection, so pip cannot produce one valid environment containing
both dependency sets. This is a package-metadata conflict, not merely a concern
about how old checkpoints were serialized.

The facility installers consequently use two matsim-owned environments:

```text
matsim-agents/.venv       HydraGNN, matsim-agents, optional FairChem/UMA
matsim-agents/.venv-mace  MACE compatibility process with e3nn 0.4.4
```

`.venv-mace` explicitly adds `.venv`'s site-packages through a `.pth` file. It
reuses the large, facility-qualified PyTorch, NumPy, and SciPy installations
while its own site-packages appear first and shadow e3nn with 0.4.4. Python's
`--system-site-packages` option is intentionally not used because it exposes
global interpreter packages, not packages belonging to another virtual
environment. The result must be treated as a separate process environment:
code running from it must not import HydraGNN, and code running from `.venv`
must not import MACE.

Use the canonical facility installer rather than installing MACE manually:

```bash
INSTALL_MACE=1 bash deployments/<facility>/setup/install.sh
```

The installer verifies the MACE package, e3nn version, calculators, and
matsim-agents MACE adapter. Accelerator execution and checkpoint inference must
still pass the on-machine qualification suite. Aurora is especially provisional:
the compatibility setup preserves its XPU PyTorch stack and repairs h5py, but
upstream MACE does not advertise Intel XPU as a supported accelerator.

The workflow exposes every foundation loader included in `mace-torch==0.3.16`:

| Family | Intended domain | Named models |
| --- | --- | --- |
| `mace_mp` | periodic materials | MP, MPA, OMAT, MATPES, and MH aliases |
| `mace_off` | organic molecules | OFF23 small, medium, large |
| `mace_omol` | organic molecules | OMOL-0 extra large |
| `mace_polar` | polarizable systems | Polar-1 small, medium, large |
| `mace_anicc` | H/C/N/O molecules | ANI-CC |
| `checkpoint` | checkpoint-dependent | local `.model` file |

These families are not interchangeable scientific approximations. In
particular, molecular models must not be treated as general bulk-crystal
potentials. MACE-MP and ANI-CC are declared MIT by the installed package;
MACE-OFF and MACE-OMOL are distributed under the non-commercial Academic
Software License. Check the upstream release terms for MPA, OMAT, MATPES, MH,
and Polar before use or redistribution.

Perlmutter compute nodes cannot fetch missing weights. Prefetch each selected
family/model from a networked login node before submitting work:

```bash
bash deployments/perlmutter/download/download-mace-perlmutter.sh \
	mace_mp medium-omat-0
```

Before collapsing these environments in a future release, check upstream
MACE's published metadata again. They may be combined only after MACE removes
or widens its exact e3nn pin and MACE inference/training passes against the same
e3nn version required by HydraGNN.
