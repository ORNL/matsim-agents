# Frozen HydraGNN source for reproducible model execution

For the E4-B HydraGNN benchmark and workflows using its model stack,
MATSIM-AGENTS must use the following user-approved source version:

| Setting | Required value |
| --- | --- |
| Repository | `https://github.com/allaffa/HydraGNN.git` |
| Immutable commit | `81d5b88304f20bb45c840571e2e26c490d1b19da` |
| Branch context only | `fix/multibranch-bf16-autocast-dtype` |
| Model example directory | `examples/multidataset_hpo_sc26` |

The commit is the pin. Do not substitute the current branch tip, ORNL main,
or a package version number for this source revision. If the commit cannot be
retrieved, stop and report the failure rather than falling back to another
version. This choice does not change the default source for unrelated workflows.

## Clone only: preserve existing installations

From the MATSIM-AGENTS repository root, use a separate, previously nonexistent
directory:

```bash
export HYDRAGNN_REPO=https://github.com/allaffa/HydraGNN.git
export HYDRAGNN_REF=81d5b88304f20bb45c840571e2e26c490d1b19da
export HYDRAGNN_DIR="$(dirname "$PWD")/HydraGNN-e4b-frozen-81d5b883"

git clone --no-checkout "$HYDRAGNN_REPO" "$HYDRAGNN_DIR"
git -C "$HYDRAGNN_DIR" checkout --detach "$HYDRAGNN_REF"
test "$(git -C "$HYDRAGNN_DIR" rev-parse HEAD)" = "$HYDRAGNN_REF"
```

Run these commands sequentially and stop on any error. Cloning into a new
directory leaves existing HydraGNN source trees, checkpoints and Python
environments unchanged. Do not overwrite or switch a working checkout to
prepare this benchmark. An authenticated SSH URL for the same repository may
be used when HTTPS access is unavailable.

## Request the frozen source through the Aurora installer

The existing Aurora installer accepts `HYDRAGNN_REPO`, `HYDRAGNN_REF` and
`HYDRAGNN_DIR`. To request this exact source during an explicitly approved
installation:

```bash
HYDRAGNN_REPO=https://github.com/allaffa/HydraGNN.git \
HYDRAGNN_REF=81d5b88304f20bb45c840571e2e26c490d1b19da \
HYDRAGNN_DIR="$(dirname "$PWD")/HydraGNN-e4b-frozen-81d5b883" \
bash deployments/aurora/setup/install.sh
```

Unlike clone-only preparation, this command runs dependency installation and
package validation. It is not needed merely to upload checkpoints or prepare
E4-B, and must not be run without approval to modify the target environments.
The installer clones the selected repository when absent and fetches/checks out
the requested commit. Omitting `HYDRAGNN_REF` preserves an existing checkout,
but does not establish that it is the required frozen revision.

Aurora's dependency-installer checkout is separate from the model source:
`HYDRAGNN_INSTALLER_REPO` and `HYDRAGNN_INSTALLER_REF` select dependency recipes.
They must not replace the model-code pin above. Record their actual revisions
separately for environment reproducibility.

## Checkpoint upload and runtime selection

Upload the pretrained files under:

```text
<HYDRAGNN_DIR>/examples/multidataset_hpo_sc26/
  mlp_branch_weights.pt
  multidataset_hpo-BEST6-fp64/
    config.json
    multidataset_hpo-BEST6-fp64_epoch_97.pk
```

These are the filenames expected by the current E4-B audit. If the uploaded
checkpoint has a different layout or architecture, document and review that
change rather than renaming it to imply equivalence.

Set `HYDRAGNN_ROOT` to the frozen checkout and `HYDRAGNN_EXAMPLE` to its example
directory for the generated E4-B PBS scripts. Those scripts put the selected
source directories on `PYTHONPATH`; execution verifies both the source commit
and the imported HydraGNN package location.

Re-audit after every checkpoint upload:

```bash
.venv/bin/python -m benchmarks.e4_mlip.prepare \
  --example "$HYDRAGNN_DIR/examples/multidataset_hpo_sc26" \
  --output /path/to/new/e4b-preparation-directory
```

The audit records checkpoint/config hashes without running inference.
Freezing code does not establish checkpoint species support, numerical
accuracy, or accelerator compatibility. E4-B remains planning-only until its
inventory and model gates are satisfied and execution is explicitly approved.

See [E4-B methodology and readiness](../benchmarks/e4_mlip/README.md) and the
[machine-readable source pin](../benchmarks/e4_mlip/frozen-source.json).
