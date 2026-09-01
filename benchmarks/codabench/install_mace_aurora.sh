#!/usr/bin/env bash
# install_baselines_aurora.sh
#
# Add MACE-MP-0, UMA, and AllScAIP to the currently-active Python venv on Aurora
# (Intel PVC/XPU). Run AFTER activating the matsim-agents or HydraGNN venv.
#
# Known issues fixed here (Aurora-specific):
#   1. mace-torch >= 0.3.7 introduces openequivariance, which hard-asserts
#      CUDA or HIP at import time → not usable on Intel XPU.
#      Fix: pin mace-torch==0.3.6 (last release without that dependency).
#
#   2. The Aurora frameworks stack ships h5py 3.15.1 in system site-packages,
#      compiled against libhdf5.so.310 which is not in LD_LIBRARY_PATH on UANs.
#      When mace-torch pulls h5py as a dependency pip sees it as "already
#      satisfied" and leaves the broken system copy in place.
#      Fix: force-reinstall h5py from PyPI; the wheel bundles its own HDF5.
#
#   3. torch >= 2.6 defaults to weights_only=True in torch.load; e3nn's
#      constants.pt contains a `slice` global that is not allowlisted.
#      Fix: export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (patched into activate).
#
#   4. fairchem-core (UMA / AllScAIP) requires huggingface_hub for model
#      weight download.  After installing, run:
#        huggingface-cli login
#      and accept the FAIR Chemistry License at:
#        https://huggingface.co/facebook/UMA
#        https://huggingface.co/facebook/AllScAIP
#
# Usage:
#   module load frameworks/2025.3.1
#   source /path/to/venv/bin/activate
#   bash install_baselines_aurora.sh

set -Eeuo pipefail

MACE_VERSION="0.3.6"

hr()     { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }

# ── Require an active venv ────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: no venv is active. Activate the matsim-agents venv first."
  echo "  source /path/to/matsim-agents-venv/bin/activate"
  exit 1
fi
VENV="$VIRTUAL_ENV"
banner "Target venv: $VENV"
echo "Python: $(python --version) at $(which python)"

# ── Proxy (required for outbound pip on Aurora UANs) ─────────────────────────
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export ftp_proxy="$HTTP_PROXY"
export no_proxy="admin,*.hostmgmt.cm.aurora.alcf.anl.gov,*.alcf.anl.gov,localhost"

# ── Fix 2: h5py ───────────────────────────────────────────────────────────────
banner "Fix 2: reinstall h5py from PyPI (avoid broken system libhdf5.so.310)"
# --no-deps: h5py's only runtime dep is numpy, which must NOT be overwritten
# (numpy is pinned to 1.26.4 for HydraGNN compatibility).
pip install --force-reinstall --no-deps h5py 2>&1 | tail -4
python -c "import h5py; print(f'  h5py {h5py.__version__} OK  ({h5py.__file__})')"

# ── Fix 1: pin mace-torch ────────────────────────────────────────────────────
banner "Install mace-torch==${MACE_VERSION} (no openequivariance, XPU-safe)"
pip install "mace-torch==${MACE_VERSION}" 2>&1 | tail -6
python -c "import mace; print(f'  mace {mace.__version__}  ({mace.__file__})')"

# ── Fix 3: persist env var in activate ───────────────────────────────────────
banner "Fix 3: persist TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD in venv activate"
ACTIVATE="${VENV}/bin/activate"
MARKER="# >>> mace-aurora-fix <<<"
if ! grep -q "$MARKER" "$ACTIVATE"; then
  cat >> "$ACTIVATE" <<'EOF'

# >>> mace-aurora-fix <<<
# torch >= 2.6 weights_only=True breaks e3nn constants.pt (slice global).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# >>> end mace-aurora-fix <<<
EOF
  echo "  Patched $ACTIVATE"
else
  echo "  Already patched — skipped."
fi

# Re-export now so the verify step works without re-sourcing activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ── Verify ───────────────────────────────────────────────────────────────────
banner "Verify: from mace.calculators import mace_mp"
python -c "from mace.calculators import mace_mp; print('  mace_mp importable: OK')"

hr
echo "Done. MACE-MP-0 is ready in: $VENV"
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 will be set automatically on activate."
hr

# ── fairchem-core (UMA + AllScAIP) ──────────────────────────────────────────
banner "Install fairchem-core (UMA + AllScAIP)"
# fairchem-core requires torch (already present) and torch-geometric (present).
pip install fairchem-core 2>&1 | tail -8
python -c "
from fairchem.core import FAIRChemCalculator, pretrained_mlip
print('  fairchem-core importable: OK')
print('  Available models:', pretrained_mlip.available_models)
"

banner "Install / upgrade huggingface_hub + huggingface-cli"
pip install --upgrade huggingface_hub 2>&1 | tail -4
huggingface-cli version 2>&1 || true

hr
echo "Done. UMA and AllScAIP packages installed in: $VENV"
echo ""
echo "NEXT STEPS — model weights require HuggingFace authentication:"
echo "  1. huggingface-cli login"
echo "  2. Accept UMA license:      https://huggingface.co/facebook/UMA"
echo "  3. Accept AllScAIP license: https://huggingface.co/facebook/AllScAIP"
echo "Weights download automatically on first use of from_checkpoint()."
hr
