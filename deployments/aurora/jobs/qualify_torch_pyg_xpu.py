"""Aurora entry point for the shared Torch/PyG XPU qualification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deployments" / "common"))

from qualify_torch_pyg_device import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--device", "xpu"]
    raise SystemExit(main())