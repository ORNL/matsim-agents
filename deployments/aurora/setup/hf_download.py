"""Download a HuggingFace repository snapshot to a local model directory.

Usage:
    python deployments/aurora/setup/hf_download.py <repo_id> [--name LOCAL_NAME]

Honors $HF_TOKEN (set it before calling for gated repos).  Uses
huggingface_hub.snapshot_download under the hood so partial downloads resume
cleanly on re-invocation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_DEST = Path(os.environ.get("MODEL_ROOT", Path.cwd() / "models"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id", help="e.g. mistralai/Mistral-Small-24B-Instruct-2501")
    p.add_argument("--name", default=None,
                   help="local directory name (defaults to repo basename)")
    p.add_argument("--dest", default=str(DEFAULT_DEST),
                   help="parent directory (default: %(default)s)")
    p.add_argument("--allow-patterns", default=None,
                   help="comma-separated patterns to keep")
    p.add_argument("--ignore-patterns", default="*.gguf,original/*,consolidated*",
                   help="comma-separated patterns to skip (default skips GGUF + raw shards)")
    args = p.parse_args()

    name = args.name or args.repo_id.split("/", 1)[-1]
    target = Path(args.dest) / name
    target.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"[hf-download] repo:   {args.repo_id}", flush=True)
    print(f"[hf-download] dest:   {target}", flush=True)
    print(f"[hf-download] token:  {'SET' if token else 'unset'}", flush=True)

    kwargs = {
        "repo_id": args.repo_id,
        "local_dir": str(target),
        "max_workers": 8,
        "token": token,
    }
    if args.allow_patterns:
        kwargs["allow_patterns"] = args.allow_patterns.split(",")
    if args.ignore_patterns:
        kwargs["ignore_patterns"] = args.ignore_patterns.split(",")

    snapshot_download(**kwargs)
    print(f"[hf-download] DONE -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
