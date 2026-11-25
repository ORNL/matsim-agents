"""Run E1 development questions for one candidate: ``python -m benchmarks.e1 MANIFEST -o DIR``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.e1.protocol import E1DevelopmentManifest, run_e1_development


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=None)
    args = parser.parse_args(argv)
    config = E1DevelopmentManifest.model_validate(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    result = run_e1_development(
        config,
        args.output,
        retry_errors=args.retry_errors,
        max_new_requests=args.max_new_requests,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if result.get("state") == "complete_with_errors" else 0


if __name__ == "__main__":
    sys.exit(main())
