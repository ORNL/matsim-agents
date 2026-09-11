#!/usr/bin/env bash
# Backward-compatible name; the implementation lives in install.sh.
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh" "$@"
