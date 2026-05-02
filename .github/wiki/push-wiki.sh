#!/usr/bin/env bash
# push-wiki.sh — Clone the GitHub wiki repo and push the staged wiki pages.
#
# Prerequisites:
#   1. Enable the wiki from GitHub: Settings → General → Features → Wikis
#   2. Create the first page manually via the GitHub web UI (this initialises
#      the underlying git repo at https://github.com/ORNL/matsim-agents.wiki.git)
#   3. Run this script from the matsim-agents root:
#        bash .github/wiki/push-wiki.sh
#
# The script clones the wiki repo, copies the staged .md files, commits, and
# pushes. Re-running it is safe — it will update existing pages.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WIKI_SRC="${REPO_ROOT}/.github/wiki"
WIKI_REMOTE="https://github.com/ORNL/matsim-agents.wiki.git"
WIKI_DIR="$(mktemp -d)"

echo "Cloning wiki repo → ${WIKI_DIR}"
git clone "${WIKI_REMOTE}" "${WIKI_DIR}"

echo "Copying wiki pages..."
for f in "${WIKI_SRC}"/*.md; do
    name="$(basename "${f}")"
    cp "${f}" "${WIKI_DIR}/${name}"
    echo "  copied ${name}"
done

cd "${WIKI_DIR}"
git add -A

if git diff --cached --quiet; then
    echo "No changes to push."
else
    git commit -m "docs: update wiki pages from .github/wiki"
    git push
    echo "Wiki updated successfully."
fi

cd /
rm -rf "${WIKI_DIR}"
