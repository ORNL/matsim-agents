#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo "Usage: NBTAO5_PAW_RELEASE_APPROVED=YES $0 <authorized-paw-root> <external-case-dir>" >&2
  exit 2
fi
[[ "${NBTAO5_PAW_RELEASE_APPROVED:-}" == "YES" ]] || {
  echo "Refusing POTCAR assembly without explicit PAW release approval." >&2
  exit 2
}
PAW_ROOT="$(realpath -e "$1")"
CASE_DIR="$(realpath -e "$2")"
REPO_REAL="$(realpath -e "${REPO}")"
case "${CASE_DIR}/" in
  "${REPO_REAL}/"*)
    echo "Refusing to place licensed POTCAR content inside the Git worktree." >&2
    exit 2
    ;;
esac
ORDER_FILE="${CASE_DIR}/POTCAR.order"
[[ -s "${ORDER_FILE}" ]] || { echo "Missing ${ORDER_FILE}" >&2; exit 2; }
[[ ! -e "${CASE_DIR}/POTCAR" ]] || {
  echo "POTCAR already exists; refusing to overwrite it." >&2
  exit 2
}

TEMP_POTCAR="$(mktemp "${CASE_DIR}/.POTCAR.XXXXXX")"
trap 'rm -f "${TEMP_POTCAR}"' EXIT
COMPONENT_COUNT=0
while read -r COMPONENT EXPECTED_SHA256; do
  [[ -z "${COMPONENT:-}" || "${COMPONENT}" == \#* ]] && continue
  [[ "${EXPECTED_SHA256:-}" =~ ^[[:xdigit:]]{64}$ ]] || {
    echo "Invalid hash entry for ${COMPONENT}." >&2
    exit 2
  }
  COMPONENT_PATH="${PAW_ROOT}/${COMPONENT}/POTCAR"
  [[ -s "${COMPONENT_PATH}" ]] || { echo "Missing ${COMPONENT_PATH}" >&2; exit 2; }
  ACTUAL_SHA256="$(sha256sum "${COMPONENT_PATH}" | awk '{print $1}')"
  [[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256}" ]] || {
    echo "Hash mismatch for ${COMPONENT}; refusing assembly." >&2
    exit 2
  }
  cat "${COMPONENT_PATH}" >>"${TEMP_POTCAR}"
  COMPONENT_COUNT=$((COMPONENT_COUNT + 1))
done <"${ORDER_FILE}"
[[ "${COMPONENT_COUNT}" -gt 0 ]] || { echo "No components in ${ORDER_FILE}." >&2; exit 2; }
mv "${TEMP_POTCAR}" "${CASE_DIR}/POTCAR"
trap - EXIT
(cd "${CASE_DIR}" && sha256sum POTCAR > POTCAR.sha256)
printf 'Assembled %s hash-verified components outside the repository.\n' "${COMPONENT_COUNT}"