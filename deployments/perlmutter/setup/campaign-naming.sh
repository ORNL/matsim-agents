#!/bin/bash

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../../common/runtime-env.sh"

# Print the MLIP backend selected by a config's mlip block.
campaign_mlip_backend() {
  local python="$2" config="$3"
  mlip_backend "$python" "$config"
}

configure_campaign_run_name() {
  local repo="$1" mode="$2" backend="$3" mlip="${4:-}" workflow panel
  if [[ -z "${MATSIM_CAMPAIGN_SOURCE_REVISION:-}" ]]; then
    MATSIM_CAMPAIGN_SOURCE_REVISION="$(git -C "$repo" rev-parse HEAD)" || return
    local status
    status="$(git -C "$repo" status --porcelain)" || return
    MATSIM_CAMPAIGN_SOURCE_DIRTY=0
    [[ -z "$status" ]] || MATSIM_CAMPAIGN_SOURCE_DIRTY=1
  fi
  local short_revision
  short_revision="$(git -C "$repo" rev-parse --short=7 "$MATSIM_CAMPAIGN_SOURCE_REVISION")" || return
  panel=7llm
  [[ "$mode" != "single-llm-once" ]] || panel=1llm
  case "$mode" in
    single-llm-once|debate-only) workflow="$panel-debate" ;;
    mlip-only)
      [[ -n "$mlip" ]] || { echo "ERROR: mlip-only naming requires the MLIP backend" >&2; return 2; }
      workflow="$panel-$mlip-screen"
      ;;
    dft)
      [[ -n "$mlip" ]] || { echo "ERROR: dft naming requires the MLIP backend" >&2; return 2; }
      workflow="$panel-$mlip-al-$backend"
      [[ "${MATSIM_CAMPAIGN_DFT_REFINE:-1}" != "1" ]] || workflow="$workflow-hull"
      ;;
    *) echo "ERROR: unsupported campaign naming mode: $mode" >&2; return 2 ;;
  esac
  MATSIM_CAMPAIGN_RUN_TAG="nb-ta-o--$workflow--$short_revision"
  export MATSIM_CAMPAIGN_RUN_TAG
  export MATSIM_CAMPAIGN_SOURCE_REVISION MATSIM_CAMPAIGN_SOURCE_DIRTY
}
