#!/bin/bash
# Activate the checkout's Aurora Python environment.

configure_matsim_aurora_source() {
    local repo="${1:?matsim-agents checkout is required}"
    export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

activate_matsim_aurora_python() {
    local repo="${1:?matsim-agents checkout is required}"
    local venv="${2:-${MATSIM_AURORA_VENV:-${repo}/.venv}}"
    if [[ ! -f "${venv}/bin/activate" ]]; then
        echo "ERROR: Aurora virtual environment not found: ${venv}" >&2
        return 2
    fi

    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    configure_matsim_aurora_source "${repo}"
}