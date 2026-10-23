# Sourced by setup_matsim_<facility>.sh: export DFT launcher and companion-venv defaults.
# Usage: source matsim-runtime-env.sh <facility> <matsim-root>
# Any variable already set is kept; a default is exported only if its target exists.

_matsim_facility="$1"
_matsim_root="$2"

_matsim_default() {
    if [[ -n "${!1:-}" ]]; then
        export "${1?}"
    elif [[ -e "$2" ]]; then
        export "$1=$2"
    fi
}

# DFT launchers are only defaulted when the binary they wrap is built.
_matsim_qe_prefix="${QE_PREFIX:-${_matsim_root}/external/quantum-espresso}"
if [[ -x "${PW_BIN:-${_matsim_qe_prefix}/install-gpu/bin/pw.x}" ]]; then
    _matsim_default MATSIM_QE_LAUNCHER \
        "${_matsim_root}/deployments/${_matsim_facility}/launchers/run-pw-gpu-${_matsim_facility}.sh"
fi
_matsim_default MATSIM_QE_PSEUDO_DIR "${_matsim_qe_prefix}/src/pseudo"

_matsim_vasp_root="${VASP_ROOT:-${_matsim_root}/external/vasp6/src/vasp.6.6.1}"
if [[ -x "${VASP_BIN:-${_matsim_vasp_root}/bin/vasp_${VASP_VARIANT:-std}}" ]]; then
    _matsim_default MATSIM_VASP_LAUNCHER \
        "${_matsim_root}/deployments/${_matsim_facility}/launchers/run-vasp-gpu-${_matsim_facility}.sh"
fi
_matsim_default MATSIM_VASP_POTCAR_DIR "${_matsim_root}/external/vasp6/potcar/potpaw_PBE.64"

# UMA and MACE need their own venvs (incompatible torch/e3nn pins).
_matsim_default MATSIM_FAIRCHEM_VENV "${_matsim_root}/.venv-uma"
_matsim_default MATSIM_MACE_VENV "${_matsim_root}/.venv-mace"

for _matsim_var in MATSIM_QE_LAUNCHER MATSIM_QE_PSEUDO_DIR MATSIM_VASP_LAUNCHER \
    MATSIM_VASP_POTCAR_DIR MATSIM_FAIRCHEM_VENV MATSIM_MACE_VENV; do
    echo "  - ${_matsim_var}=${!_matsim_var:-<unset>}"
done

unset -f _matsim_default
unset _matsim_facility _matsim_root _matsim_qe_prefix _matsim_vasp_root _matsim_var
