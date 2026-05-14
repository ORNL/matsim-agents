#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_env.sh
#
# Create a local Python virtual environment and install everything needed
# to run matsim-agents *and* HydraGNN. PyTorch / PyG / HydraGNN are
# installed by delegating to HydraGNN's own installation scripts, which
# already handle the tricky platform matrix (laptops, workstations, and
# DOE supercomputers Frontier/Perlmutter/Aurora/Andes).
#
# Usage:
#   ./scripts/setup_env.sh [VENV_DIR]
#
# Environment overrides:
#   PYTHON          Python interpreter (default: python3)
#   HYDRAGNN_REPO   Git URL                (default: https://github.com/ORNL/HydraGNN.git)
#   HYDRAGNN_REF    Branch/tag/commit      (default: main)
#   HYDRAGNN_DIR    Pre-existing checkout  (skip clone if set)
#   PLATFORM        workstation | frontier-rocm71 | frontier-rocm64
#                   | perlmutter | aurora | andes
#                   (default: workstation -> uses ./install_dependencies.sh)
#   HYDRAGNN_EXTRAS Args forwarded to install_dependencies.sh
#                   (default: "all dev")
#   LLM_BACKENDS    matsim-agents LLM extras to install
#                   space-separated subset of: ollama vllm openai anthropic
#                   (default: "ollama vllm")
#   BOOTSTRAP_OLLAMA  1 to install Ollama, start its daemon, and pull
#                     OLLAMA_MODELS (workstation only). Default: 0.
#   OLLAMA_MODELS   Space-separated models to pull when BOOTSTRAP_OLLAMA=1
#                   (default: "qwen2.5:14b")
#   SKIP_HYDRAGNN   set to 1 to skip HydraGNN install entirely
#
# Examples:
#   # Local workstation (CPU or single GPU)
#   ./scripts/setup_env.sh
#
#   # Frontier (OLCF, ROCm 7.1)
#   PLATFORM=frontier-rocm71 ./scripts/setup_env.sh /lustre/orion/.../venv
#
#   # Perlmutter (NERSC)
#   PLATFORM=perlmutter ./scripts/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON="${PYTHON:-python3}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/ORNL/HydraGNN.git}"
HYDRAGNN_REF="${HYDRAGNN_REF:-main}"
PLATFORM="${PLATFORM:-workstation}"
HYDRAGNN_EXTRAS="${HYDRAGNN_EXTRAS:-all dev}"
LLM_BACKENDS="${LLM_BACKENDS:-ollama vllm}"
BOOTSTRAP_OLLAMA="${BOOTSTRAP_OLLAMA:-0}"
OLLAMA_MODELS="${OLLAMA_MODELS:-qwen2.5:14b}"
SKIP_HYDRAGNN="${SKIP_HYDRAGNN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-${THIRD_PARTY_DIR}/HydraGNN}"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "Python interpreter '$PYTHON' not found."
command -v git       >/dev/null 2>&1 || die "git is required."

# ------------------------------- Clone HydraGNN ----------------------------
if [[ "$SKIP_HYDRAGNN" != "1" ]]; then
    if [[ -d "${HYDRAGNN_DIR}/.git" ]]; then
        log "Reusing HydraGNN checkout at ${HYDRAGNN_DIR}"
        git -C "$HYDRAGNN_DIR" fetch origin "$HYDRAGNN_REF" || true
        git -C "$HYDRAGNN_DIR" checkout "$HYDRAGNN_REF"
        git -C "$HYDRAGNN_DIR" pull --ff-only origin "$HYDRAGNN_REF" || true
    else
        log "Cloning HydraGNN (${HYDRAGNN_REF}) -> ${HYDRAGNN_DIR}"
        mkdir -p "$(dirname "$HYDRAGNN_DIR")"
        git clone --branch "$HYDRAGNN_REF" "$HYDRAGNN_REPO" "$HYDRAGNN_DIR" \
            || git clone "$HYDRAGNN_REPO" "$HYDRAGNN_DIR"
    fi

    # ----- Relax HydraGNN's overly-tight pins ------------------------------
    # HydraGNN's requirements-base.txt pins click==8.0.0 and tqdm==4.67.1,
    # which conflict with Typer (needs click>=8.2.1) and pymatgen-core
    # (needs tqdm>=4.67.3). HydraGNN imports and runs cleanly with the
    # newer versions, so we relax these pins once and keep everything happy.
    REQ_BASE="${HYDRAGNN_DIR}/requirements-base.txt"
    if [[ "$(uname)" == "Darwin" ]]; then SED_INPLACE=(sed -i ''); else SED_INPLACE=(sed -i); fi
    if [[ -f "$REQ_BASE" ]]; then
        if grep -qE '^click==8\.0\.0$' "$REQ_BASE"; then
            log "Relaxing HydraGNN pin: click==8.0.0 -> click>=8.2.1,<9"
            "${SED_INPLACE[@]}" 's/^click==8\.0\.0$/click>=8.2.1,<9/' "$REQ_BASE"
        fi
        if grep -qE '^tqdm==4\.67\.1$' "$REQ_BASE"; then
            log "Relaxing HydraGNN pin: tqdm==4.67.1 -> tqdm>=4.67.3,<5"
            "${SED_INPLACE[@]}" 's/^tqdm==4\.67\.1$/tqdm>=4.67.3,<5/' "$REQ_BASE"
        fi
    fi
    SETUP_PY="${HYDRAGNN_DIR}/setup.py"
    if [[ -f "$SETUP_PY" ]]; then
        if grep -q "click==8.0.0" "$SETUP_PY"; then
            log "Relaxing HydraGNN click pin in setup.py"
            "${SED_INPLACE[@]}" 's/click==8\.0\.0/click>=8.2.1,<9/g' "$SETUP_PY"
        fi
        if grep -q "tqdm==4.67.1" "$SETUP_PY"; then
            log "Relaxing HydraGNN tqdm pin in setup.py"
            "${SED_INPLACE[@]}" 's/tqdm==4\.67\.1/tqdm>=4.67.3,<5/g' "$SETUP_PY"
        fi
    fi
fi

# ------------------------------- Install path ------------------------------
case "$PLATFORM" in
    workstation)
        # ---- standard local install via HydraGNN's install_dependencies.sh ----
        if [[ -d "$VENV_DIR" ]]; then
            log "Reusing virtual environment at ${VENV_DIR}"
        else
            log "Creating virtual environment at ${VENV_DIR}"
            "$PYTHON" -m venv "$VENV_DIR"
        fi
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
        python -m pip install --upgrade pip setuptools wheel

        if [[ "$SKIP_HYDRAGNN" != "1" ]]; then
            INSTALLER="${HYDRAGNN_DIR}/install_dependencies.sh"
            [[ -x "$INSTALLER" ]] || chmod +x "$INSTALLER" 2>/dev/null || true
            [[ -f "$INSTALLER" ]] || die "Cannot find ${INSTALLER}"
            log "Running HydraGNN installer: install_dependencies.sh ${HYDRAGNN_EXTRAS}"
            ( cd "$HYDRAGNN_DIR" && bash ./install_dependencies.sh ${HYDRAGNN_EXTRAS} )
            log "Installing HydraGNN (editable) into ${VENV_DIR}"
            python -m pip install -e "$HYDRAGNN_DIR"
        fi
        ;;

    frontier-rocm71|frontier-rocm64|perlmutter|aurora|andes)
        # ---- DOE supercomputer install ----
        SC_SCRIPT="${HYDRAGNN_DIR}/installation_DOE_supercomputers/hydragnn_installation_bash_script_${PLATFORM}.sh"
        [[ -f "$SC_SCRIPT" ]] || die "Supercomputer installer not found: ${SC_SCRIPT}"
        warn "Supercomputer installs manage their own modules and Python environment."
        warn "Delegating to: ${SC_SCRIPT}"
        warn "After it finishes, this script will install matsim-agents on top."
        ( cd "${HYDRAGNN_DIR}/installation_DOE_supercomputers" \
            && bash "hydragnn_installation_bash_script_${PLATFORM}.sh" )

        # On HPC the supercomputer script sets up a conda/venv environment.
        # The user must activate it before re-running this script with
        # SKIP_HYDRAGNN=1 PLATFORM=workstation, OR pass VENV_DIR pointing at
        # the activated env's prefix. We detect activation here:
        if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
            die "Supercomputer install finished. Activate the environment it created, then rerun:
    SKIP_HYDRAGNN=1 PLATFORM=workstation ./scripts/setup_env.sh \$VIRTUAL_ENV"
        fi
        log "Active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
        ;;

    *) die "Unknown PLATFORM='${PLATFORM}'. Valid: workstation | frontier-rocm71 | frontier-rocm64 | perlmutter | aurora | andes." ;;
esac

# --------------------------- matsim-agents itself --------------------------
EXTRAS=""
for backend in $LLM_BACKENDS; do
    case "$backend" in
        ollama|vllm|openai|anthropic) EXTRAS+="${EXTRAS:+,}${backend}" ;;
        *) warn "Ignoring unknown LLM backend extra: ${backend}" ;;
    esac
done
EXTRAS+="${EXTRAS:+,}dev"

log "Installing matsim-agents (editable) with extras: [${EXTRAS}]"
python -m pip install -e "${REPO_ROOT}[${EXTRAS}]"

# --------------------------- Optional: Ollama daemon -----------------------
bootstrap_ollama() {
    [[ "$PLATFORM" != "workstation" ]] && {
        warn "BOOTSTRAP_OLLAMA is only honored on PLATFORM=workstation; skipping."
        return 0
    }

    # 1) Install the binary if missing.
    if ! command -v ollama >/dev/null 2>&1; then
        case "$(uname)" in
            Darwin)
                if command -v brew >/dev/null 2>&1; then
                    log "Installing Ollama via Homebrew"
                    brew install ollama
                else
                    die "Homebrew not found. Install Ollama manually from https://ollama.com and re-run."
                fi
                ;;
            Linux)
                log "Installing Ollama via official install script"
                curl -fsSL https://ollama.com/install.sh | sh
                ;;
            *)
                die "Cannot auto-install Ollama on $(uname). Install manually from https://ollama.com."
                ;;
        esac
    else
        log "Ollama already installed: $(ollama --version 2>/dev/null | head -1)"
    fi

    # 2) Start the daemon if not already responding on :11434.
    if ! curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        case "$(uname)" in
            Darwin)
                if command -v brew >/dev/null 2>&1; then
                    log "Starting Ollama via 'brew services start ollama'"
                    brew services start ollama || true
                else
                    log "Starting 'ollama serve' in the background"
                    nohup ollama serve >/tmp/ollama.log 2>&1 &
                fi
                ;;
            Linux)
                if command -v systemctl >/dev/null 2>&1 && systemctl --user list-unit-files 2>/dev/null | grep -q '^ollama'; then
                    log "Starting Ollama via systemctl --user"
                    systemctl --user start ollama || true
                else
                    log "Starting 'ollama serve' in the background"
                    nohup ollama serve >/tmp/ollama.log 2>&1 &
                fi
                ;;
        esac
        # Wait up to 30 s for the daemon to come up.
        for _ in $(seq 1 30); do
            if curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        if ! curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
            die "Ollama daemon did not come up on http://localhost:11434 (see /tmp/ollama.log)."
        fi
    else
        log "Ollama daemon already responding on http://localhost:11434"
    fi

    # 3) Pull requested models.
    for model in $OLLAMA_MODELS; do
        log "Pulling Ollama model: ${model}"
        ollama pull "$model"
    done
}

if [[ "$BOOTSTRAP_OLLAMA" == "1" ]]; then
    bootstrap_ollama
fi

# ------------------------------- Smoke test --------------------------------
log "Running smoke import test"
python - <<'PY'
import importlib, sys
required = ("torch", "torch_geometric", "ase", "langgraph", "langchain_core", "matsim_agents")
optional = ("hydragnn",)
ok = True
for mod in required:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:<18} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  FAIL {mod:<18} {exc!r}")
        ok = False
for mod in optional:
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod}")
    except Exception as exc:
        print(f"  warn {mod:<18} {exc!r}")
sys.exit(0 if ok else 1)
PY

cat <<EOF

------------------------------------------------------------
matsim-agents environment is ready.

Platform:           ${PLATFORM}
HydraGNN checkout:  $([[ "$SKIP_HYDRAGNN" == "1" ]] && echo "(skipped)" || echo "${HYDRAGNN_DIR} @ ${HYDRAGNN_REF}")
LLM backends:       ${LLM_BACKENDS}
Active env:         \${VIRTUAL_ENV:-\$CONDA_PREFIX}

Activate (workstation):
    source ${VENV_DIR}/bin/activate

Quick start:
    matsim-agents --help
    matsim-agents run "Relax structures/foo.vasp" \\
        --logdir /path/to/hydragnn_logdir \\
        --mlp-checkpoint /path/to/mlp_branch_weights.pt \\
        --llm-provider ollama
------------------------------------------------------------
EOF
