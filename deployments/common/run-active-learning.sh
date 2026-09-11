#!/bin/bash
# Shared production active-learning runner. Source after activating the site venv.
set -euo pipefail

: "${REPO:?set REPO}"
: "${RUN_DIR:?set RUN_DIR}"
: "${MATSIM_SEED_STRUCTURES:?colon-separated ASE-readable seed paths required}"
: "${LOGDIR:?set LOGDIR}"
: "${HYDRAGNN_BRANCH_MLP_CHECKPOINT:?set HYDRAGNN_BRANCH_MLP_CHECKPOINT}"
: "${MATSIM_DFT_BACKEND:=qe}"

AL_CONFIG="${RUN_DIR}/active-learning-config.json"
export AL_CONFIG RUN_DIR LOGDIR HYDRAGNN_BRANCH_MLP_CHECKPOINT
python3 -c '
import json, os
from pathlib import Path

backend = os.environ["MATSIM_DFT_BACKEND"]
if backend == "qe":
    required = ("MATSIM_PW_BIN", "MATSIM_PSEUDO_DIR", "MATSIM_DFT_WRAPPER")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"QE active learning requires environment variables: {missing}")
    dft = {"backend": "qe", "qe": {
        "pw_bin": os.environ["MATSIM_PW_BIN"],
        "pw_wrapper": os.environ["MATSIM_DFT_WRAPPER"],
        "pseudo_dir": os.environ["MATSIM_PSEUDO_DIR"],
        "nodes_per_job": int(os.environ.get("MATSIM_DFT_NODES_PER_JOB", "1")),
        "ranks_per_node": int(os.environ["MATSIM_DFT_RANKS_PER_NODE"]),
        "threads_per_rank": int(os.environ.get("MATSIM_DFT_THREADS_PER_RANK", "1")),
    }}
elif backend == "vasp":
    required = ("MATSIM_VASP_BIN", "MATSIM_POTCAR_DIR", "MATSIM_DFT_WRAPPER", "MATSIM_INCAR_TEMPLATE")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"VASP active learning requires environment variables: {missing}")
    dft = {"backend": "vasp", "vasp": {
        "vasp_bin": os.environ["MATSIM_VASP_BIN"],
        "vasp_wrapper": os.environ["MATSIM_DFT_WRAPPER"],
        "incar_template": os.environ["MATSIM_INCAR_TEMPLATE"],
        "potcar_dir": os.environ["MATSIM_POTCAR_DIR"],
        "nodes_per_job": int(os.environ.get("MATSIM_DFT_NODES_PER_JOB", "1")),
        "ranks_per_node": int(os.environ["MATSIM_DFT_RANKS_PER_NODE"]),
        "threads_per_rank": int(os.environ.get("MATSIM_DFT_THREADS_PER_RANK", "1")),
    }}
else:
    raise SystemExit("MATSIM_DFT_BACKEND must be qe or vasp")

retrain = os.environ.get("MATSIM_RETRAIN", "0") == "1"
trainer = {"enabled": False, "promote_model": False, "promotion_approved": False}
if retrain:
    train_script = os.environ.get("MATSIM_TRAIN_SCRIPT")
    if not train_script:
        raise SystemExit("MATSIM_RETRAIN=1 requires MATSIM_TRAIN_SCRIPT")
    trainer.update({"enabled": True, "train_script": train_script,
                    "train_launcher": os.environ.get("MATSIM_TRAIN_LAUNCHER")})

cfg = {
    "mlip": {"backend": "hydragnn", "hydragnn": {
        "logdir": os.environ["LOGDIR"],
        "mlp_checkpoint": os.environ["HYDRAGNN_BRANCH_MLP_CHECKPOINT"],
        "mlp_device": os.environ.get("MATSIM_MLP_DEVICE", "cuda"),
    }},
    "md": {"seed_source": {"kind": "paths", "paths": os.environ["MATSIM_SEED_STRUCTURES"].split(":")},
           "n_steps": int(os.environ.get("MATSIM_MD_STEPS", "200")), "sample_every": 10,
           "temperature_K": float(os.environ.get("MATSIM_MD_TEMPERATURE_K", "600")),
           "random_seed": int(os.environ.get("MATSIM_RANDOM_SEED", "0"))},
    "acquisition": {"strategy": os.environ.get("MATSIM_ACQUISITION", "mc_dropout"),
                    "n_select": int(os.environ.get("MATSIM_N_SELECT", "8")),
                    "mc_dropout_passes": int(os.environ.get("MATSIM_MC_DROPOUT_PASSES", "8")),
                    "diversity_filter": True},
    "dft": dft,
    "trainer": trainer,
    "loop": {"n_iterations": int(os.environ.get("MATSIM_AL_ITERATIONS", "1")),
             "out_dir": str(Path(os.environ["RUN_DIR"]) / "workflow"),
             "dataset_format": "extxyz", "resume": True, "fail_fast": False},
}
Path(os.environ["AL_CONFIG"]).write_text(json.dumps(cfg, indent=2) + "\n")
'

matsim-agents al validate-config "${AL_CONFIG}" >"${RUN_DIR}/active-learning-config.resolved.json"
matsim-agents al run "${AL_CONFIG}" --log-level "${LOG_LEVEL:-INFO}" \
  2>&1 | tee "${RUN_DIR}/active-learning.log"
