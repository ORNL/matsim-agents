"""Reusable Typer option factories shared by several commands."""

from __future__ import annotations

from pathlib import Path

import typer


def opt_mlip_backend():
    return typer.Option(
        "hydragnn",
        "--mlip-backend",
        help="Relaxation backend: hydragnn | uma.",
        case_sensitive=False,
    )


def opt_logdir():
    return typer.Option(
        None,
        help=(
            "HydraGNN logdir with config.json and checkpoint "
            "(required for --mlip-backend hydragnn)."
        ),
    )


def opt_hydragnn_branch_mlp_checkpoint():
    return typer.Option(
        None,
        help="Path to BranchWeightMLP checkpoint (.pt), required for --mlip-backend hydragnn.",
    )


def opt_hydragnn_inference_head():
    return typer.Option(
        None,
        "--hydragnn-inference-head",
        help=(
            "Use only one HydraGNN decoding head by dataset name "
            "(for example OMat24) or index 0..15; bypasses BranchWeightMLP."
        ),
    )


def opt_uma_model_name():
    return typer.Option(
        "uma-s-1p1",
        "--uma-model-name",
        help="UMA model name/checkpoint (used when --mlip-backend uma).",
    )


def opt_uma_task():
    return typer.Option(
        "omat",
        "--uma-task",
        help="UMA task head: omat | omol (used when --mlip-backend uma).",
    )


def opt_checkpoint(help_text: str):
    return typer.Option(None, help=help_text)


def opt_mlp_device():
    return typer.Option("cuda", help="Device for the auxiliary MLP (cuda|cpu).")


def opt_precision(help_text: str):
    return typer.Option(None, help=help_text)


def opt_mlp_precision(help_text: str):
    return typer.Option(None, help=help_text)


def opt_trigger_al_handoff(default: bool, help_text: str):
    return typer.Option(
        default,
        "--trigger-al-handoff/--no-trigger-al-handoff",
        help=help_text,
    )


def opt_active_learning_config(help_text: str):
    return typer.Option(None, "--al-config", help=help_text)


def opt_active_learning_dry_run(help_text: str):
    return typer.Option(True, "--al-dry-run/--al-run", help=help_text)


def opt_uq_top_weight_threshold():
    return typer.Option(
        0.6,
        "--uq-top-weight-threshold",
        help="Trigger handoff when mean top branch weight is below this value.",
    )


def opt_uq_min_unreliable_fraction():
    return typer.Option(
        0.25,
        "--uq-min-unreliable-fraction",
        help="Trigger handoff when this fraction of relaxations is low-confidence.",
    )


def opt_uq_min_relaxations_for_handoff(default: int, help_text: str):
    return typer.Option(
        default,
        "--uq-min-relaxations-for-handoff",
        help=help_text,
    )


def opt_al_handoff_audit_path(help_text: str):
    return typer.Option(None, "--al-handoff-audit-path", help=help_text)


def path_or_none(value: Path | None) -> str | None:
    return str(value) if value else None
