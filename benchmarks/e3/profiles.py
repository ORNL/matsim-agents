"""Explicit acquisition ceilings, independent of physical GPU worker counts."""

import os


def structures_per_round() -> int:
    value = os.environ.get("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "64")
    if value not in {"64", "5000"}:
        raise ValueError("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND must be 64 or 5000")
    return int(value)


def campaign_id() -> str:
    return "e3-mlip-collaborator-5000" if structures_per_round() == 5000 else "e3-mlip-production"
