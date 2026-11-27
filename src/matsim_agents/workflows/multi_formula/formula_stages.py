"""Per-formula campaign stage status, persisted as campaign_stages.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.workflows.multi_formula.formula_config import CampaignFormulaExecutionConfig


class _StageTracker:
    """Persist per-formula campaign stage status to ``campaign_stages.json``."""

    STAGE_NAMES = (
        "initial_mlip_exploration",
        "active_learning_labels_and_training",
        "post_training_mlip_exploration",
        "independent_dft_ranking",
    )

    def __init__(self, path: Path, formula: str, config: CampaignFormulaExecutionConfig):
        self.path = path
        self.formula = formula
        self.stages: dict[str, dict[str, Any]] = {
            name: {"name": name, "status": "pending"} for name in self.STAGE_NAMES
        }
        if not config.phase_policy.active_learning:
            self.stages["active_learning_labels_and_training"]["status"] = "skipped"
            self.stages["active_learning_labels_and_training"]["reason"] = (
                "active learning is disabled by the phase policy"
            )
        if not config.phase_policy.reevaluate_after_retraining:
            self.stages["post_training_mlip_exploration"]["status"] = "skipped"
            self.stages["post_training_mlip_exploration"]["reason"] = (
                "post-training reevaluation is disabled"
            )
        if config.dft_refinement is None:
            self.stages["independent_dft_ranking"]["status"] = "skipped"
            self.stages["independent_dft_ranking"]["reason"] = "DFT refinement is not configured"
        self.write()

    def status(self, name: str) -> str:
        return self.stages[name]["status"]

    def write(self) -> None:
        payload = {
            "schema_version": 1,
            "formula": self.formula,
            "stages": list(self.stages.values()),
        }
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def set(self, name: str, status: str, **evidence: Any) -> None:
        record = self.stages.setdefault(name, {"name": name})
        now = datetime.now(UTC).isoformat()
        record["status"] = status
        if status == "in_progress":
            record["started_at_utc"] = now
        if status in {"completed", "failed", "skipped", "blocked"}:
            record["completed_at_utc"] = now
        record.update(evidence)
        self.write()
