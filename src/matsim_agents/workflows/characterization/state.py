"""The persisted progress of one contract, and its resume checks."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from matsim_agents.core.hashing import sha256_file
from matsim_agents.execution.locking import LockTimeoutError, exclusive_lock
from matsim_agents.workflows.characterization.contract import CharacterizationContract
from matsim_agents.workflows.characterization.plan import RecipeStep, plan_recipe

STATE_FILE = "characterization-state.json"


LEDGER_FILE = "evidence-ledger.jsonl"


REPORT_FILE = "characterization-report.json"


class _Run:
    """The persisted progress of one contract (``<output_root>/characterization-<id>``)."""

    def __init__(self, contract: CharacterizationContract) -> None:
        self.contract = contract
        self.directory = Path(contract.output_root) / f"characterization-{contract.contract_id}"
        self.directory.mkdir(parents=True, exist_ok=True)
        # One execution owns the run: its state is read, checked, and written under the
        # lock (a lock file, since flock is unsupported on NERSC CFS compute nodes).
        self._stack = contextlib.ExitStack()
        try:
            self._stack.enter_context(exclusive_lock(self.directory / ".lock", timeout=0))
        except LockTimeoutError:
            raise RuntimeError(
                f"{self.directory} is in use by another execution of this contract"
            ) from None
        try:
            self._open(contract)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> _Run:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _open(self, contract: CharacterizationContract) -> None:
        self.state_path = self.directory / STATE_FILE
        digest = contract.digest()
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text())
            if self.state["contract_sha256"] != digest:
                raise ValueError(
                    f"{self.directory} holds a different contract or different input "
                    "files; give the changed contract a new contract_id"
                )
            self.state["contract"] = contract.model_dump(mode="json")
        else:
            self.state = {
                "schema_version": 1,
                "contract_sha256": digest,
                "contract": contract.model_dump(mode="json"),
                "attempts": {"mlip": 0, "dft": 0},
                "steps": {},
            }
        self.state.setdefault("recovery", {})
        self._save()

    def _save(self) -> None:
        staged = self.state_path.with_name(f".{self.state_path.name}.tmp")
        staged.write_text(json.dumps(self.state, indent=2) + "\n")
        staged.replace(self.state_path)

    def attempts(self, fidelity: str) -> int:
        return int(self.state["attempts"][fidelity])

    def charge(self, fidelity: str) -> None:
        """Count a calculation against the budget before it starts."""
        self.state["attempts"][fidelity] += 1
        self._save()

    def record(self, step: RecipeStep, outcome: dict[str, Any]) -> None:
        self.state["steps"][step.step_id] = outcome
        self._save()

    def done(self, step_id: str) -> dict[str, Any] | None:
        outcome = self.state["steps"].get(step_id)
        return outcome if outcome is not None and outcome["status"] == "done" else None

    def pending(self, step_id: str) -> dict[str, Any] | None:
        """The next recovery attempt of a step that the budget stopped, if any."""
        return self.state["recovery"].get(step_id)

    def set_pending(self, step_id: str, recovery: dict[str, Any] | None) -> None:
        if recovery is None:
            self.state["recovery"].pop(step_id, None)
        else:
            self.state["recovery"][step_id] = recovery
        self._save()

    def record_attempt(
        self, step: RecipeStep, outcome: dict[str, Any], recovery: dict[str, Any] | None
    ) -> None:
        """Save an attempt's outcome and the step's next retry (None once terminal) together."""
        self.state["steps"][step.step_id] = outcome
        self.set_pending(step.step_id, recovery)

    def forget(self, step_id: str) -> None:
        self.state["steps"].pop(step_id, None)
        self._save()


def outstanding_steps(contract: CharacterizationContract) -> list[RecipeStep]:
    """Steps of the contract's recipe that have no completed outcome yet."""
    state_path = (
        Path(contract.output_root) / f"characterization-{contract.contract_id}" / STATE_FILE
    )
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    steps, recovery = state.get("steps", {}), state.get("recovery", {})
    return [
        step
        for step in plan_recipe(contract)
        if steps.get(step.step_id, {}).get("status") != "done" or step.step_id in recovery
    ]


def _verify_completed_artifacts(run: _Run, steps: list[RecipeStep]) -> None:
    """Refuse to resume from relaxation outputs, completed or saved for a retry, that changed."""
    changed = []
    for step in steps:
        pending = run.pending(step.step_id) if step.operator == "relax" else None
        if pending is not None and pending["structure"] != step.structure:
            path = Path(pending["structure"])
            recorded = pending.get("structure_sha256")
            if not path.is_file():
                changed.append(f"{step.step_id}: retry structure {path} no longer exists")
            elif recorded is None or sha256_file(path) != recorded:
                changed.append(f"{step.step_id}: retry structure {path} has changed")
        outcome = run.done(step.step_id) if step.operator == "relax" else None
        if outcome is None:
            continue
        path = Path(outcome["optimized_structure_path"])
        recorded = outcome.get("optimized_structure_sha256")
        if not path.is_file():
            changed.append(f"{step.step_id}: {path} no longer exists")
        elif recorded is None or sha256_file(path) != recorded:
            changed.append(f"{step.step_id}: {path} differs from the relaxation's output")
    if changed:
        raise ValueError(
            f"{run.directory} cannot be resumed; completed relaxation outputs changed: "
            + "; ".join(changed)
            + ". Give the contract a new contract_id to recompute them."
        )
