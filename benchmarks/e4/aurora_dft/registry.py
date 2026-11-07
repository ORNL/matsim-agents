"""SQLite registry with atomic reservations, duplicate guards, and restart history."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

ALLOCATIONS = {
    "bcc_refractory": "HydraGNN",
    "fcc_hea": "GenesisHackathonOct26",
    "lifepo4": "GridAI",
}
ACTIVE_STATES = {"RESERVED", "SUBMITTED", "RUNNING", "RESTART_PENDING"}
TERMINAL_ATTEMPT_STATES = {"COMPLETED", "FAILED", "INTERRUPTED", "QUARANTINED"}


class RegistryConflict(ValueError):
    pass


class CalculationRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS calculations (
                    calculation_id TEXT PRIMARY KEY,
                    material_family TEXT NOT NULL,
                    allocation TEXT NOT NULL,
                    structure_hash TEXT NOT NULL,
                    method_signature TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reservation_run_id TEXT,
                    current_attempt INTEGER NOT NULL DEFAULT 0,
                    total_node_hours REAL NOT NULL DEFAULT 0,
                    UNIQUE (structure_hash, method_signature)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_calculation
                    ON calculations(calculation_id)
                    WHERE state IN ('RESERVED', 'SUBMITTED', 'RUNNING', 'RESTART_PENDING');
                CREATE TABLE IF NOT EXISTS attempts (
                    calculation_id TEXT NOT NULL REFERENCES calculations(calculation_id),
                    attempt_number INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    pbs_job_id TEXT,
                    state TEXT NOT NULL,
                    node_hours REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (calculation_id, attempt_number)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def register(
        self,
        calculation_id: str,
        material_family: str,
        allocation: str,
        structure_hash: str,
        method_signature: str,
    ) -> None:
        if material_family not in ALLOCATIONS or allocation != ALLOCATIONS[material_family]:
            raise RegistryConflict("allocation does not match the material family")
        if not all((calculation_id, structure_hash, method_signature)):
            raise ValueError("calculation ID, structure hash, and method signature are required")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO calculations
                       (calculation_id, material_family, allocation, structure_hash,
                        method_signature, state)
                       VALUES (?, ?, ?, ?, ?, 'PLANNED')""",
                    (calculation_id, material_family, allocation, structure_hash, method_signature),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise RegistryConflict(
                "calculation ID or structure/method pair is already registered"
            ) from exc

    def reserve(self, calculation_id: str, run_id: str) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, current_attempt FROM calculations WHERE calculation_id=?",
                (calculation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RegistryConflict("calculation is not registered")
            if row["state"] not in {"PLANNED", "RESTART_PENDING"}:
                connection.rollback()
                raise RegistryConflict(f"calculation cannot be reserved from state {row['state']}")
            attempt = int(row["current_attempt"]) + 1
            connection.execute(
                """UPDATE calculations SET state='RESERVED', reservation_run_id=?, current_attempt=?
                   WHERE calculation_id=?""",
                (run_id, attempt, calculation_id),
            )
            connection.execute(
                """INSERT INTO attempts (calculation_id, attempt_number, run_id, state)
                   VALUES (?, ?, ?, 'RESERVED')""",
                (calculation_id, attempt, run_id),
            )
            connection.commit()
            return attempt

    def transition(self, calculation_id: str, state: str, *, pbs_job_id: str | None = None) -> None:
        allowed = {
            "SUBMITTED",
            "RUNNING",
            "COMPLETED",
            "VERIFIED",
            "FAILED",
            "INTERRUPTED",
            "QUARANTINED",
        }
        if state not in allowed:
            raise ValueError(f"unsupported state transition target: {state}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, current_attempt FROM calculations WHERE calculation_id=?",
                (calculation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RegistryConflict("calculation is not registered")
            current = row["state"]
            valid = {
                "RESERVED": {
                    "SUBMITTED",
                    "RUNNING",
                    "COMPLETED",
                    "FAILED",
                    "INTERRUPTED",
                    "QUARANTINED",
                },
                "SUBMITTED": {"RUNNING", "FAILED", "INTERRUPTED"},
                "RUNNING": {"COMPLETED", "FAILED", "INTERRUPTED", "QUARANTINED"},
                "COMPLETED": {"VERIFIED", "QUARANTINED"},
            }
            if state not in valid.get(current, set()):
                connection.rollback()
                raise RegistryConflict(f"invalid transition {current} -> {state}")
            terminal = state in TERMINAL_ATTEMPT_STATES or state == "VERIFIED"
            connection.execute(
                "UPDATE calculations SET state=?, reservation_run_id=? WHERE calculation_id=?",
                (
                    state,
                    None
                    if terminal
                    else self._run_id(connection, calculation_id, row["current_attempt"]),
                    calculation_id,
                ),
            )
            connection.execute(
                """UPDATE attempts SET state=?, pbs_job_id=COALESCE(?, pbs_job_id)
                   WHERE calculation_id=? AND attempt_number=?""",
                (state, pbs_job_id, calculation_id, row["current_attempt"]),
            )
            connection.commit()

    @staticmethod
    def _run_id(connection: sqlite3.Connection, calculation_id: str, attempt: int) -> str:
        row = connection.execute(
            "SELECT run_id FROM attempts WHERE calculation_id=? AND attempt_number=?",
            (calculation_id, attempt),
        ).fetchone()
        if row is None:
            raise RegistryConflict("active attempt record is missing")
        return str(row["run_id"])

    def finish_attempt(
        self,
        calculation_id: str,
        state: str,
        *,
        allocated_node_hours: float,
        pbs_job_id: str | None = None,
    ) -> None:
        if state not in TERMINAL_ATTEMPT_STATES:
            raise ValueError("attempt finish state must be terminal")
        if not math.isfinite(allocated_node_hours) or allocated_node_hours < 0:
            raise ValueError("allocated node-hours must be finite and nonnegative")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, current_attempt FROM calculations WHERE calculation_id=?",
                (calculation_id,),
            ).fetchone()
            if row is None or row["state"] not in {"RESERVED", "SUBMITTED", "RUNNING"}:
                connection.rollback()
                raise RegistryConflict("calculation has no active attempt to finish")
            connection.execute(
                """UPDATE attempts SET state=?, node_hours=node_hours+?,
                   pbs_job_id=COALESCE(?, pbs_job_id)
                   WHERE calculation_id=? AND attempt_number=?""",
                (state, allocated_node_hours, pbs_job_id, calculation_id, row["current_attempt"]),
            )
            connection.execute(
                """UPDATE calculations SET state=?, reservation_run_id=NULL,
                   total_node_hours=total_node_hours+? WHERE calculation_id=?""",
                (state, allocated_node_hours, calculation_id),
            )
            connection.commit()

    def request_restart(self, calculation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM calculations WHERE calculation_id=?", (calculation_id,)
            ).fetchone()
            if row is None or row["state"] not in {"FAILED", "INTERRUPTED"}:
                connection.rollback()
                raise RegistryConflict("only failed or interrupted calculations can be restarted")
            connection.execute(
                "UPDATE calculations SET state='RESTART_PENDING' WHERE calculation_id=?",
                (calculation_id,),
            )
            connection.commit()

    def get(self, calculation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calculations WHERE calculation_id=?", (calculation_id,)
            ).fetchone()
            if row is None:
                return None
            value = dict(row)
            value["attempts"] = [
                dict(attempt)
                for attempt in connection.execute(
                    "SELECT * FROM attempts WHERE calculation_id=? ORDER BY attempt_number",
                    (calculation_id,),
                )
            ]
            return value


def qualify_relaxation(
    *,
    ionic_terminated: bool,
    scf_converged_each_step: list[bool],
    max_force_eV_A: float | None,
    force_threshold_eV_A: float,
) -> dict[str, Any]:
    electronic = bool(scf_converged_each_step) and all(scf_converged_each_step)
    forces_ok = max_force_eV_A is not None and max_force_eV_A <= force_threshold_eV_A
    return {
        "ionic_status": "terminated" if ionic_terminated else "not_terminated",
        "electronic_status": "converged_each_step"
        if electronic
        else "scf_failure_or_missing_steps",
        "force_status": "within_threshold" if forces_ok else "above_threshold_or_missing",
        "qualification_status": "qualified"
        if ionic_terminated and electronic and forces_ok
        else "unqualified",
    }


def qualify_static(*, normal_termination: bool, scf_converged: bool) -> dict[str, str]:
    converged = normal_termination and scf_converged
    return {
        "scf_status": "converged" if converged else "unqualified",
        "relaxation_status": "not_applicable",
    }
