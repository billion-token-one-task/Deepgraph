"""Canonical ResourceGrant GPU admission and per-attempt settlement.

The agenda ledger reserves a grant's full GPU ceiling once.  This control
plane atomically subdivides that ceiling among attempts and moves each
attempt's real wall-clock use from agenda ``reserved`` to ``spent`` exactly
once.  Callers must not reconstruct grant usage from iteration rows, timer
state, or backend-specific logs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from db import database as db


GPU_BACKENDS = {"local_gpu", "ssh_gpu", "colab_gpu"}
ACTIVE_ATTEMPT_STATES = {"reserved", "running"}
TERMINAL_ATTEMPT_STATES = {"settled", "released"}
_EPSILON_SECONDS = 1e-6
# The persisted reservation includes a small controller/transport cleanup
# margin.  The executable timeout excludes it, so a normally interrupted
# attempt can still settle its full start-to-completion wall clock without
# crossing the grant ceiling.
_SETTLEMENT_GUARD_WALL_SECONDS = 5


class AttemptGPUUsageError(RuntimeError):
    """Fail-closed admission or lifecycle error with a stable reason code."""

    def __init__(self, reason_code: str, detail: str | None = None):
        self.reason_code = str(reason_code)
        super().__init__(
            self.reason_code
            if not detail
            else f"{self.reason_code}:{str(detail).strip()}"
        )


@dataclass(frozen=True)
class AttemptGPUReservation:
    reservation_id: int
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    attempt_key: str
    backend_kind: str
    gpu_count: int
    reserved_gpu_seconds: float
    timeout_seconds: int
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    actual_gpu_seconds: float | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class GrantGPUUsage:
    resource_grant_id: int
    cap_gpu_seconds: float
    settled_gpu_seconds: float
    active_reserved_gpu_seconds: float
    active_reservations: int
    grant_status: str

    @property
    def remaining_gpu_seconds(self) -> float:
        return max(
            0.0,
            self.cap_gpu_seconds
            - self.settled_gpu_seconds
            - self.active_reserved_gpu_seconds,
        )

    @property
    def exhausted(self) -> bool:
        return (
            (
                self.settled_gpu_seconds + _EPSILON_SECONDS
                >= self.cap_gpu_seconds
                or self.remaining_gpu_seconds + _EPSILON_SECONDS < 1.0
            )
            and self.active_reservations == 0
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _snapshot(row: dict[str, Any]) -> AttemptGPUReservation:
    return AttemptGPUReservation(
        reservation_id=int(row["id"]),
        agenda_id=int(row["agenda_id"]),
        idea_id=int(row["idea_id"]),
        resource_grant_id=int(row["resource_grant_id"]),
        attempt_key=str(row["attempt_key"]),
        backend_kind=str(row["backend_kind"]),
        gpu_count=int(row.get("gpu_count") or 1),
        reserved_gpu_seconds=float(row.get("reserved_gpu_seconds") or 0.0),
        timeout_seconds=int(row.get("timeout_seconds") or 0),
        status=str(row["status"]),
        started_at=_as_utc(row.get("started_at")),
        completed_at=_as_utc(row.get("completed_at")),
        actual_gpu_seconds=(
            float(row["actual_gpu_seconds"])
            if row.get("actual_gpu_seconds") is not None
            else None
        ),
        reason_code=(
            str(row["reason_code"]) if row.get("reason_code") else None
        ),
    )


class GrantGPUUsageControl:
    """The only supported GPU budget interface for grant-scoped attempts."""

    def _grant(self, grant_id: int, *, lock: bool) -> dict[str, Any] | None:
        suffix = " FOR UPDATE" if lock and db._use_pg() else ""  # noqa: SLF001
        return db.fetchone(
            f"SELECT * FROM resource_grants WHERE id=?{suffix}",
            (int(grant_id),),
        )

    def _usage_locked(self, grant: dict[str, Any]) -> GrantGPUUsage:
        usage = db.fetchone(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status='settled'
                                  THEN actual_gpu_seconds ELSE 0 END), 0)
                    AS settled_gpu_seconds,
                COALESCE(SUM(CASE WHEN status IN ('reserved','running')
                                  THEN reserved_gpu_seconds ELSE 0 END), 0)
                    AS active_reserved_gpu_seconds,
                COALESCE(SUM(CASE WHEN status IN ('reserved','running')
                                  THEN 1 ELSE 0 END), 0)
                    AS active_reservations
            FROM experiment_attempt_gpu_reservations_v1
            WHERE resource_grant_id=?
            """,
            (int(grant["id"]),),
        ) or {}
        return GrantGPUUsage(
            resource_grant_id=int(grant["id"]),
            cap_gpu_seconds=float(grant.get("max_gpu_hours") or 0.0) * 3600.0,
            settled_gpu_seconds=float(
                usage.get("settled_gpu_seconds") or 0.0
            ),
            active_reserved_gpu_seconds=float(
                usage.get("active_reserved_gpu_seconds") or 0.0
            ),
            active_reservations=int(usage.get("active_reservations") or 0),
            grant_status=str(grant.get("status") or ""),
        )

    def grant_usage(self, resource_grant_id: int) -> GrantGPUUsage:
        try:
            grant = self._grant(resource_grant_id, lock=False)
            if not grant:
                raise AttemptGPUUsageError("resource_grant_not_found")
            usage = self._usage_locked(grant)
            # Public read boundaries must not leave a metadata transaction
            # open while the caller proceeds to a long GPU/SSH operation.
            db.commit()
            return usage
        except Exception:
            db.rollback()
            raise

    def reservation(self, reservation_id: int) -> AttemptGPUReservation:
        try:
            row = db.fetchone(
                "SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                (int(reservation_id),),
            )
            if not row:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            snapshot = _snapshot(row)
            db.commit()
            return snapshot
        except Exception:
            db.rollback()
            raise

    def remaining_attempt_wall_seconds(self, reservation_id: int) -> float:
        reservation = self.reservation(reservation_id)
        if reservation.status == "reserved":
            return float(reservation.timeout_seconds)
        if reservation.status != "running" or reservation.started_at is None:
            return 0.0
        elapsed = max(0.0, (_now() - reservation.started_at).total_seconds())
        return max(0.0, float(reservation.timeout_seconds) - elapsed)

    def reserve_attempt(
        self,
        *,
        agenda_id: int,
        idea_id: int,
        resource_grant_id: int,
        attempt_key: str,
        backend_kind: str,
        requested_timeout_seconds: int,
        gpu_count: int = 1,
        experiment_run_id: int | None = None,
        commit: bool = True,
    ) -> AttemptGPUReservation:
        """Atomically reserve the remaining grant GPU seconds.

        The returned timeout is always clamped to the grant remainder after
        settled usage and every other active reservation are subtracted.
        """
        attempt_key = str(attempt_key or "").strip()
        backend_kind = str(backend_kind or "").strip()
        requested_timeout_seconds = int(requested_timeout_seconds)
        gpu_count = int(gpu_count)
        if min(int(agenda_id), int(idea_id), int(resource_grant_id)) <= 0:
            raise AttemptGPUUsageError("attempt_scope_invalid")
        if not attempt_key:
            raise AttemptGPUUsageError("attempt_key_required")
        if backend_kind not in GPU_BACKENDS:
            raise AttemptGPUUsageError("attempt_backend_not_gpu")
        if requested_timeout_seconds <= 0 or gpu_count <= 0:
            raise AttemptGPUUsageError("attempt_gpu_request_invalid")
        try:
            grant = self._grant(resource_grant_id, lock=True)
            expires_at = _as_utc((grant or {}).get("expires_at"))
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != int(agenda_id)
                or int(grant.get("idea_id") or 0) != int(idea_id)
            ):
                raise AttemptGPUUsageError("resource_grant_scope_mismatch")
            if str(grant.get("status") or "") != "active":
                raise AttemptGPUUsageError("resource_grant_not_active")
            if expires_at is None or expires_at <= _now():
                raise AttemptGPUUsageError("resource_grant_expired")
            try:
                allowlist = set(json.loads(grant.get("backend_allowlist_json") or "[]"))
            except (TypeError, ValueError):
                allowlist = set()
            if backend_kind not in allowlist:
                raise AttemptGPUUsageError("attempt_backend_not_granted")
            existing = db.fetchone(
                """
                SELECT * FROM experiment_attempt_gpu_reservations_v1
                WHERE resource_grant_id=? AND attempt_key=?
                """,
                (int(resource_grant_id), attempt_key),
            )
            if existing:
                expected = {
                    "agenda_id": int(agenda_id),
                    "idea_id": int(idea_id),
                    "backend_kind": backend_kind,
                    "gpu_count": gpu_count,
                }
                mismatches = [
                    name
                    for name, expected_value in expected.items()
                    if str(existing.get(name)) != str(expected_value)
                ]
                if mismatches:
                    raise AttemptGPUUsageError(
                        "attempt_key_scope_mismatch", ",".join(mismatches)
                    )
                if str(existing.get("status")) in TERMINAL_ATTEMPT_STATES:
                    raise AttemptGPUUsageError(
                        "attempt_already_terminal", str(existing.get("status"))
                    )
                if commit:
                    db.commit()
                return _snapshot(existing)
            usage = self._usage_locked(grant)
            remaining = usage.remaining_gpu_seconds
            minimum_gpu_seconds = float(gpu_count)
            if remaining + _EPSILON_SECONDS < minimum_gpu_seconds:
                if usage.exhausted and commit:
                    self._finalize_grant_locked(
                        grant,
                        usage=usage,
                        reason_code="grant_gpu_hours_exhausted",
                    )
                    db.commit()
                raise AttemptGPUUsageError("grant_gpu_hours_exhausted")
            available_wall_seconds = remaining / gpu_count
            settlement_guard = min(
                _SETTLEMENT_GUARD_WALL_SECONDS,
                max(0, int(available_wall_seconds) - 1),
            )
            timeout_seconds = int(
                min(
                    float(requested_timeout_seconds),
                    available_wall_seconds - settlement_guard,
                )
            )
            if timeout_seconds <= 0:
                raise AttemptGPUUsageError("grant_gpu_hours_exhausted")
            reserved_gpu_seconds = float(
                min(
                    remaining,
                    (timeout_seconds + settlement_guard) * gpu_count,
                )
            )
            lease_expires_at = _now() + timedelta(minutes=5)
            reservation_id = db.insert_returning_id(
                """
                INSERT INTO experiment_attempt_gpu_reservations_v1
                    (agenda_id, idea_id, resource_grant_id, experiment_run_id,
                     attempt_key, backend_kind, gpu_count,
                     reserved_gpu_seconds, timeout_seconds, status,
                     lease_expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?)
                RETURNING id
                """,
                (
                    int(agenda_id),
                    int(idea_id),
                    int(resource_grant_id),
                    int(experiment_run_id) if experiment_run_id else None,
                    attempt_key,
                    backend_kind,
                    gpu_count,
                    reserved_gpu_seconds,
                    timeout_seconds,
                    lease_expires_at.isoformat(),
                ),
            )
            row = db.fetchone(
                "SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                (reservation_id,),
            )
            if not row:
                raise AttemptGPUUsageError("attempt_reservation_persistence_failed")
            if commit:
                db.commit()
            return _snapshot(row)
        except Exception:
            if commit:
                db.rollback()
            raise

    def bind_compute_job(
        self,
        reservation_id: int,
        compute_job_id: int,
        *,
        commit: bool = True,
    ) -> None:
        try:
            cursor = db.execute(
                """
                UPDATE experiment_attempt_gpu_reservations_v1
                SET compute_job_id=?, updated_at=CURRENT_TIMESTAMP,
                    lease_expires_at=(
                        SELECT expires_at FROM resource_grants
                        WHERE id=resource_grant_id
                    )
                WHERE id=? AND status IN ('reserved','running')
                  AND (compute_job_id IS NULL OR compute_job_id=?)
                """,
                (int(compute_job_id), int(reservation_id), int(compute_job_id)),
            )
            if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                raise AttemptGPUUsageError("attempt_compute_bind_failed")
            db.execute(
                """
                UPDATE compute_jobs_v1 SET gpu_attempt_reservation_id=?
                WHERE id=? AND (gpu_attempt_reservation_id IS NULL
                                OR gpu_attempt_reservation_id=?)
                  AND agenda_id=(
                      SELECT agenda_id
                      FROM experiment_attempt_gpu_reservations_v1
                      WHERE id=?
                  )
                """,
                (
                    int(reservation_id),
                    int(compute_job_id),
                    int(reservation_id),
                    int(reservation_id),
                ),
            )
            if commit:
                db.commit()
        except Exception:
            if commit:
                db.rollback()
            raise

    def bind_gpu_job(
        self,
        reservation_id: int,
        gpu_job_id: int,
        *,
        commit: bool = True,
    ) -> None:
        try:
            cursor = db.execute(
                """
                UPDATE experiment_attempt_gpu_reservations_v1
                SET gpu_job_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status IN ('reserved','running')
                  AND (gpu_job_id IS NULL OR gpu_job_id=?)
                """,
                (int(gpu_job_id), int(reservation_id), int(gpu_job_id)),
            )
            if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                raise AttemptGPUUsageError("attempt_gpu_job_bind_failed")
            db.execute(
                """
                UPDATE gpu_jobs SET gpu_attempt_reservation_id=?
                WHERE id=? AND (gpu_attempt_reservation_id IS NULL
                                OR gpu_attempt_reservation_id=?)
                  AND agenda_id=(
                      SELECT agenda_id
                      FROM experiment_attempt_gpu_reservations_v1
                      WHERE id=?
                  )
                """,
                (
                    int(reservation_id),
                    int(gpu_job_id),
                    int(reservation_id),
                    int(reservation_id),
                ),
            )
            if commit:
                db.commit()
        except Exception:
            if commit:
                db.rollback()
            raise

    def start_attempt(
        self,
        reservation_id: int,
        *,
        started_at: datetime | None = None,
        commit: bool = True,
    ) -> AttemptGPUReservation:
        """Persist the attempt boundary before any GPU/SSH work begins."""
        started_at = _as_utc(started_at) or _now()
        try:
            seed = db.fetchone(
                "SELECT resource_grant_id FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                (int(reservation_id),),
            )
            if not seed:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            grant = self._grant(int(seed["resource_grant_id"]), lock=True)
            suffix = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?{suffix}",
                (int(reservation_id),),
            )
            if not row or not grant:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            if str(row.get("status")) == "running":
                if commit:
                    db.commit()
                return _snapshot(row)
            if str(row.get("status")) in TERMINAL_ATTEMPT_STATES:
                raise AttemptGPUUsageError("attempt_already_terminal")
            expires_at = _as_utc(grant.get("expires_at"))
            if str(grant.get("status")) != "active":
                raise AttemptGPUUsageError("resource_grant_not_active")
            if expires_at is None or expires_at <= started_at:
                raise AttemptGPUUsageError("resource_grant_expired")
            lease_expires_at = started_at + timedelta(
                seconds=int(row["timeout_seconds"])
            )
            db.execute(
                """
                UPDATE experiment_attempt_gpu_reservations_v1
                SET status='running', started_at=?, lease_expires_at=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='reserved'
                """,
                (
                    started_at.isoformat(),
                    lease_expires_at.isoformat(),
                    int(reservation_id),
                ),
            )
            if commit:
                db.commit()
            return _snapshot(
                db.fetchone(
                    "SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                    (int(reservation_id),),
                )
            )
        except Exception:
            db.rollback()
            raise

    def reconcile_terminal_attempts(self) -> list[int]:
        """Settle terminal GPU jobs left between backend and ledger commits."""
        rows = db.fetchall(
            """
            SELECT ar.id AS reservation_id, gj.id AS gpu_job_id,
                   gj.status, gj.completed_at
            FROM experiment_attempt_gpu_reservations_v1 ar
            JOIN gpu_jobs gj ON gj.id=ar.gpu_job_id
            WHERE ar.status='running'
              AND gj.status IN ('completed','failed','timed_out','cancelled')
              AND gj.completed_at IS NOT NULL
            ORDER BY ar.id
            """
        )
        db.commit()
        for row in rows:
            terminal_status = str(row.get("status") or "failed")
            reason_code = {
                "completed": "attempt_completed",
                "timed_out": "attempt_timed_out",
                "cancelled": "attempt_cancelled",
            }.get(terminal_status, "attempt_failed")
            self.settle_attempt(
                int(row["reservation_id"]),
                completed_at=row.get("completed_at"),
                reason_code=reason_code,
            )
        pending_durable = db.fetchall(
            """
            SELECT DISTINCT gj.id AS gpu_job_id
            FROM experiment_attempt_gpu_reservations_v1 ar
            JOIN gpu_jobs gj ON gj.id=ar.gpu_job_id
            JOIN compute_jobs_v1 cj ON cj.id=ar.compute_job_id
            WHERE ar.status='settled'
              AND gj.status IN ('completed','failed','timed_out','cancelled')
              AND cj.status IN ('submitting','submitted','running',
                                'cancel_requested','collecting')
            ORDER BY gj.id
            """
        )
        db.commit()
        return [int(row["gpu_job_id"]) for row in pending_durable]

    def reconcile_terminal_colab_attempts(self) -> list[int]:
        """Settle Colab results persisted before their GPU ledger commit."""
        rows = db.fetchall(
            """
            SELECT ar.id AS reservation_id, cwr.id AS request_id,
                   cwr.status, cwr.completed_at
            FROM experiment_attempt_gpu_reservations_v1 ar
            JOIN compute_jobs_v1 cj ON cj.id=ar.compute_job_id
            JOIN colab_work_requests_v1 cwr ON cwr.compute_job_id=cj.id
            WHERE ar.status='running'
              AND cwr.status IN ('succeeded','failed','timed_out','cancelled')
              AND cwr.completed_at IS NOT NULL
            ORDER BY ar.id
            """
        )
        db.commit()
        for row in rows:
            terminal_status = str(row.get("status") or "failed")
            reason_code = {
                "succeeded": "attempt_completed",
                "timed_out": "attempt_timed_out",
                "cancelled": "attempt_cancelled",
            }.get(terminal_status, "attempt_failed")
            self.settle_attempt(
                int(row["reservation_id"]),
                completed_at=row.get("completed_at"),
                reason_code=reason_code,
            )
        pending_durable = db.fetchall(
            """
            SELECT DISTINCT cwr.id AS request_id
            FROM experiment_attempt_gpu_reservations_v1 ar
            JOIN compute_jobs_v1 cj ON cj.id=ar.compute_job_id
            JOIN colab_work_requests_v1 cwr ON cwr.compute_job_id=cj.id
            WHERE ar.status='settled'
              AND cwr.status IN ('succeeded','failed','timed_out','cancelled')
              AND cj.status IN ('submitting','submitted','running',
                                'cancel_requested','collecting')
            ORDER BY cwr.id
            """
        )
        db.commit()
        return [int(row["request_id"]) for row in pending_durable]

    def release_unstarted(
        self,
        reservation_id: int,
        *,
        reason_code: str,
    ) -> bool:
        reason_code = str(reason_code or "attempt_released").strip()
        try:
            cursor = db.execute(
                """
                UPDATE experiment_attempt_gpu_reservations_v1
                SET status='released', reason_code=?, completed_at=CURRENT_TIMESTAMP,
                    actual_gpu_seconds=0, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='reserved' AND started_at IS NULL
                """,
                (reason_code, int(reservation_id)),
            )
            db.commit()
            return int(getattr(cursor, "rowcount", 0) or 0) == 1
        except Exception:
            db.rollback()
            raise

    def settle_attempt(
        self,
        reservation_id: int,
        *,
        completed_at: datetime | None = None,
        reason_code: str,
    ) -> AttemptGPUReservation:
        """Settle real ``started_at -> completed_at`` GPU wall time once."""
        completed_at = _as_utc(completed_at) or _now()
        reason_code = str(reason_code or "attempt_completed").strip()
        try:
            seed = db.fetchone(
                "SELECT resource_grant_id FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                (int(reservation_id),),
            )
            if not seed:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            grant = self._grant(int(seed["resource_grant_id"]), lock=True)
            suffix = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
            row = db.fetchone(
                f"SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?{suffix}",
                (int(reservation_id),),
            )
            if not row or not grant:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            if str(row.get("status")) == "settled":
                db.commit()
                return _snapshot(row)
            if str(row.get("status")) == "released":
                raise AttemptGPUUsageError("attempt_already_terminal", "released")
            started_at = _as_utc(row.get("started_at"))
            if started_at is None:
                raise AttemptGPUUsageError("attempt_not_started")
            if completed_at < started_at:
                raise AttemptGPUUsageError("attempt_completion_before_start")
            wall_seconds = max(0.0, (completed_at - started_at).total_seconds())
            actual_gpu_seconds = wall_seconds * int(row.get("gpu_count") or 1)
            ledger = db.fetchone(
                "SELECT * FROM agenda_resource_ledger WHERE id=?" + suffix,
                (int(grant["reservation_id"]),),
            )
            if not ledger or str(ledger.get("status")) != "reserved":
                raise AttemptGPUUsageError("grant_ledger_not_reservable")
            actual_hours = actual_gpu_seconds / 3600.0
            outstanding_hours = max(
                0.0,
                float(ledger.get("gpu_hours_reserved") or 0.0)
                - float(ledger.get("gpu_hours_used") or 0.0),
            )
            if actual_hours > outstanding_hours + 1e-9:
                # Never discard measured usage.  The timeout/guard is the
                # preventative boundary; if a controller stalls beyond it,
                # preserve the truth and expose a stable overrun reason so the
                # acceptance gate fails closed instead of stranding the grant.
                reason_code = (
                    f"{reason_code}:grant_gpu_hours_overrun"
                )
            reserved_release_hours = min(actual_hours, outstanding_hours)
            overrun_hours = max(0.0, actual_hours - outstanding_hours)
            db.execute(
                """
                UPDATE experiment_attempt_gpu_reservations_v1
                SET status='settled', completed_at=?, actual_gpu_seconds=?,
                    reason_code=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status IN ('reserved','running')
                """,
                (
                    completed_at.isoformat(),
                    actual_gpu_seconds,
                    reason_code,
                    int(reservation_id),
                ),
            )
            db.execute(
                """
                UPDATE agenda_resource_ledger
                SET gpu_hours_used=COALESCE(gpu_hours_used,0)+?,
                    gpu_hours_overrun=COALESCE(gpu_hours_overrun,0)+?
                WHERE id=? AND status='reserved'
                """,
                (
                    reserved_release_hours,
                    overrun_hours,
                    int(grant["reservation_id"]),
                ),
            )
            db.execute(
                """
                UPDATE research_agendas
                SET gpu_hours_reserved=gpu_hours_reserved-?,
                    gpu_hours_spent=gpu_hours_spent+?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    reserved_release_hours,
                    actual_hours,
                    int(grant["agenda_id"]),
                ),
            )
            usage = self._usage_locked(grant)
            if usage.exhausted:
                self._finalize_grant_locked(
                    grant,
                    usage=usage,
                    reason_code="grant_gpu_hours_exhausted",
                )
            db.commit()
            return _snapshot(
                db.fetchone(
                    "SELECT * FROM experiment_attempt_gpu_reservations_v1 WHERE id=?",
                    (int(reservation_id),),
                )
            )
        except Exception:
            db.rollback()
            raise

    def reconcile_exhausted_grant(
        self,
        resource_grant_id: int,
        *,
        reason_code: str = "grant_gpu_hours_exhausted",
    ) -> GrantGPUUsage:
        """Atomically move a fully used grant to its terminal state.

        Admission errors deliberately roll back their transaction.  Timers and
        startup recovery call this method separately so terminalization is not
        accidentally rolled back with a rejected attempt.
        """
        try:
            grant = self._grant(resource_grant_id, lock=True)
            if not grant:
                raise AttemptGPUUsageError("resource_grant_not_found")
            usage = self._usage_locked(grant)
            if usage.exhausted and str(grant.get("status") or "") == "active":
                self._finalize_grant_locked(
                    grant,
                    usage=usage,
                    reason_code=str(reason_code),
                )
            db.commit()
            refreshed = self._grant(resource_grant_id, lock=False)
            if not refreshed:
                raise AttemptGPUUsageError("resource_grant_not_found")
            refreshed_usage = self._usage_locked(refreshed)
            db.commit()
            return refreshed_usage
        except Exception:
            db.rollback()
            raise

    def _finalize_grant_locked(
        self,
        grant: dict[str, Any],
        *,
        usage: GrantGPUUsage,
        reason_code: str,
    ) -> None:
        """Consume an exhausted grant and settle its agenda reservation once."""
        if str(grant.get("status")) == "consumed":
            return
        if usage.active_reservations:
            return
        suffix = " FOR UPDATE" if db._use_pg() else ""  # noqa: SLF001
        ledger = db.fetchone(
            f"SELECT * FROM agenda_resource_ledger WHERE id=?{suffix}",
            (int(grant["reservation_id"]),),
        )
        if not ledger or str(ledger.get("status")) != "reserved":
            return
        token_usage = db.fetchone(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status='settled'
                                  THEN tokens_used ELSE 0 END), 0) AS used,
                COALESCE(SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END), 0)
                    AS open_count
            FROM resource_grant_usage_reservations
            WHERE resource_grant_id=?
            """,
            (int(grant["id"]),),
        ) or {}
        if int(token_usage.get("open_count") or 0):
            db.execute(
                """
                UPDATE resource_grant_usage_reservations
                SET status='released', release_reason=?,
                    settled_at=CURRENT_TIMESTAMP
                WHERE resource_grant_id=? AND status='reserved'
                """,
                (reason_code, int(grant["id"])),
            )
        tokens_used = int(token_usage.get("used") or 0)
        token_reserved = int(ledger.get("token_reserved") or 0)
        gpu_cap_hours = float(ledger.get("gpu_hours_reserved") or 0.0)
        gpu_used_hours = usage.settled_gpu_seconds / 3600.0
        gpu_ledger_used = min(gpu_cap_hours, gpu_used_hours)
        gpu_overrun = max(0.0, gpu_used_hours - gpu_cap_hours)
        gpu_outstanding = max(0.0, gpu_cap_hours - gpu_used_hours)
        db.execute(
            """
            UPDATE research_agendas
            SET token_reserved=token_reserved-?, token_spent=token_spent+?,
                gpu_hours_reserved=gpu_hours_reserved-?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                token_reserved,
                tokens_used,
                gpu_outstanding,
                int(grant["agenda_id"]),
            ),
        )
        db.execute(
            """
            UPDATE agenda_resource_ledger
            SET tokens_used=?, gpu_hours_used=?, gpu_hours_overrun=?,
                status='settled',
                release_reason=?, settled_at=CURRENT_TIMESTAMP
            WHERE id=? AND status='reserved'
            """,
            (
                tokens_used,
                gpu_ledger_used,
                gpu_overrun,
                reason_code,
                int(grant["reservation_id"]),
            ),
        )
        db.execute(
            """
            UPDATE resource_grants SET status='consumed'
            WHERE id=? AND status='active' AND agenda_id=?
            """,
            (int(grant["id"]), int(grant["agenda_id"])),
        )

    def usage_for_compute_job(self, compute_job_id: int) -> dict[str, Any]:
        try:
            row = db.fetchone(
                """
                SELECT * FROM experiment_attempt_gpu_reservations_v1
                WHERE compute_job_id=?
                """,
                (int(compute_job_id),),
            )
            if not row:
                raise AttemptGPUUsageError("attempt_reservation_not_found")
            if str(row.get("status")) not in {"settled", "released"}:
                raise AttemptGPUUsageError("attempt_usage_not_settled")
            actual_gpu_seconds = float(row.get("actual_gpu_seconds") or 0.0)
            gpu_count = max(1, int(row.get("gpu_count") or 1))
            result = {
                "wall_seconds": actual_gpu_seconds / gpu_count,
                "gpu_hours": actual_gpu_seconds / 3600.0,
                "gpu_count": gpu_count,
                "started_at": row.get("started_at"),
                "completed_at": row.get("completed_at"),
                "reason_code": row.get("reason_code"),
                "attempt_reservation_id": int(row["id"]),
            }
            db.commit()
            return result
        except Exception:
            db.rollback()
            raise

    def release_orphaned_reservations(self) -> int:
        """Release unstarted claims that died before durable compute binding."""
        cursor = db.execute(
            """
            UPDATE experiment_attempt_gpu_reservations_v1
            SET status='released', reason_code='controller_lost_before_submit',
                actual_gpu_seconds=0, completed_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE status='reserved' AND started_at IS NULL
              AND compute_job_id IS NULL
              AND lease_expires_at <= CURRENT_TIMESTAMP
            """
        )
        db.commit()
        return int(getattr(cursor, "rowcount", 0) or 0)

    def release_prelaunch_blocked_reservations(self) -> int:
        """Release claims whose GPU job died before it ever started.

        release_orphaned_reservations only covers claims that never reached a
        compute job. A claim that did reach one, whose legacy GPU job was then
        refused at the launch boundary, is stranded instead: settling the
        compute job demands settled attempt usage, and the usage cannot settle
        because the claim is still reserved. The lease eventually expires but
        the compute_job_id IS NULL condition never matches, so the deadlock is
        permanent.

        Zero usage is a fact here rather than an estimate: both the claim and
        the GPU job carry no started_at, and the job is terminal.
        """
        cursor = db.execute(
            """
            UPDATE experiment_attempt_gpu_reservations_v1
            SET status='released', reason_code='attempt_blocked_before_launch',
                actual_gpu_seconds=0, completed_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP
            WHERE status='reserved' AND started_at IS NULL
              AND gpu_job_id IN (
                  SELECT id FROM gpu_jobs
                  WHERE status IN ('failed', 'cancelled', 'canceled', 'timed_out')
                    AND started_at IS NULL
                    AND completed_at IS NOT NULL
              )
            """
        )
        db.commit()
        return int(getattr(cursor, "rowcount", 0) or 0)

    def import_legacy_terminal_attempts(self) -> int:
        """Adopt pre-control-plane terminal GPU jobs exactly once.

        This is a deployment bridge, not an alternative usage source.  It is
        limited to still-active GPU grants, derives usage only from persisted
        GPU job boundaries, rewrites durable compute usage to the canonical
        source, and atomically reconciles the grant/agenda ledgers.  Future
        attempts enter through :meth:`reserve_attempt` and never use this path.
        """
        if not db._use_pg():  # noqa: SLF001
            return 0
        grant_rows = db.fetchall(
            """
            SELECT DISTINCT rg.id
            FROM resource_grants rg
            JOIN gpu_jobs gj ON gj.resource_grant_id=rg.id
            WHERE rg.status='active' AND rg.max_gpu_hours > 0
              AND gj.status IN ('completed','failed','timed_out','cancelled')
              AND gj.started_at IS NOT NULL AND gj.completed_at IS NOT NULL
              AND gj.gpu_attempt_reservation_id IS NULL
            ORDER BY rg.id
            """
        )
        db.commit()
        imported = 0
        for grant_seed in grant_rows:
            try:
                grant = self._grant(int(grant_seed["id"]), lock=True)
                if not grant or str(grant.get("status") or "") != "active":
                    db.commit()
                    continue
                suffix = " FOR UPDATE"
                ledger = db.fetchone(
                    f"SELECT * FROM agenda_resource_ledger WHERE id=?{suffix}",
                    (int(grant["reservation_id"]),),
                )
                if not ledger or str(ledger.get("status") or "") != "reserved":
                    raise AttemptGPUUsageError("legacy_grant_ledger_not_reservable")
                legacy_jobs = db.fetchall(
                    """
                    SELECT gj.*, cj.id AS compute_job_id,
                           cj.backend_kind AS compute_backend_kind
                    FROM gpu_jobs gj
                    LEFT JOIN compute_jobs_v1 cj
                      ON cj.resource_grant_id=gj.resource_grant_id
                     AND cj.idempotency_key=gj.meta_harness_idempotency_key
                    WHERE gj.resource_grant_id=?
                      AND gj.status IN ('completed','failed','timed_out','cancelled')
                      AND gj.started_at IS NOT NULL
                      AND gj.completed_at IS NOT NULL
                    ORDER BY gj.id
                    """,
                    (int(grant["id"]),),
                )
                try:
                    allowed_backends = {
                        str(value)
                        for value in json.loads(
                            grant.get("backend_allowlist_json") or "[]"
                        )
                        if str(value) in GPU_BACKENDS
                    }
                except (TypeError, ValueError):
                    allowed_backends = set()
                for job in legacy_jobs:
                    existing = db.fetchone(
                        """
                        SELECT * FROM experiment_attempt_gpu_reservations_v1
                        WHERE gpu_job_id=? OR
                              (resource_grant_id=? AND attempt_key=?)
                        """,
                        (
                            int(job["id"]),
                            int(grant["id"]),
                            str(
                                job.get("meta_harness_idempotency_key")
                                or f"legacy-gpu-job:{int(job['id'])}"
                            ),
                        ),
                    )
                    if existing:
                        if str(existing.get("status") or "") != "settled":
                            raise AttemptGPUUsageError(
                                "legacy_attempt_import_incomplete",
                                str(existing.get("status") or ""),
                            )
                        continue
                    compute_backend = str(
                        job.get("compute_backend_kind") or ""
                    )
                    if compute_backend in GPU_BACKENDS:
                        backend_kind = compute_backend
                    elif len(allowed_backends) == 1:
                        backend_kind = next(iter(allowed_backends))
                    else:
                        raise AttemptGPUUsageError(
                            "legacy_gpu_backend_ambiguous",
                            f"gpu_job_id={int(job['id'])}",
                        )
                    started_at = _as_utc(job.get("started_at"))
                    completed_at = _as_utc(job.get("completed_at"))
                    if not started_at or not completed_at or completed_at < started_at:
                        raise AttemptGPUUsageError(
                            "legacy_gpu_timestamps_invalid",
                            f"gpu_job_id={int(job['id'])}",
                        )
                    gpu_count = max(1, int(job.get("gpu_count") or 1))
                    wall_seconds = (completed_at - started_at).total_seconds()
                    actual_gpu_seconds = max(0.0, wall_seconds) * gpu_count
                    attempt_key = str(
                        job.get("meta_harness_idempotency_key")
                        or f"legacy-gpu-job:{int(job['id'])}"
                    )
                    reason_code = f"legacy_terminal_import:{str(job['status'])}"
                    reservation_id = db.insert_returning_id(
                        """
                        INSERT INTO experiment_attempt_gpu_reservations_v1
                            (agenda_id, idea_id, resource_grant_id,
                             compute_job_id, experiment_run_id, gpu_job_id,
                             attempt_key, backend_kind, gpu_count,
                             reserved_gpu_seconds, timeout_seconds, status,
                             started_at, completed_at, actual_gpu_seconds,
                             reason_code, lease_expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'settled',
                                ?, ?, ?, ?, ?)
                        RETURNING id
                        """,
                        (
                            int(grant["agenda_id"]),
                            int(grant["idea_id"]),
                            int(grant["id"]),
                            int(job["compute_job_id"])
                            if job.get("compute_job_id")
                            else None,
                            int(job["experiment_run_id"])
                            if job.get("experiment_run_id")
                            else None,
                            int(job["id"]),
                            attempt_key,
                            backend_kind,
                            gpu_count,
                            max(_EPSILON_SECONDS, actual_gpu_seconds),
                            max(1, int(math.ceil(wall_seconds))),
                            started_at.isoformat(),
                            completed_at.isoformat(),
                            actual_gpu_seconds,
                            reason_code,
                            completed_at.isoformat(),
                        ),
                    )
                    db.execute(
                        """
                        UPDATE gpu_jobs SET gpu_attempt_reservation_id=?
                        WHERE id=? AND agenda_id=?
                        """,
                        (
                            reservation_id,
                            int(job["id"]),
                            int(grant["agenda_id"]),
                        ),
                    )
                    if job.get("compute_job_id"):
                        usage_json = json.dumps(
                            {
                                "wall_seconds": wall_seconds,
                                "gpu_hours": actual_gpu_seconds / 3600.0,
                                "cpu_core_hours": 0.0,
                                "backend_report": {
                                    "source": (
                                        "experiment_attempt_gpu_reservations_v1"
                                    ),
                                    "import_source": (
                                        "legacy_terminal_gpu_job_timestamps"
                                    ),
                                    "attempt_reservation_id": reservation_id,
                                    "legacy_gpu_job_id": int(job["id"]),
                                    "gpu_count": gpu_count,
                                    "reason_code": reason_code,
                                },
                            },
                            sort_keys=True,
                        )
                        db.execute(
                            """
                            UPDATE compute_jobs_v1
                            SET gpu_attempt_reservation_id=?, usage_json=?,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=? AND agenda_id=?
                            """,
                            (
                                reservation_id,
                                usage_json,
                                int(job["compute_job_id"]),
                                int(grant["agenda_id"]),
                            ),
                        )
                    imported += 1

                usage = self._usage_locked(grant)
                canonical_hours = usage.settled_gpu_seconds / 3600.0
                ledger_cap = float(ledger.get("gpu_hours_reserved") or 0.0)
                prior_accounted = float(ledger.get("gpu_hours_used") or 0.0) + float(
                    ledger.get("gpu_hours_overrun") or 0.0
                )
                delta_hours = canonical_hours - prior_accounted
                if delta_hours < -1e-9:
                    raise AttemptGPUUsageError(
                        "legacy_gpu_usage_below_existing_ledger"
                    )
                delta_hours = max(0.0, delta_hours)
                reserved_release = min(
                    delta_hours,
                    max(
                        0.0,
                        ledger_cap - float(ledger.get("gpu_hours_used") or 0.0),
                    ),
                )
                db.execute(
                    """
                    UPDATE agenda_resource_ledger
                    SET gpu_hours_used=?, gpu_hours_overrun=?
                    WHERE id=? AND status='reserved'
                    """,
                    (
                        min(ledger_cap, canonical_hours),
                        max(0.0, canonical_hours - ledger_cap),
                        int(grant["reservation_id"]),
                    ),
                )
                if delta_hours:
                    db.execute(
                        """
                        UPDATE research_agendas
                        SET gpu_hours_reserved=gpu_hours_reserved-?,
                            gpu_hours_spent=gpu_hours_spent+?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                        """,
                        (
                            reserved_release,
                            delta_hours,
                            int(grant["agenda_id"]),
                        ),
                    )
                if usage.exhausted:
                    self._finalize_grant_locked(
                        grant,
                        usage=usage,
                        reason_code="grant_gpu_hours_exhausted",
                    )
                db.commit()
            except Exception:
                db.rollback()
                raise
        return imported
