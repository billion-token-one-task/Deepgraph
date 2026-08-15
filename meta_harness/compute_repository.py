"""Durable compute submission/idempotency state for PostgreSQL.

A claim is committed before a backend call. If the process loses the backend
response, the claim becomes ``submission_unknown`` and cannot be resubmitted
under the same idempotency key until an operator reconciles it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from db import database as db
from meta_harness.compute import (
    ACTIVE_JOB_STATES,
    ArtifactCollection,
    ComputeBackendError,
    ComputeClaim,
    ComputeJob,
    ComputeSubmission,
    TERMINAL_JOB_STATES,
    UsageAccounting,
)
from meta_harness.attempt_gpu_usage import GrantGPUUsageControl


_BACKENDS = {"cpu", "local_gpu", "ssh_gpu", "colab_gpu"}
_LEGACY_JOB_PREFIX = "legacy-gpu-job:"
_TERMINAL = {"succeeded", "failed", "cancelled", "timed_out"}
_TRANSITIONS = {
    "submitting": {"submitted", "running", "submission_unknown"},
    "submitted": {
        "running",
        "cancel_requested",
        "collecting",
        "failed",
        "cancelled",
        "timed_out",
    },
    "running": {
        "cancel_requested",
        "collecting",
        "failed",
        "cancelled",
        "timed_out",
    },
    "cancel_requested": {"cancelled", "failed", "timed_out"},
    "collecting": {"succeeded", "failed", "timed_out"},
    "submission_unknown": set(),
    "usage_unknown": set(),
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
    "timed_out": set(),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _usage_is_zero(usage_json) -> bool:
    """True when recorded usage exists and every metered quantity is zero.

    A pre-launch refusal still writes a usage record -- zeroed, annotated with
    the reason the attempt never began. Testing for an absent record therefore
    misses exactly the case that matters. Any unparsable or non-zero usage is
    treated as real work.
    """
    text = str(usage_json or "").strip()
    if not text:
        return True
    try:
        usage = json.loads(text)
    except (TypeError, ValueError):
        return False
    if not isinstance(usage, Mapping):
        return False
    for field in ("gpu_hours", "wall_seconds", "cpu_core_hours"):
        try:
            if float(usage.get(field) or 0.0) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _never_reached_backend(row) -> bool:
    """True when a terminal job was refused before any backend started it.

    Judged on the recorded facts rather than a failure_reason string: the job
    is terminal, no backend job id was ever bound, and no usage was recorded.
    Such a job did not happen, so it must not consume the run's only
    idempotency key. Anything that reached a backend stays terminal.
    """
    if str((row or {}).get("status") or "") not in TERMINAL_JOB_STATES:
        return False
    if not _usage_is_zero((row or {}).get("usage_json")):
        return False
    backend_job_id = str((row or {}).get("backend_job_id") or "").strip()
    if not backend_job_id:
        return True
    # A legacy mirror's backend_job_id points at a gpu_jobs row rather than at
    # anything a backend issued, so its presence alone does not mean the work
    # started. Consult the row it names.
    if not backend_job_id.startswith(_LEGACY_JOB_PREFIX):
        return False
    try:
        legacy_id = int(backend_job_id[len(_LEGACY_JOB_PREFIX):])
    except ValueError:
        return False
    legacy = db.fetchone(
        "SELECT started_at, status FROM gpu_jobs WHERE id=?", (legacy_id,)
    )
    if not legacy or legacy.get("started_at") is not None:
        return False
    return str(legacy.get("status") or "") in TERMINAL_JOB_STATES | {"canceled"}


def _require_postgresql() -> None:
    if not db._use_pg():  # noqa: SLF001
        raise ComputeBackendError(
            "durable compute job persistence requires isolated PostgreSQL"
        )


def _expect_one(cursor: Any, *, operation: str) -> None:
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        raise ComputeBackendError(f"durable compute state race:{operation}")


def _manifest_requirement_names(manifest: Mapping[str, Any]) -> set[str]:
    """Extract explicit artifact names from the stable manifest shapes."""
    names = {str(key) for key in manifest}
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        names.update(str(key) for key in artifacts)
    elif isinstance(artifacts, (list, tuple)):
        for item in artifacts:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, Mapping):
                name = item.get("name") or item.get("kind")
                if name:
                    names.add(str(name))
    return names


class ComputeJobRepository:
    def record_id_for_job(self, job: ComputeJob) -> int:
        job.validate()
        _require_postgresql()
        row = db.fetchone(
            """
            SELECT id FROM compute_jobs_v1
            WHERE backend_kind=? AND backend_job_id=?
            """,
            (job.backend_kind, job.backend_job_id),
        )
        if not row:
            raise ComputeBackendError("durable compute job not found")
        return int(row["id"])

    def _claim_from_row(self, row: dict, *, is_new: bool) -> ComputeClaim:
        return ComputeClaim(
            record_id=int(row["id"]),
            is_new=is_new,
            backend_kind=str(row["backend_kind"]),
            idempotency_key=str(row["idempotency_key"]),
            status=str(row["status"]),
            backend_job_id=str(row["backend_job_id"])
            if row.get("backend_job_id")
            else None,
            heartbeat_at=str(row["heartbeat_at"])
            if row.get("heartbeat_at")
            else None,
            failure_reason=str(row["failure_reason"])
            if row.get("failure_reason")
            else None,
        )

    def _validate_existing(
        self,
        row: dict,
        request: ComputeSubmission,
        *,
        backend_kind: str,
    ) -> None:
        expected = {
            "agenda_id": request.agenda_id,
            "idea_id": request.idea_id,
            "resource_grant_id": request.resource_grant_id,
            "stage": request.stage,
            "backend_kind": backend_kind,
            "command_ref": request.command_ref,
            "artifact_namespace": request.artifact_namespace,
            "timeout_seconds": request.timeout_seconds,
        }
        mismatches = [
            key
            for key, value in expected.items()
            if str(row.get(key)) != str(value)
        ]
        if abs(
            float(row.get("requested_gpu_hours") or 0)
            - request.requested_gpu_hours
        ) > 1e-12:
            mismatches.append("requested_gpu_hours")
        if mismatches:
            raise ComputeBackendError(
                "idempotency_key_reused_with_different_request:"
                + ",".join(sorted(set(mismatches)))
            )

    def claim(
        self,
        request: ComputeSubmission,
        *,
        backend_kind: str,
    ) -> ComputeClaim:
        request.validate()
        if backend_kind not in _BACKENDS:
            raise ComputeBackendError("unknown compute backend")
        _require_postgresql()
        try:
            existing = db.fetchone(
                """
                SELECT * FROM compute_jobs_v1
                WHERE agenda_id=? AND idempotency_key=? FOR UPDATE
                """,
                (request.agenda_id, request.idempotency_key),
            )
            if existing and _never_reached_backend(existing):
                # idempotency_key is derived from agenda, idea, run and stage,
                # so it is the only key a run will ever present. A job that
                # failed before the backend ever started it consumed nothing,
                # yet leaving it terminal made the caller raise
                # idempotency_key_already_terminal for every later attempt and
                # put the run permanently beyond retry. Reopen that row rather
                # than refuse; a job that actually reached a backend is left
                # terminal so a real attempt is never silently repeated.
                db.execute(
                    """
                    UPDATE compute_jobs_v1
                    SET status='submitting', failure_reason=NULL,
                        backend_job_id=NULL, heartbeat_at=?, timeout_at=?,
                        resource_grant_id=?, requested_gpu_hours=?,
                        timeout_seconds=?, backend_kind=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND status=?
                    """,
                    (
                        _now(),
                        _now() + timedelta(seconds=request.timeout_seconds),
                        request.resource_grant_id,
                        request.requested_gpu_hours,
                        request.timeout_seconds,
                        backend_kind,
                        int(existing["id"]),
                        str(existing.get("status")),
                    ),
                )
                reopened = db.fetchone(
                    "SELECT * FROM compute_jobs_v1 WHERE id=?",
                    (int(existing["id"]),),
                )
                db.commit()
                return self._claim_from_row(reopened or existing, is_new=True)
            if existing:
                self._validate_existing(existing, request, backend_kind=backend_kind)
                db.commit()
                return self._claim_from_row(existing, is_new=False)
            grant = db.fetchone(
                """
                SELECT agenda_id, idea_id, stage, status, expires_at,
                       max_gpu_hours, backend_allowlist_json
                FROM resource_grants
                WHERE id=? FOR UPDATE
                """,
                (request.resource_grant_id,),
            )
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != request.agenda_id
                or int(grant.get("idea_id") or 0) != request.idea_id
                or str(grant.get("stage") or "") != request.stage
                or str(grant.get("status") or "") != "active"
            ):
                raise ComputeBackendError("persisted ResourceGrant scope is invalid")
            expires_at = datetime.fromisoformat(
                str(grant["expires_at"]).replace("Z", "+00:00")
            )
            if expires_at.tzinfo is None or expires_at.astimezone(timezone.utc) <= _now():
                raise ComputeBackendError("persisted ResourceGrant is expired")
            allowlist = set(
                json.loads(grant.get("backend_allowlist_json") or "[]")
            )
            if backend_kind not in allowlist:
                raise ComputeBackendError("backend is not allowed by persisted grant")
            if request.requested_gpu_hours > float(
                grant.get("max_gpu_hours") or 0
            ):
                raise ComputeBackendError("compute request exceeds persisted GPU cap")
            now = _now()
            timeout_at = now + timedelta(seconds=request.timeout_seconds)
            inserted = db.fetchone(
                """
                INSERT INTO compute_jobs_v1
                    (agenda_id, idea_id, resource_grant_id, stage, backend_kind,
                     idempotency_key, command_ref, artifact_namespace,
                     requested_gpu_hours, timeout_seconds, status, heartbeat_at,
                     timeout_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitting', ?, ?)
                ON CONFLICT (agenda_id, idempotency_key) DO NOTHING
                RETURNING id
                """,
                (
                    request.agenda_id,
                    request.idea_id,
                    request.resource_grant_id,
                    request.stage,
                    backend_kind,
                    request.idempotency_key,
                    request.command_ref,
                    request.artifact_namespace,
                    request.requested_gpu_hours,
                    request.timeout_seconds,
                    now.isoformat(),
                    timeout_at.isoformat(),
                ),
            )
            if not inserted:
                concurrent = db.fetchone(
                    """
                    SELECT * FROM compute_jobs_v1
                    WHERE agenda_id=? AND idempotency_key=? FOR UPDATE
                    """,
                    (request.agenda_id, request.idempotency_key),
                )
                if not concurrent:
                    raise ComputeBackendError(
                        "idempotency conflict row disappeared"
                    )
                self._validate_existing(
                    concurrent,
                    request,
                    backend_kind=backend_kind,
                )
                db.commit()
                return self._claim_from_row(concurrent, is_new=False)
            record_id = int(inserted["id"])
            if backend_kind != "cpu":
                attempt = db.fetchone(
                    """
                    SELECT id, reserved_gpu_seconds, timeout_seconds, status
                    FROM experiment_attempt_gpu_reservations_v1
                    WHERE resource_grant_id=? AND attempt_key=?
                    """,
                    (request.resource_grant_id, request.idempotency_key),
                )
                if (
                    not attempt
                    or str(attempt.get("status")) not in {"reserved", "running"}
                    or int(attempt.get("timeout_seconds") or 0)
                    != request.timeout_seconds
                    or abs(
                        float(attempt.get("reserved_gpu_seconds") or 0.0)
                        - request.requested_gpu_hours * 3600.0
                    )
                    > 1e-6
                ):
                    raise ComputeBackendError(
                        "canonical GPU attempt reservation is missing or mismatched"
                    )
                GrantGPUUsageControl().bind_compute_job(
                    int(attempt["id"]), record_id, commit=False
                )
            db.commit()
            return ComputeClaim(
                record_id=record_id,
                is_new=True,
                backend_kind=backend_kind,
                idempotency_key=request.idempotency_key,
                status="submitting",
            )
        except Exception:
            db.rollback()
            raise

    def bind_submitted_job(self, record_id: int, job: ComputeJob) -> None:
        job.validate()
        if job.status not in ACTIVE_JOB_STATES:
            raise ComputeBackendError("only a live backend job can bind a claim")
        _require_postgresql()
        try:
            row = db.fetchone(
                "SELECT * FROM compute_jobs_v1 WHERE id=? FOR UPDATE",
                (int(record_id),),
            )
            if not row or row.get("status") != "submitting":
                raise ComputeBackendError("compute claim is not bindable")
            if row.get("backend_kind") != job.backend_kind:
                raise ComputeBackendError("backend kind changed after claim")
            if row.get("idempotency_key") != job.idempotency_key:
                raise ComputeBackendError("backend idempotency key mismatch")
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET backend_job_id=?, status=?, heartbeat_at=?,
                    failure_reason=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='submitting'
                """,
                (
                    job.backend_job_id,
                    job.status,
                    job.heartbeat_at or _now().isoformat(),
                    record_id,
                    int(row["agenda_id"]),
                ),
            )
            _expect_one(cursor, operation="bind_submitted_job")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def mark_submission_unknown(self, record_id: int, *, reason: str) -> None:
        if not str(reason or "").strip():
            raise ComputeBackendError("submission uncertainty reason is required")
        _require_postgresql()
        try:
            row = db.fetchone(
                "SELECT agenda_id FROM compute_jobs_v1 WHERE id=? FOR UPDATE",
                (int(record_id),),
            )
            if not row:
                raise ComputeBackendError("durable compute job not found")
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='submission_unknown', failure_reason=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='submitting'
                """,
                (reason, int(record_id), int(row["agenda_id"])),
            )
            _expect_one(cursor, operation="mark_submission_unknown")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def record_backend_state(self, job: ComputeJob) -> str:
        """Persist a truthful backend state; success first enters collecting."""
        job.validate()
        if job.status in {"failed", "cancelled", "timed_out"}:
            raise ComputeBackendError(
                "terminal backend state requires durable usage settlement"
            )
        _require_postgresql()
        try:
            row = db.fetchone(
                """
                SELECT * FROM compute_jobs_v1
                WHERE backend_kind=? AND backend_job_id=? FOR UPDATE
                """,
                (job.backend_kind, job.backend_job_id),
            )
            if not row:
                raise ComputeBackendError("durable compute job not found")
            current = str(row["status"])
            target = "collecting" if job.status == "succeeded" else job.status
            if target == current:
                db.commit()
                return current
            if target not in _TRANSITIONS.get(current, set()):
                raise ComputeBackendError(
                    f"invalid compute state transition:{current}->{target}"
                )
            failure_reason = job.failure_reason
            if target in {"failed", "timed_out"} and not failure_reason:
                failure_reason = f"backend_{target}"
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status=?, heartbeat_at=?, failure_reason=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status=?
                """,
                (
                    target,
                    job.heartbeat_at or _now().isoformat(),
                    failure_reason,
                    int(row["id"]),
                    int(row["agenda_id"]),
                    current,
                ),
            )
            _expect_one(cursor, operation="record_backend_state")
            db.commit()
            return target
        except Exception:
            db.rollback()
            raise

    def finalize_terminal(
        self,
        job: ComputeJob,
        *,
        usage: UsageAccounting,
    ) -> None:
        """Persist measured usage and a truthful non-success terminal state."""
        job.validate()
        if job.status not in {"failed", "cancelled", "timed_out"}:
            raise ComputeBackendError(
                "finalize_terminal requires a non-success terminal job"
            )
        usage.validate()
        _require_postgresql()
        try:
            row = db.fetchone(
                """
                SELECT j.id, j.agenda_id, j.status, j.requested_gpu_hours,
                       g.max_gpu_hours
                FROM compute_jobs_v1 AS j
                JOIN resource_grants AS g ON g.id=j.resource_grant_id
                WHERE j.backend_kind=? AND j.backend_job_id=? FOR UPDATE
                """,
                (job.backend_kind, job.backend_job_id),
            )
            if not row:
                raise ComputeBackendError("durable compute job not found")
            current = str(row.get("status") or "")
            if job.status not in _TRANSITIONS.get(current, set()):
                raise ComputeBackendError(
                    f"invalid compute state transition:{current}->{job.status}"
                )
            requested_cap = float(row.get("requested_gpu_hours") or 0)
            grant_cap = float(row.get("max_gpu_hours") or 0)
            if usage.gpu_hours > requested_cap or usage.gpu_hours > grant_cap:
                raise ComputeBackendError(
                    "reported GPU usage exceeds request or ResourceGrant cap"
                )
            failure_reason = (
                job.failure_reason
                or f"backend_{job.status}"
            )
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status=?, heartbeat_at=?, failure_reason=?, usage_json=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status=?
                """,
                (
                    job.status,
                    job.heartbeat_at or _now().isoformat(),
                    failure_reason,
                    _dump(
                        {
                            "wall_seconds": usage.wall_seconds,
                            "gpu_hours": usage.gpu_hours,
                            "cpu_core_hours": usage.cpu_core_hours,
                            "backend_report": usage.backend_report,
                        }
                    ),
                    int(row["id"]),
                    int(row["agenda_id"]),
                    current,
                ),
            )
            _expect_one(cursor, operation="finalize_terminal")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def finalize_success(
        self,
        record_id: int,
        *,
        artifacts: ArtifactCollection,
        usage: UsageAccounting,
    ) -> None:
        if not artifacts.complete or artifacts.missing_requirements:
            raise ComputeBackendError("cannot finalize incomplete artifacts")
        if not artifacts.manifest:
            raise ComputeBackendError("artifact manifest is required")
        usage.validate()
        _require_postgresql()
        try:
            row = db.fetchone(
                """
                SELECT j.agenda_id, j.status, j.requested_gpu_hours,
                       g.max_gpu_hours,
                       g.artifact_requirements_json
                FROM compute_jobs_v1 AS j
                JOIN resource_grants AS g ON g.id=j.resource_grant_id
                WHERE j.id=? FOR UPDATE
                """,
                (int(record_id),),
            )
            if not row or row.get("status") != "collecting":
                raise ComputeBackendError("compute job is not collecting artifacts")
            requested_cap = float(row.get("requested_gpu_hours") or 0)
            grant_cap = float(row.get("max_gpu_hours") or 0)
            if usage.gpu_hours > requested_cap or usage.gpu_hours > grant_cap:
                raise ComputeBackendError(
                    "reported GPU usage exceeds request or ResourceGrant cap"
                )
            required = set(
                json.loads(row.get("artifact_requirements_json") or "[]")
            )
            missing = required - _manifest_requirement_names(artifacts.manifest)
            if missing:
                raise ComputeBackendError(
                    "artifact manifest misses persisted grant requirements:"
                    + ",".join(sorted(str(item) for item in missing))
                )
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='succeeded', artifact_manifest_json=?, usage_json=?,
                    failure_reason=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='collecting'
                """,
                (
                    _dump(artifacts.manifest),
                    _dump(
                        {
                            "wall_seconds": usage.wall_seconds,
                            "gpu_hours": usage.gpu_hours,
                            "cpu_core_hours": usage.cpu_core_hours,
                            "backend_report": usage.backend_report,
                        }
                    ),
                    record_id,
                    int(row["agenda_id"]),
                ),
            )
            _expect_one(cursor, operation="finalize_success")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def reconcile_expired(self, *, agenda_id: int) -> dict[str, int]:
        """Fail closed after restart; never resubmit or mark success."""
        if int(agenda_id or 0) <= 0:
            raise ComputeBackendError(
                "compute recovery requires an explicit agenda scope"
            )
        _require_postgresql()
        try:
            unknown = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='submission_unknown',
                    failure_reason='process_restarted_before_submission_bind',
                    updated_at=CURRENT_TIMESTAMP
                WHERE agenda_id=? AND status='submitting'
                  AND timeout_at <= CURRENT_TIMESTAMP
                """
                ,
                (int(agenda_id),),
            )
            usage_unknown = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='usage_unknown',
                    failure_reason='durable_compute_timeout_usage_unknown',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status IN ('submitted', 'running', 'cancel_requested',
                                 'collecting')
                  AND agenda_id=?
                  AND timeout_at <= CURRENT_TIMESTAMP
                  AND NOT EXISTS (
                      SELECT 1
                      FROM experiment_attempt_gpu_reservations_v1 AS ar
                      WHERE ar.compute_job_id=compute_jobs_v1.id
                        AND ar.status='running'
                  )
                """
                ,
                (int(agenda_id),),
            )
            db.commit()
            return {
                "submission_unknown": int(
                    getattr(unknown, "rowcount", 0) or 0
                ),
                "usage_unknown": int(
                    getattr(usage_unknown, "rowcount", 0) or 0
                ),
            }
        except Exception:
            db.rollback()
            raise

    def reconciliation_queue(self, *, agenda_id: int | None = None) -> list[dict]:
        _require_postgresql()
        params: tuple[Any, ...] = ()
        scope = ""
        if agenda_id is not None:
            if int(agenda_id) <= 0:
                raise ComputeBackendError("agenda_id must be positive")
            scope = " AND agenda_id=?"
            params = (int(agenda_id),)
        return db.fetchall(
            f"""
            SELECT id, agenda_id, idea_id, resource_grant_id, backend_kind,
                   backend_job_id, idempotency_key, status, heartbeat_at,
                   timeout_at, failure_reason
            FROM compute_jobs_v1
            WHERE status IN ('submission_unknown', 'collecting',
                             'usage_unknown'){scope}
            ORDER BY updated_at ASC, id ASC
            """,
            params,
        )
