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
    UsageAccounting,
)


_BACKENDS = {"cpu", "local_gpu", "ssh_gpu", "colab_gpu"}
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
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
    "timed_out": set(),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


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
                WHERE id=? AND status='submitting'
                """,
                (
                    job.backend_job_id,
                    job.status,
                    job.heartbeat_at or _now().isoformat(),
                    record_id,
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
            cursor = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='submission_unknown', failure_reason=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='submitting'
                """,
                (reason, int(record_id)),
            )
            _expect_one(cursor, operation="mark_submission_unknown")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def record_backend_state(self, job: ComputeJob) -> str:
        """Persist a truthful backend state; success first enters collecting."""
        job.validate()
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
                WHERE id=? AND status=?
                """,
                (
                    target,
                    job.heartbeat_at or _now().isoformat(),
                    failure_reason,
                    int(row["id"]),
                    current,
                ),
            )
            _expect_one(cursor, operation="record_backend_state")
            db.commit()
            return target
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
                SELECT j.status, j.requested_gpu_hours, g.max_gpu_hours,
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
                WHERE id=? AND status='collecting'
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
                ),
            )
            _expect_one(cursor, operation="finalize_success")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def reconcile_expired(self) -> dict[str, int]:
        """Fail closed after restart; never resubmit or mark success."""
        _require_postgresql()
        try:
            unknown = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='submission_unknown',
                    failure_reason='process_restarted_before_submission_bind',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status='submitting' AND timeout_at <= CURRENT_TIMESTAMP
                """
            )
            timed_out = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='timed_out',
                    failure_reason='durable_compute_timeout',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status IN ('submitted', 'running', 'cancel_requested',
                                 'collecting')
                  AND timeout_at <= CURRENT_TIMESTAMP
                """
            )
            db.commit()
            return {
                "submission_unknown": int(
                    getattr(unknown, "rowcount", 0) or 0
                ),
                "timed_out": int(getattr(timed_out, "rowcount", 0) or 0),
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
            WHERE status IN ('submission_unknown', 'collecting'){scope}
            ORDER BY updated_at ASC, id ASC
            """,
            params,
        )
