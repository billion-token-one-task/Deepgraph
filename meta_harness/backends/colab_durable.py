"""Durable Colab admission, transport state, and worker claims.

The operator request is persisted before ``ComputeScheduler`` admission.  The
backend ``submit`` call only binds a deterministic identity; it never starts a
Colab session.  A single durable worker later claims the joined request and
compute row before invoking the isolated CLI executor.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from contracts.meta_harness import ResourceGrant
from db import database as db
from meta_harness.backends.colab_cli import (
    ColabCLIConfig,
    ColabCLIError,
    ColabCLIExecutor,
    ColabExecutionRequest,
    ColabExecutionResult,
)
from meta_harness.compute import (
    ArtifactCollection,
    BackendCapability,
    ColabAccount,
    ComputeBackendError,
    ComputeJob,
    ComputeSubmission,
    UsageAccounting,
)
from meta_harness.attempt_gpu_usage import GrantGPUUsageControl


_BACKEND_PREFIX = "colab-work-request:"
_ENV_REF = re.compile(r"env:([A-Z][A-Z0-9_]*)\Z")
_WORK_STATES = {
    "admitting",
    "queued",
    "running",
    "succeeded",
    "failed",
    "timed_out",
    "cancelled",
    "manual_reconciliation",
}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_object(value: Any, *, label: str) -> dict:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ColabCLIError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ColabCLIError(f"{label} must be a JSON object")
    return parsed


def _load_list(value: Any, *, label: str) -> list:
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ColabCLIError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ColabCLIError(f"{label} must be a JSON array")
    return parsed


def _request_id(backend_job_id: str) -> int:
    if not str(backend_job_id).startswith(_BACKEND_PREFIX):
        raise ColabCLIError("Colab backend job id is invalid")
    try:
        value = int(str(backend_job_id)[len(_BACKEND_PREFIX) :])
    except ValueError as exc:
        raise ColabCLIError("Colab backend job id is invalid") from exc
    if value <= 0:
        raise ColabCLIError("Colab backend job id is invalid")
    return value


def _request_ref(command_ref: str) -> int:
    return _request_id(command_ref)


def _safe_relative(value: str, *, label: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ColabCLIError(f"{label} must be a safe relative path")
    normalized = path.as_posix().lstrip("./")
    if not normalized:
        raise ColabCLIError(f"{label} cannot be empty")
    return normalized


@dataclass(frozen=True)
class ColabWorkSpec:
    agenda_id: int
    idea_id: int
    experiment_run_id: int
    resource_grant_id: int
    stage: str
    idempotency_key: str
    code_dir: str
    command_tokens: tuple[str, ...]
    environment: Mapping[str, str]
    artifact_map: Mapping[str, str]
    artifact_output_dir: str
    timeout_seconds: int

    def validate(self) -> None:
        if min(
            self.agenda_id,
            self.idea_id,
            self.experiment_run_id,
            self.resource_grant_id,
        ) <= 0:
            raise ColabCLIError("Colab work requires positive scope ids")
        if not self.stage.strip() or not self.idempotency_key.strip():
            raise ColabCLIError("Colab stage and idempotency key are required")
        if not self.command_tokens or any(
            not str(token).strip() for token in self.command_tokens
        ):
            raise ColabCLIError("Colab command tokens must be non-empty")
        if self.timeout_seconds <= 0:
            raise ColabCLIError("Colab timeout must be positive")
        if not self.artifact_map:
            raise ColabCLIError("Colab artifact map is required")
        for name, relative in self.artifact_map.items():
            if not str(name).strip():
                raise ColabCLIError("Colab artifact names cannot be empty")
            _safe_relative(str(relative), label=f"artifact {name}")


class ColabWorkRepository:
    """PostgreSQL-only queue with scope-checked immutable request payloads."""

    def create(self, spec: ColabWorkSpec) -> int:
        spec.validate()
        if not db._use_pg():  # noqa: SLF001
            raise ColabCLIError("durable Colab queue requires PostgreSQL")
        try:
            existing = db.fetchone(
                """
                SELECT * FROM colab_work_requests_v1
                WHERE agenda_id=? AND idempotency_key=? FOR UPDATE
                """,
                (spec.agenda_id, spec.idempotency_key),
            )
            if existing:
                expected = {
                    "idea_id": spec.idea_id,
                    "experiment_run_id": spec.experiment_run_id,
                    "resource_grant_id": spec.resource_grant_id,
                    "stage": spec.stage,
                    "code_dir": spec.code_dir,
                    "artifact_output_dir": spec.artifact_output_dir,
                    "timeout_seconds": spec.timeout_seconds,
                    "command_tokens_json": _dump(list(spec.command_tokens)),
                    "environment_json": _dump(dict(spec.environment)),
                    "artifact_map_json": _dump(dict(spec.artifact_map)),
                }
                mismatches = [
                    key
                    for key, value in expected.items()
                    if str(existing.get(key)) != str(value)
                ]
                if mismatches:
                    raise ColabCLIError(
                        "Colab idempotency key reused with different request:"
                        + ",".join(sorted(mismatches))
                    )
                db.commit()
                return int(existing["id"])
            scope = db.fetchone(
                """
                SELECT er.agenda_id, er.deep_insight_id,
                       er.resource_grant_id, rg.stage, rg.status,
                       rg.backend_allowlist_json,
                       rg.artifact_requirements_json
                FROM experiment_runs AS er
                JOIN resource_grants AS rg ON rg.id=er.resource_grant_id
                WHERE er.id=? AND rg.id=? AND rg.expires_at > CURRENT_TIMESTAMP
                FOR UPDATE
                """,
                (spec.experiment_run_id, spec.resource_grant_id),
            )
            if (
                not scope
                or int(scope.get("agenda_id") or 0) != spec.agenda_id
                or int(scope.get("deep_insight_id") or 0) != spec.idea_id
                or int(scope.get("resource_grant_id") or 0)
                != spec.resource_grant_id
                or str(scope.get("stage") or "") != spec.stage
                or str(scope.get("status") or "") != "active"
            ):
                raise ColabCLIError(
                    "Colab request does not match an active run ResourceGrant"
                )
            allowlist = set(
                _load_list(
                    scope.get("backend_allowlist_json"),
                    label="ResourceGrant backend allowlist",
                )
            )
            requirements = set(
                _load_list(
                    scope.get("artifact_requirements_json"),
                    label="ResourceGrant artifact requirements",
                )
            )
            if "colab_gpu" not in allowlist:
                raise ColabCLIError("ResourceGrant does not allow Colab")
            missing = requirements - set(spec.artifact_map)
            if missing:
                raise ColabCLIError(
                    "Colab artifact map misses ResourceGrant requirements:"
                    + ",".join(sorted(str(value) for value in missing))
                )
            request_id = db.insert_returning_id(
                """
                INSERT INTO colab_work_requests_v1
                    (agenda_id, idea_id, experiment_run_id, resource_grant_id,
                     stage, idempotency_key, code_dir, command_tokens_json,
                     environment_json, artifact_map_json, artifact_output_dir,
                     timeout_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'admitting')
                RETURNING id
                """,
                (
                    spec.agenda_id,
                    spec.idea_id,
                    spec.experiment_run_id,
                    spec.resource_grant_id,
                    spec.stage,
                    spec.idempotency_key,
                    spec.code_dir,
                    _dump(list(spec.command_tokens)),
                    _dump(dict(spec.environment)),
                    _dump(dict(spec.artifact_map)),
                    spec.artifact_output_dir,
                    spec.timeout_seconds,
                ),
            )
            db.commit()
            return int(request_id)
        except Exception:
            db.rollback()
            raise

    def bind_compute_job(self, request_id: int) -> int:
        if not db._use_pg():  # noqa: SLF001
            raise ColabCLIError("durable Colab queue requires PostgreSQL")
        try:
            row = db.fetchone(
                """
                SELECT cwr.*, cj.id AS durable_compute_job_id,
                       cj.status AS compute_status,
                       cj.timeout_seconds AS durable_timeout_seconds
                FROM colab_work_requests_v1 AS cwr
                JOIN compute_jobs_v1 AS cj
                  ON cj.agenda_id=cwr.agenda_id
                 AND cj.idempotency_key=cwr.idempotency_key
                 AND cj.backend_kind='colab_gpu'
                 AND cj.backend_job_id=?
                WHERE cwr.id=? FOR UPDATE
                """,
                (f"{_BACKEND_PREFIX}{int(request_id)}", int(request_id)),
            )
            if not row:
                raise ColabCLIError("Colab compute admission was not persisted")
            compute_job_id = int(row["durable_compute_job_id"])
            if row.get("compute_job_id") not in {None, compute_job_id}:
                raise ColabCLIError("Colab request is bound to another compute job")
            if str(row.get("status") or "") == "admitting":
                db.execute(
                    """
                    UPDATE colab_work_requests_v1
                    SET compute_job_id=?, status='queued',
                        timeout_seconds=LEAST(timeout_seconds, ?),
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='admitting'
                    """,
                    (
                        compute_job_id,
                        int(row["durable_timeout_seconds"]),
                        int(request_id),
                        int(row["agenda_id"]),
                    ),
                )
            elif str(row.get("status") or "") not in _WORK_STATES:
                raise ColabCLIError("Colab request has an invalid persisted state")
            db.commit()
            return compute_job_id
        except Exception:
            db.rollback()
            raise

    def claim_next(self, *, worker_id: str) -> dict | None:
        if not worker_id.strip():
            raise ColabCLIError("Colab worker identity is required")
        if not db._use_pg():  # noqa: SLF001
            raise ColabCLIError("durable Colab worker requires PostgreSQL")
        try:
            row = db.fetchone(
                """
                SELECT cwr.*, cj.backend_job_id, cj.status AS compute_status,
                       cj.gpu_attempt_reservation_id
                FROM colab_work_requests_v1 AS cwr
                JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
                JOIN resource_grants AS rg ON rg.id=cwr.resource_grant_id
                WHERE cwr.status='queued' AND cj.status='submitted'
                  AND rg.status='active'
                  AND rg.expires_at > CURRENT_TIMESTAMP
                ORDER BY cwr.created_at, cwr.id
                LIMIT 1 FOR UPDATE OF cwr, cj SKIP LOCKED
                """
            )
            if not row:
                db.commit()
                return None
            now = datetime.now(timezone.utc).isoformat()
            attempt_reservation_id = int(
                row.get("gpu_attempt_reservation_id") or 0
            )
            if attempt_reservation_id <= 0:
                raise ColabCLIError(
                    "Colab canonical GPU attempt reservation is missing"
                )
            GrantGPUUsageControl().start_attempt(
                attempt_reservation_id,
                started_at=datetime.fromisoformat(now),
                commit=False,
            )
            changed = db.execute(
                """
                UPDATE colab_work_requests_v1
                SET status='running', worker_id=?, attempt_count=attempt_count+1,
                    started_at=COALESCE(started_at, ?),
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='queued'
                """,
                (worker_id, now, int(row["id"]), int(row["agenda_id"])),
            )
            if int(getattr(changed, "rowcount", 0) or 0) != 1:
                raise ColabCLIError("Colab work claim race")
            changed_compute = db.execute(
                """
                UPDATE compute_jobs_v1
                SET status='running', heartbeat_at=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='submitted'
                """,
                (
                    now,
                    int(row["compute_job_id"]),
                    int(row["agenda_id"]),
                ),
            )
            if int(getattr(changed_compute, "rowcount", 0) or 0) != 1:
                raise ColabCLIError("Colab compute claim race")
            db.commit()
            row["status"] = "running"
            row["compute_status"] = "running"
            row["worker_id"] = worker_id
            return row
        except Exception:
            db.rollback()
            raise

    def save_result(
        self,
        request_id: int,
        *,
        result: ColabExecutionResult,
    ) -> None:
        if result.status not in {"succeeded", "failed", "timed_out"}:
            raise ColabCLIError("Colab executor returned an invalid terminal state")
        try:
            row = db.fetchone(
                """
                SELECT agenda_id FROM colab_work_requests_v1
                WHERE id=? FOR UPDATE
                """,
                (int(request_id),),
            )
            if not row:
                raise ColabCLIError("Colab work request was not found")
            changed = db.execute(
                """
                UPDATE colab_work_requests_v1
                SET status=?, account_ref=?, session_ref=?, result_json=?,
                    artifact_manifest_json=?, wall_seconds=?,
                    failure_reason=?, completed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                """,
                (
                    result.status,
                    result.account_ref,
                    result.session,
                    _dump(
                        {
                            "returncode": result.returncode,
                            "stdout_sha256": __import__("hashlib").sha256(
                                result.stdout.encode("utf-8")
                            ).hexdigest(),
                            "gpu_type": result.gpu_type,
                        }
                    ),
                    _dump(result.artifact_manifest),
                    result.wall_seconds,
                    result.failure_reason,
                    int(request_id),
                    int(row["agenda_id"]),
                ),
            )
            if int(getattr(changed, "rowcount", 0) or 0) != 1:
                raise ColabCLIError("Colab result persistence race")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def requeue_control_lost(self) -> int:
        """Re-queue requests a controller defect failed before any session.

        claim_next only takes queued requests, so a request the worker dropped
        on its own bug stays failed for good even once that bug is fixed. The
        idempotency key is derived from the run, so no replacement request can
        be created either, and the experiment is stranded holding a scaffold it
        already paid for.

        Recovery is limited to what is provably a non-attempt: the worker never
        recorded a session or a result, and the request is still bound to a
        compute job that never reached a backend. Anything that reached Colab
        keeps its terminal state so remote usage is never double counted.
        """
        if not db._use_pg():  # noqa: SLF001
            raise ColabCLIError("durable Colab recovery requires PostgreSQL")
        requeued = 0
        try:
            rows = db.fetchall(
                """
                SELECT cwr.id, cwr.agenda_id, cwr.compute_job_id
                FROM colab_work_requests_v1 AS cwr
                JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
                WHERE cwr.status='failed'
                  AND cwr.session_ref IS NULL
                  AND cwr.result_json IS NULL
                  AND cwr.failure_reason LIKE 'colab_worker_control_lost:%'
                  -- The compute row's backend_job_id is this request's own
                  -- reference, assigned at admission; it is not a Colab
                  -- session and says nothing about whether work began. The
                  -- session and result columns above are that evidence.
                  AND (
                        cj.backend_job_id IS NULL
                     OR cj.backend_job_id = 'colab-work-request:' || cwr.id
                  )
                FOR UPDATE OF cwr, cj
                """
            )
            for row in rows:
                db.execute(
                    """
                    UPDATE colab_work_requests_v1
                    SET status='queued', worker_id=NULL, started_at=NULL,
                        completed_at=NULL, failure_reason=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='failed'
                    """,
                    (int(row["id"]), int(row["agenda_id"])),
                )
                db.execute(
                    """
                    UPDATE compute_jobs_v1
                    SET status='submitted', failure_reason=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='failed' 
                    """,
                    (int(row["compute_job_id"]), int(row["agenda_id"])),
                )
                requeued += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        return requeued

    def quarantine_restarted_running(self) -> int:
        """A lost synchronous CLI process has unknown remote usage."""
        if not db._use_pg():  # noqa: SLF001
            raise ColabCLIError("durable Colab recovery requires PostgreSQL")
        try:
            rows = db.fetchall(
                """
                SELECT cwr.id, cwr.agenda_id, cwr.compute_job_id
                FROM colab_work_requests_v1 AS cwr
                JOIN compute_jobs_v1 AS cj ON cj.id=cwr.compute_job_id
                WHERE cwr.status='running'
                  AND cj.status IN ('running', 'collecting')
                FOR UPDATE OF cwr, cj
                """
            )
            for row in rows:
                db.execute(
                    """
                    UPDATE colab_work_requests_v1
                    SET status='failed', completed_at=CURRENT_TIMESTAMP,
                        failure_reason='controller_lost',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='running'
                    """,
                    (int(row["id"]), int(row["agenda_id"])),
                )
                db.execute(
                    """
                    UPDATE compute_jobs_v1
                    SET failure_reason='controller_lost',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=?
                      AND status IN ('running', 'collecting')
                    """,
                    (int(row["compute_job_id"]), int(row["agenda_id"])),
                )
            db.commit()
            return len(rows)
        except Exception:
            db.rollback()
            raise

    def reconcile_on_startup(self) -> dict[str, int]:
        """Recover safe bind gaps and quarantine uncertain remote work."""
        running = self.quarantine_restarted_running()
        try:
            rebound = db.execute(
                """
                UPDATE colab_work_requests_v1 AS cwr
                SET compute_job_id=cj.id, status='queued',
                    updated_at=CURRENT_TIMESTAMP
                FROM compute_jobs_v1 AS cj
                WHERE cwr.status='admitting'
                  AND cj.agenda_id=cwr.agenda_id
                  AND cj.idempotency_key=cwr.idempotency_key
                  AND cj.backend_kind='colab_gpu'
                  AND cj.backend_job_id=(
                      'colab-work-request:' || CAST(cwr.id AS TEXT)
                  )
                  AND cj.status='submitted'
                """
            )
            uncertain = db.execute(
                """
                UPDATE colab_work_requests_v1 AS cwr
                SET compute_job_id=cj.id, status='manual_reconciliation',
                    failure_reason='compute_submission_or_usage_unknown',
                    updated_at=CURRENT_TIMESTAMP
                FROM compute_jobs_v1 AS cj
                WHERE cwr.status IN ('admitting', 'queued')
                  AND cj.agenda_id=cwr.agenda_id
                  AND cj.idempotency_key=cwr.idempotency_key
                  AND cj.backend_kind='colab_gpu'
                  AND cj.status IN ('submission_unknown', 'usage_unknown')
                """
            )
            db.commit()
            return {
                "running_quarantined": running,
                "admission_rebound": int(
                    getattr(rebound, "rowcount", 0) or 0
                ),
                "uncertain_quarantined": int(
                    getattr(uncertain, "rowcount", 0) or 0
                ),
            }
        except Exception:
            db.rollback()
            raise

    def quarantine_claim(
        self,
        request_id: int,
        *,
        worker_id: str,
        reason: str,
    ) -> None:
        """Fail closed when a claimed synchronous session loses control."""
        try:
            row = db.fetchone(
                """
                SELECT agenda_id, compute_job_id
                FROM colab_work_requests_v1
                WHERE id=? AND status='running' AND worker_id=?
                FOR UPDATE
                """,
                (int(request_id), worker_id),
            )
            if not row:
                raise ColabCLIError("Colab worker no longer owns the claim")
            db.execute(
                """
                UPDATE colab_work_requests_v1
                SET status='failed', failure_reason=?,
                    completed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                """,
                (reason, int(request_id), int(row["agenda_id"])),
            )
            db.execute(
                """
                UPDATE compute_jobs_v1
                SET failure_reason=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                """,
                (
                    reason,
                    int(row["compute_job_id"]),
                    int(row["agenda_id"]),
                ),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise


class DurableColabTransport:
    """ComputeBackend transport whose state is the durable Colab work row."""

    backend_kind = "colab_gpu"

    def __init__(
        self,
        *,
        executor: ColabCLIExecutor,
        accounts: Sequence[ColabAccount],
    ):
        self.executor = executor
        self.accounts = tuple(accounts)

    def capability(self) -> BackendCapability:
        return BackendCapability(
            backend_kind=self.backend_kind,
            available=bool(self.accounts),
            gpu_count=len(self.accounts),
            accelerator_names=(self.executor.config.gpu_type,),
            detail={
                "transport": "durable_colab_work_requests_v1",
                "claim_before_session": True,
            },
        )

    def _row(self, backend_job_id: str) -> dict:
        row = db.fetchone(
            "SELECT * FROM colab_work_requests_v1 WHERE id=?",
            (_request_id(backend_job_id),),
        )
        if not row:
            raise ColabCLIError("Colab work request was not found")
        return row

    def submit(self, request: ComputeSubmission) -> ComputeJob:
        request_id = _request_ref(request.command_ref)
        row = self._row(request.command_ref)
        if (
            int(row.get("agenda_id") or 0) != request.agenda_id
            or int(row.get("idea_id") or 0) != request.idea_id
            or int(row.get("resource_grant_id") or 0)
            != request.resource_grant_id
            or str(row.get("stage") or "") != request.stage
            or str(row.get("idempotency_key") or "") != request.idempotency_key
            or str(row.get("status") or "") != "admitting"
        ):
            raise ColabCLIError("Colab durable request scope mismatch")
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=f"{_BACKEND_PREFIX}{request_id}",
            idempotency_key=request.idempotency_key,
            status="submitted",
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

    def status(self, backend_job_id: str) -> ComputeJob:
        row = self._row(backend_job_id)
        state = str(row.get("status") or "")
        status_map = {
            "admitting": "submitted",
            "queued": "submitted",
            "running": "running",
            "succeeded": "succeeded",
            "failed": "failed",
            "timed_out": "timed_out",
            "cancelled": "cancelled",
        }
        status = status_map.get(state)
        if status is None:
            raise ColabCLIError(f"Colab work requires reconciliation:{state}")
        reason = str(row.get("failure_reason") or "").strip() or None
        if status == "failed" and reason is None:
            reason = "colab_execution_failed"
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=backend_job_id,
            idempotency_key=str(row["idempotency_key"]),
            status=status,
            heartbeat_at=str(
                row.get("completed_at")
                or row.get("started_at")
                or row.get("updated_at")
                or ""
            )
            or None,
            failure_reason=reason if status == "failed" else None,
        )

    def heartbeat(self, backend_job_id: str) -> ComputeJob:
        return self.status(backend_job_id)

    def cancel(self, backend_job_id: str) -> ComputeJob:
        row = self._row(backend_job_id)
        if str(row.get("status") or "") != "queued":
            raise ColabCLIError("running Colab cancellation requires reconciliation")
        changed = db.execute(
            """
            UPDATE colab_work_requests_v1
            SET status='cancelled', completed_at=CURRENT_TIMESTAMP,
                failure_reason='cancelled_before_worker_claim',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=? AND status='queued'
            """,
            (int(row["id"]), int(row["agenda_id"])),
        )
        if int(getattr(changed, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise ColabCLIError("Colab cancellation race")
        attempt = db.fetchone(
            """
            SELECT gpu_attempt_reservation_id
            FROM compute_jobs_v1 WHERE id=?
            """,
            (int(row.get("compute_job_id") or 0),),
        ) or {}
        db.commit()
        reservation_id = int(attempt.get("gpu_attempt_reservation_id") or 0)
        if reservation_id > 0:
            GrantGPUUsageControl().release_unstarted(
                reservation_id,
                reason_code="attempt_cancelled_before_start",
            )
        return self.status(backend_job_id)

    def collect_artifacts(
        self,
        backend_job_id: str,
        requirements: tuple[str, ...],
    ) -> ArtifactCollection:
        row = self._row(backend_job_id)
        manifest = _load_object(
            row.get("artifact_manifest_json"),
            label="Colab artifact manifest",
        )
        artifact_map = _load_object(
            row.get("artifact_map_json"),
            label="Colab artifact map",
        )
        files = {
            str(item.get("path") or "")
            for item in manifest.get("files", [])
            if isinstance(item, dict)
        }
        logical = {
            str(name): str(relative)
            for name, relative in artifact_map.items()
            if str(relative) in files
        }
        combined = dict(manifest)
        combined["artifacts"] = [
            {"name": name, "path": relative}
            for name, relative in sorted(logical.items())
        ]
        for name, relative in logical.items():
            combined[name] = {"path": relative}
        missing = tuple(sorted(set(requirements) - set(logical)))
        return ArtifactCollection(
            manifest=combined,
            complete=(
                str(row.get("status") or "") == "succeeded"
                and bool(logical)
                and not missing
            ),
            missing_requirements=missing,
        )

    def usage(self, backend_job_id: str) -> UsageAccounting:
        row = self._row(backend_job_id)
        compute_job_id = int(row.get("compute_job_id") or 0)
        if compute_job_id <= 0:
            raise ColabCLIError("Colab durable compute binding is missing")
        try:
            measured = GrantGPUUsageControl().usage_for_compute_job(
                compute_job_id
            )
        except Exception as exc:
            raise ColabCLIError(str(exc)) from exc
        return UsageAccounting(
            wall_seconds=float(measured["wall_seconds"]),
            gpu_hours=float(measured["gpu_hours"]),
            cpu_core_hours=0.0,
            backend_report={
                "source": "experiment_attempt_gpu_reservations_v1",
                "attempt_reservation_id": int(
                    measured["attempt_reservation_id"]
                ),
                "executor_reported_wall_seconds": float(
                    row.get("wall_seconds") or 0
                ),
                "account_ref": row.get("account_ref"),
                "session_ref": row.get("session_ref"),
                "attempt_count": int(row.get("attempt_count") or 0),
            },
        )


def load_colab_accounts(manifest_ref: str) -> tuple[ColabAccount, ...]:
    match = _ENV_REF.fullmatch(str(manifest_ref or ""))
    if not match:
        raise ColabCLIError("Colab accounts manifest must be an env:NAME reference")
    manifest_path = os.environ.get(match.group(1), "").strip()
    if not manifest_path:
        raise ColabCLIError("Colab accounts manifest reference is unresolved")
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ColabCLIError("Colab accounts manifest path is not a file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabCLIError("Colab accounts manifest is unreadable") from exc
    if not isinstance(payload, list) or not payload:
        raise ColabCLIError("Colab accounts manifest must be a non-empty array")
    accounts = tuple(
        ColabAccount(
            account_ref=str(item.get("account_ref") or ""),
            credential_ref=str(item.get("credential_ref") or ""),
            isolated_home=str(item.get("isolated_home") or ""),
            oauth_store=str(item.get("oauth_store") or ""),
            session_namespace=str(item.get("session_namespace") or ""),
            quota_gpu_hours=float(item.get("quota_gpu_hours") or 0),
        )
        for item in payload
        if isinstance(item, dict)
    )
    if len(accounts) != len(payload):
        raise ColabCLIError("Colab accounts manifest entries must be objects")
    for account in accounts:
        account.validate()
        isolated_home = Path(account.isolated_home)
        oauth_store = Path(account.oauth_store)
        if not isolated_home.is_absolute() or not oauth_store.is_absolute():
            raise ColabCLIError("Colab HOME and OAuth store must be absolute")
        try:
            oauth_store.resolve().relative_to(isolated_home.resolve())
        except ValueError as exc:
            raise ColabCLIError(
                "Colab OAuth store must be below its isolated HOME"
            ) from exc
        if oauth_store.resolve() == isolated_home.resolve():
            raise ColabCLIError(
                "Colab OAuth store requires a dedicated child path"
            )
    return accounts


def pre_materialized_secret_check(account: ColabAccount) -> None:
    """Validate that an external secret manager populated this account only."""
    match = _ENV_REF.fullmatch(account.credential_ref)
    if not match:
        raise ColabCLIError("Colab credential_ref must be an env:NAME reference")
    credential_path = os.environ.get(match.group(1), "").strip()
    if not credential_path:
        raise ColabCLIError("Colab credential reference is unresolved")
    credential = Path(credential_path).resolve()
    isolated_home = Path(account.isolated_home).resolve()
    oauth_store = Path(account.oauth_store).resolve()
    if not credential.exists() or not oauth_store.exists():
        raise ColabCLIError("Colab isolated credential material is not pre-provisioned")
    if credential != oauth_store and isolated_home not in credential.parents:
        raise ColabCLIError("Colab credential material is outside isolated HOME")


def build_transport(
    *,
    binary: str,
    accounts_manifest_ref: str,
    allowed_code_root: str,
    allowed_artifact_root: str,
) -> DurableColabTransport:
    accounts = load_colab_accounts(accounts_manifest_ref)
    executor = ColabCLIExecutor(
        ColabCLIConfig(
            binary=binary,
            allowed_code_root=allowed_code_root,
            allowed_artifact_root=allowed_artifact_root,
        ),
        accounts,
        secret_materializer=pre_materialized_secret_check,
    )
    return DurableColabTransport(executor=executor, accounts=accounts)


def execution_request_from_row(row: Mapping[str, Any]) -> ColabExecutionRequest:
    artifact_map = _load_object(
        row.get("artifact_map_json"),
        label="Colab artifact map",
    )
    return ColabExecutionRequest(
        agenda_id=int(row["agenda_id"]),
        idea_id=int(row["idea_id"]),
        stage=str(row["stage"]),
        resource_grant_id=int(row["resource_grant_id"]),
        idempotency_key=str(row["idempotency_key"]),
        code_dir=str(row["code_dir"]),
        command_tokens=tuple(
            str(value)
            for value in _load_list(
                row.get("command_tokens_json"),
                label="Colab command tokens",
            )
        ),
        environment={
            str(key): str(value)
            for key, value in _load_object(
                row.get("environment_json"),
                label="Colab environment",
            ).items()
        },
        timeout_seconds=int(row["timeout_seconds"]),
        artifact_paths=tuple(str(value) for value in artifact_map.values()),
        artifact_output_dir=str(row["artifact_output_dir"]),
    )


def grant_from_row(row: Mapping[str, Any]) -> ResourceGrant:
    return ResourceGrant(
        agenda_id=int(row["agenda_id"]),
        idea_id=int(row["idea_id"]),
        decision_packet_id=int(row["decision_packet_id"]),
        stage=str(row["stage"]),
        token_cap=int(row.get("token_cap") or 0),
        gpu_class=str(row.get("gpu_class") or "none"),
        max_gpu_hours=float(row.get("max_gpu_hours") or 0),
        backend_allowlist=[
            str(value)
            for value in _load_list(
                row.get("backend_allowlist_json"),
                label="ResourceGrant backend allowlist",
            )
        ],
        artifact_requirements=[
            str(value)
            for value in _load_list(
                row.get("artifact_requirements_json"),
                label="ResourceGrant artifact requirements",
            )
        ],
        expires_at=str(row["expires_at"]),
        grant_reason=str(row["grant_reason"]),
        idempotency_key=str(row["grant_idempotency_key"]),
        status=str(row["grant_status"]),
        grant_id=int(row["resource_grant_id"]),
        reservation_id=int(row["reservation_id"]),
        preflight_result_id=(
            int(row["preflight_result_id"])
            if row.get("preflight_result_id")
            else None
        ),
    )
