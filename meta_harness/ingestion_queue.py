"""Durable, ResourceGrant-scoped ingestion queue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from db import database as db
from meta_harness.scoped_llm import ScopedLLMError


_TERMINAL = {"succeeded", "failed", "manual_reconciliation", "cancelled"}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _paper_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ScopedLLMError("ingestion paper_ids are invalid JSON") from exc
    if not isinstance(value, (list, tuple)):
        raise ScopedLLMError("ingestion paper_ids must be an array")
    normalized = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not normalized:
        raise ScopedLLMError("ingestion job requires at least one paper_id")
    if len(normalized) > 100:
        raise ScopedLLMError("ingestion job exceeds the 100-paper hard limit")
    return normalized


@dataclass(frozen=True)
class ScopedIngestionRequest:
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    stage: str
    idempotency_key: str
    paper_ids: tuple[str, ...]
    max_attempts: int = 3

    def validate(self) -> None:
        if min(self.agenda_id, self.idea_id, self.resource_grant_id) <= 0:
            raise ScopedLLMError("ingestion scope ids must be positive")
        if not self.stage.strip() or not self.idempotency_key.strip():
            raise ScopedLLMError("ingestion stage and idempotency key are required")
        _paper_ids(self.paper_ids)
        if self.max_attempts <= 0 or self.max_attempts > 10:
            raise ScopedLLMError("ingestion max_attempts must be within 1..10")


class ScopedIngestionRepository:
    def enqueue(self, request: ScopedIngestionRequest) -> int:
        request.validate()
        if not db._use_pg():  # noqa: SLF001
            raise ScopedLLMError("durable scoped ingestion requires PostgreSQL")
        papers = _paper_ids(request.paper_ids)
        try:
            existing = db.fetchone(
                """
                SELECT * FROM scoped_ingestion_jobs_v1
                WHERE agenda_id=? AND idempotency_key=? FOR UPDATE
                """,
                (request.agenda_id, request.idempotency_key),
            )
            if existing:
                expected = {
                    "idea_id": request.idea_id,
                    "resource_grant_id": request.resource_grant_id,
                    "stage": request.stage,
                    "paper_ids_json": _dump(list(papers)),
                    "max_attempts": request.max_attempts,
                }
                mismatch = [
                    key
                    for key, value in expected.items()
                    if str(existing.get(key)) != str(value)
                ]
                if mismatch:
                    raise ScopedLLMError(
                        "ingestion idempotency key reused with different request:"
                        + ",".join(sorted(mismatch))
                    )
                db.commit()
                return int(existing["id"])
            grant = db.fetchone(
                """
                SELECT agenda_id, idea_id, stage, status,
                       backend_allowlist_json
                FROM resource_grants
                WHERE id=? AND expires_at > CURRENT_TIMESTAMP
                FOR UPDATE
                """,
                (request.resource_grant_id,),
            )
            allowlist = set(
                json.loads((grant or {}).get("backend_allowlist_json") or "[]")
            )
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != request.agenda_id
                or int(grant.get("idea_id") or 0) != request.idea_id
                or str(grant.get("stage") or "") != request.stage
                or str(grant.get("status") or "") != "active"
                or "llm" not in allowlist
            ):
                raise ScopedLLMError(
                    "active LLM ResourceGrant does not match ingestion scope"
                )
            paper_count = db.fetchone(
                "SELECT COUNT(*) AS count FROM papers WHERE id = ANY(?)",
                (list(papers),),
            )
            if int((paper_count or {}).get("count") or 0) != len(papers):
                raise ScopedLLMError(
                    "all scoped ingestion paper_ids must already exist"
                )
            job_id = db.insert_returning_id(
                """
                INSERT INTO scoped_ingestion_jobs_v1
                    (agenda_id, idea_id, resource_grant_id, stage,
                     idempotency_key, paper_ids_json, status, max_attempts)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
                RETURNING id
                """,
                (
                    request.agenda_id,
                    request.idea_id,
                    request.resource_grant_id,
                    request.stage,
                    request.idempotency_key,
                    _dump(list(papers)),
                    request.max_attempts,
                ),
            )
            db.commit()
            return int(job_id)
        except Exception:
            db.rollback()
            raise

    def recover_expired_leases(self, *, agenda_id: int) -> dict[str, int]:
        if int(agenda_id or 0) <= 0:
            raise ScopedLLMError(
                "ingestion recovery requires an explicit agenda scope"
            )
        if not db._use_pg():  # noqa: SLF001
            raise ScopedLLMError("durable scoped ingestion requires PostgreSQL")
        try:
            retryable = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='retryable', lease_owner=NULL, lease_expires_at=NULL,
                    failure_reason='worker_lease_expired_checkpoint_resume',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status='running' AND lease_expires_at <= CURRENT_TIMESTAMP
                  AND attempt_count < max_attempts
                  AND agenda_id=?
                """,
                (int(agenda_id),),
            )
            manual = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='manual_reconciliation', lease_owner=NULL,
                    lease_expires_at=NULL,
                    failure_reason='worker_lease_expired_attempts_exhausted',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status='running' AND lease_expires_at <= CURRENT_TIMESTAMP
                  AND attempt_count >= max_attempts
                  AND agenda_id=?
                """,
                (int(agenda_id),),
            )
            db.commit()
            return {
                "retryable": int(getattr(retryable, "rowcount", 0) or 0),
                "manual_reconciliation": int(
                    getattr(manual, "rowcount", 0) or 0
                ),
            }
        except Exception:
            db.rollback()
            raise

    def claim_next(self, *, worker_id: str, lease_seconds: int) -> dict | None:
        if not worker_id.strip() or lease_seconds <= 0:
            raise ScopedLLMError("ingestion worker lease metadata is invalid")
        if not db._use_pg():  # noqa: SLF001
            raise ScopedLLMError("durable scoped ingestion requires PostgreSQL")
        try:
            row = db.fetchone(
                """
                SELECT sij.*, rg.token_cap, rg.backend_allowlist_json
                FROM scoped_ingestion_jobs_v1 AS sij
                JOIN resource_grants AS rg ON rg.id=sij.resource_grant_id
                JOIN research_agendas AS ra ON ra.id=sij.agenda_id
                WHERE sij.status IN ('queued', 'retryable')
                  AND sij.attempt_count < sij.max_attempts
                  AND rg.agenda_id=sij.agenda_id
                  AND rg.idea_id=sij.idea_id
                  AND rg.stage=sij.stage
                  AND rg.status='active'
                  AND rg.expires_at > CURRENT_TIMESTAMP
                  AND ra.is_active=1
                  AND ra.status='active'
                ORDER BY sij.created_at, sij.id
                LIMIT 1 FOR UPDATE OF sij, ra SKIP LOCKED
                """
            )
            if not row:
                db.commit()
                return None
            if "llm" not in set(
                json.loads(row.get("backend_allowlist_json") or "[]")
            ):
                raise ScopedLLMError(
                    "persisted ingestion ResourceGrant no longer allows LLM"
                )
            lease_expires = (
                datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
            ).isoformat()
            changed = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='running', attempt_count=attempt_count+1,
                    lease_owner=?, lease_expires_at=?,
                    started_at=COALESCE(started_at, CURRENT_TIMESTAMP),
                    failure_reason=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=?
                  AND status IN ('queued', 'retryable')
                """,
                (
                    worker_id,
                    lease_expires,
                    int(row["id"]),
                    int(row["agenda_id"]),
                ),
            )
            if int(getattr(changed, "rowcount", 0) or 0) != 1:
                raise ScopedLLMError("ingestion job claim race")
            db.commit()
            row["status"] = "running"
            row["attempt_count"] = int(row.get("attempt_count") or 0) + 1
            row["lease_owner"] = worker_id
            return row
        except Exception:
            db.rollback()
            raise

    def complete(
        self,
        job_id: int,
        *,
        agenda_id: int,
        worker_id: str,
        results: Sequence[dict],
    ) -> None:
        self._finish(
            job_id,
            agenda_id=agenda_id,
            worker_id=worker_id,
            status="succeeded",
            result={"papers": list(results)},
            failure_reason=None,
        )

    def renew_lease(
        self,
        job_id: int,
        *,
        agenda_id: int,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        if lease_seconds <= 0:
            raise ScopedLLMError("ingestion lease duration must be positive")
        lease_expires = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        changed = db.execute(
            """
            UPDATE scoped_ingestion_jobs_v1
            SET lease_expires_at=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=? AND status='running'
              AND lease_owner=?
            """,
            (lease_expires, int(job_id), int(agenda_id), worker_id),
        )
        if int(getattr(changed, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise ScopedLLMError("ingestion worker lease was lost")
        db.commit()

    def fail(
        self,
        job_id: int,
        *,
        agenda_id: int,
        worker_id: str,
        reason: str,
        retryable: bool,
        partial_results: Sequence[dict],
    ) -> str:
        row = db.fetchone(
            """
            SELECT attempt_count, max_attempts
            FROM scoped_ingestion_jobs_v1
            WHERE id=? AND agenda_id=? AND status='running'
              AND lease_owner=?
            """,
            (int(job_id), int(agenda_id), worker_id),
        )
        if not row:
            raise ScopedLLMError("ingestion failure does not own the active lease")
        target = (
            "retryable"
            if retryable
            and int(row.get("attempt_count") or 0)
            < int(row.get("max_attempts") or 0)
            else "failed"
        )
        self._finish(
            job_id,
            agenda_id=agenda_id,
            worker_id=worker_id,
            status=target,
            result={"papers": list(partial_results)},
            failure_reason=reason,
        )
        return target

    def _finish(
        self,
        job_id: int,
        *,
        agenda_id: int,
        worker_id: str,
        status: str,
        result: dict,
        failure_reason: str | None,
    ) -> None:
        if status not in _TERMINAL | {"retryable"}:
            raise ScopedLLMError("invalid ingestion terminal state")
        try:
            changed = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status=?, result_json=?, failure_reason=?,
                    lease_owner=NULL, lease_expires_at=NULL,
                    completed_at=CASE
                        WHEN ?='retryable' THEN NULL
                        ELSE CURRENT_TIMESTAMP
                    END,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                  AND lease_owner=?
                """,
                (
                    status,
                    _dump(result),
                    failure_reason,
                    status,
                    int(job_id),
                    int(agenda_id),
                    worker_id,
                ),
            )
            if int(getattr(changed, "rowcount", 0) or 0) != 1:
                raise ScopedLLMError("ingestion completion lost its worker lease")
            db.commit()
        except Exception:
            db.rollback()
            raise

    def count_by_status(self) -> dict[str, int]:
        rows = db.fetchall(
            """
            SELECT status, COUNT(*) AS count
            FROM scoped_ingestion_jobs_v1
            GROUP BY status
            ORDER BY status
            """
        )
        return {
            str(row["status"]): int(row.get("count") or 0)
            for row in rows
        }
