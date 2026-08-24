"""Durable, ResourceGrant-scoped ingestion queue."""

from __future__ import annotations

import json
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from db import database as db
from meta_harness.grant_stages import (
    ResourceGrantStageError,
    require_ingestion_grant_stage,
)
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


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {"unparsed_result": value}
        if isinstance(loaded, dict):
            return loaded
    return {}


def _expect_one(cursor: Any, *, operation: str) -> None:
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        raise ScopedLLMError(f"{operation} did not update exactly one row")


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
        if any(
            type(value) is not int
            for value in (
                self.agenda_id,
                self.idea_id,
                self.resource_grant_id,
                self.max_attempts,
            )
        ):
            raise ScopedLLMError(
                "ingestion scope ids and max_attempts must be integers"
            )
        if min(self.agenda_id, self.idea_id, self.resource_grant_id) <= 0:
            raise ScopedLLMError("ingestion scope ids must be positive")
        if not self.stage.strip() or not self.idempotency_key.strip():
            raise ScopedLLMError("ingestion stage and idempotency key are required")
        try:
            require_ingestion_grant_stage(self.stage)
        except ResourceGrantStageError as exc:
            raise ScopedLLMError(str(exc)) from exc
        _paper_ids(self.paper_ids)
        if self.max_attempts <= 0 or self.max_attempts > 10:
            raise ScopedLLMError("ingestion max_attempts must be within 1..10")


@dataclass(frozen=True)
class ScopedIngestionReconciliationRequest:
    """Exact operator authority to replace one historical failed job.

    This is deliberately verbose.  The caller must echo every durable scope
    field and the complete paper set, so an operator cannot turn a single-row
    reconciliation into a backlog reset by omitting selectors.
    """

    source_job_id: int
    agenda_id: int
    idea_id: int
    source_resource_grant_id: int
    replacement_resource_grant_id: int
    stage: str
    paper_ids: tuple[str, ...]
    actor: str
    reason: str
    idempotency_key: str
    max_attempts: int = 3

    def validate(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.source_job_id,
                self.agenda_id,
                self.idea_id,
                self.source_resource_grant_id,
                self.replacement_resource_grant_id,
                self.max_attempts,
            )
        ):
            raise ScopedLLMError(
                "ingestion reconciliation ids and max_attempts must be integers"
            )
        if min(
            self.source_job_id,
            self.agenda_id,
            self.idea_id,
            self.source_resource_grant_id,
            self.replacement_resource_grant_id,
        ) <= 0:
            raise ScopedLLMError("ingestion reconciliation ids must be positive")
        if self.source_resource_grant_id == self.replacement_resource_grant_id:
            raise ScopedLLMError(
                "ingestion reconciliation requires a replacement ResourceGrant"
            )
        if not self.stage.strip():
            raise ScopedLLMError("ingestion reconciliation stage is required")
        try:
            require_ingestion_grant_stage(self.stage)
        except ResourceGrantStageError as exc:
            raise ScopedLLMError(str(exc)) from exc
        _paper_ids(self.paper_ids)
        if not self.actor.strip() or not self.reason.strip():
            raise ScopedLLMError(
                "ingestion reconciliation actor and reason are required"
            )
        if not self.idempotency_key.strip():
            raise ScopedLLMError(
                "ingestion reconciliation idempotency_key is required"
            )
        if self.max_attempts <= 0 or self.max_attempts > 10:
            raise ScopedLLMError(
                "ingestion reconciliation max_attempts must be within 1..10"
            )


@dataclass(frozen=True)
class ScopedIngestionUsageDispositionRequest:
    """Exact operator judgement for one ambiguous provider reservation."""

    job_id: int
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    stage: str
    paper_ids: tuple[str, ...]
    usage_reservation_id: int
    operation: str
    usage_idempotency_key: str
    expected_token_reserved: int
    disposition: str
    tokens_used: int | None
    cost_usd: float | None
    actor: str
    reason: str
    evidence_ref: str
    operator_request_id: str
    resume: bool = False

    def validate(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.job_id,
                self.agenda_id,
                self.idea_id,
                self.resource_grant_id,
                self.usage_reservation_id,
                self.expected_token_reserved,
            )
        ):
            raise ScopedLLMError("usage disposition ids and token cap must be integers")
        if min(
            self.job_id,
            self.agenda_id,
            self.idea_id,
            self.resource_grant_id,
            self.usage_reservation_id,
            self.expected_token_reserved,
        ) <= 0:
            raise ScopedLLMError("usage disposition ids and token cap must be positive")
        try:
            require_ingestion_grant_stage(self.stage)
        except ResourceGrantStageError as exc:
            raise ScopedLLMError(str(exc)) from exc
        _paper_ids(self.paper_ids)
        for label, value in (
            ("operation", self.operation),
            ("usage_idempotency_key", self.usage_idempotency_key),
            ("actor", self.actor),
            ("reason", self.reason),
            ("evidence_ref", self.evidence_ref),
            ("operator_request_id", self.operator_request_id),
        ):
            if not str(value or "").strip():
                raise ScopedLLMError(f"usage disposition {label} is required")
        if self.disposition not in {
            "settle_measured",
            "release_confirmed_unbilled",
        }:
            raise ScopedLLMError("unsupported ingestion usage disposition")
        if self.disposition == "settle_measured":
            if self.tokens_used is None or type(self.tokens_used) is not int:
                raise ScopedLLMError("measured tokens_used must be an integer")
            if not 0 <= self.tokens_used <= self.expected_token_reserved:
                raise ScopedLLMError(
                    "measured usage must be within the exact reservation cap"
                )
            if self.cost_usd is not None and (
                isinstance(self.cost_usd, bool)
                or not isinstance(self.cost_usd, (int, float))
                or not math.isfinite(float(self.cost_usd))
                or float(self.cost_usd) < 0
            ):
                raise ScopedLLMError(
                    "measured usage cost must be a finite non-negative number"
                )
        elif (
            (self.tokens_used is not None and type(self.tokens_used) is not int)
            or self.tokens_used not in {None, 0}
            or (
                self.cost_usd is not None
                and (
                    isinstance(self.cost_usd, bool)
                    or not isinstance(self.cost_usd, (int, float))
                    or not math.isfinite(float(self.cost_usd))
                )
            )
            or self.cost_usd not in {None, 0, 0.0}
        ):
            raise ScopedLLMError(
                "confirmed-unbilled disposition cannot record usage or cost"
            )
        if not isinstance(self.resume, bool):
            raise ScopedLLMError("usage disposition resume must be a boolean")


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
                SELECT rg.agenda_id, rg.idea_id, rg.stage, rg.status,
                       rg.backend_allowlist_json, rg.max_gpu_hours,
                       arl.status AS ledger_status
                FROM resource_grants AS rg
                JOIN agenda_resource_ledger AS arl ON arl.id=rg.reservation_id
                WHERE rg.id=? AND rg.expires_at > CURRENT_TIMESTAMP
                FOR UPDATE OF rg, arl
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
                or str(grant.get("ledger_status") or "") != "reserved"
                or float(grant.get("max_gpu_hours") or 0.0) != 0.0
                or allowlist != {"llm"}
            ):
                raise ScopedLLMError(
                    "active LLM ResourceGrant does not match ingestion scope"
                )
            grant_job = db.fetchone(
                """
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? FOR UPDATE
                """,
                (request.resource_grant_id,),
            )
            if grant_job:
                raise ScopedLLMError(
                    "ResourceGrant already owns a scoped ingestion job"
                )
            paper_count = db.fetchone(
                """
                SELECT COUNT(*) AS count,
                       COALESCE(SUM(
                           CASE WHEN status='reasoned'
                                  OR processing_stage='reasoned'
                                THEN 1 ELSE 0 END
                       ), 0) AS reasoned_count
                FROM papers WHERE id = ANY(?)
                """,
                (list(papers),),
            )
            if int((paper_count or {}).get("count") or 0) != len(papers):
                raise ScopedLLMError(
                    "all scoped ingestion paper_ids must already exist"
                )
            if int((paper_count or {}).get("reasoned_count") or 0):
                raise ScopedLLMError(
                    "fresh scoped ingestion cannot enqueue already reasoned papers"
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

    def reconcile_failed_job(
        self,
        request: ScopedIngestionReconciliationRequest,
    ) -> dict[str, Any]:
        """Create one exact replacement job and retain an audit on both rows.

        The source failure remains terminal.  Its old grant must already have
        been closed through the normal grant expiry/revocation path; this
        method cannot silently reinterpret active authority.  A fresh active
        grant is mandatory and may not already own another ingestion job.

        The source grant must have no open child reservation.  Ambiguous
        provider delivery can only be resolved by the explicit usage
        disposition operator; replacement never guesses or silently releases
        historical spend.  There is no status reset and no unscoped update.
        """

        request.validate()
        if not db._use_pg():  # noqa: SLF001
            raise ScopedLLMError("durable scoped ingestion requires PostgreSQL")
        papers = _paper_ids(request.paper_ids)
        try:
            source = db.fetchone(
                """
                SELECT * FROM scoped_ingestion_jobs_v1
                WHERE id=? AND agenda_id=? FOR UPDATE
                """,
                (request.source_job_id, request.agenda_id),
            )
            if not source:
                raise ScopedLLMError("source ingestion job was not found")
            expected_source = {
                "idea_id": request.idea_id,
                "resource_grant_id": request.source_resource_grant_id,
                "stage": request.stage,
                "paper_ids_json": _dump(list(papers)),
            }
            mismatch = [
                key
                for key, value in expected_source.items()
                if str(source.get(key)) != str(value)
            ]
            if mismatch:
                raise ScopedLLMError(
                    "source ingestion reconciliation scope mismatch:"
                    + ",".join(sorted(mismatch))
                )

            source_result = _mapping(source.get("result_json"))
            prior = source_result.get("reconciliation")
            expected_audit = {
                "version": "scoped-ingestion-reconciliation-v1",
                "action": "replace_failed_job",
                "source_job_id": request.source_job_id,
                "source_resource_grant_id": request.source_resource_grant_id,
                "replacement_resource_grant_id": (
                    request.replacement_resource_grant_id
                ),
                "agenda_id": request.agenda_id,
                "idea_id": request.idea_id,
                "stage": request.stage,
                "paper_ids": list(papers),
                "actor": request.actor.strip(),
                "reason": request.reason.strip(),
                "idempotency_key": request.idempotency_key.strip(),
                "max_attempts": request.max_attempts,
            }
            if isinstance(prior, dict):
                comparable = {
                    key: prior.get(key) for key in expected_audit
                }
                if comparable != expected_audit:
                    raise ScopedLLMError(
                        "source ingestion job was already reconciled differently"
                    )
                replacement_id = int(prior.get("replacement_job_id") or 0)
                replacement = db.fetchone(
                    """
                    SELECT id, agenda_id, idea_id, resource_grant_id, stage,
                           idempotency_key, paper_ids_json, max_attempts
                    FROM scoped_ingestion_jobs_v1
                    WHERE id=? AND agenda_id=?
                    """,
                    (replacement_id, request.agenda_id),
                )
                if (
                    not replacement
                    or int(replacement.get("idea_id") or 0) != request.idea_id
                    or int(replacement.get("resource_grant_id") or 0)
                    != request.replacement_resource_grant_id
                    or str(replacement.get("stage") or "") != request.stage
                    or str(replacement.get("idempotency_key") or "")
                    != request.idempotency_key
                    or str(replacement.get("paper_ids_json") or "")
                    != _dump(list(papers))
                    or int(replacement.get("max_attempts") or 0)
                    != request.max_attempts
                ):
                    raise ScopedLLMError(
                        "persisted ingestion reconciliation audit is inconsistent"
                    )
                db.commit()
                return {
                    "status": "already_reconciled",
                    "source_job_id": request.source_job_id,
                    "replacement_job_id": replacement_id,
                    "released_source_reservations": int(
                        prior.get("released_source_reservations") or 0
                    ),
                }

            if str(source.get("status") or "") not in {
                "failed",
                "manual_reconciliation",
            }:
                raise ScopedLLMError(
                    "only failed or manual_reconciliation jobs may be replaced"
                )

            source_grant = db.fetchone(
                """
                SELECT rg.agenda_id, rg.idea_id, rg.stage, rg.status,
                       arl.status AS ledger_status
                FROM resource_grants AS rg
                JOIN agenda_resource_ledger AS arl ON arl.id=rg.reservation_id
                WHERE rg.id=? FOR UPDATE OF rg, arl
                """,
                (request.source_resource_grant_id,),
            )
            if (
                not source_grant
                or int(source_grant.get("agenda_id") or 0) != request.agenda_id
                or int(source_grant.get("idea_id") or 0) != request.idea_id
                or str(source_grant.get("stage") or "") != request.stage
                or (
                    str(source_grant.get("status") or ""),
                    str(source_grant.get("ledger_status") or ""),
                )
                not in {
                    ("expired", "settled"),
                    ("expired", "released"),
                    ("revoked", "released"),
                    ("consumed", "settled"),
                }
            ):
                raise ScopedLLMError(
                    "source ingestion ResourceGrant is not formally closed"
                )
            source_sibling = db.fetchone(
                """
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? AND id<>? FOR UPDATE
                """,
                (
                    request.source_resource_grant_id,
                    request.source_job_id,
                ),
            )
            if source_sibling:
                raise ScopedLLMError(
                    "source ResourceGrant is not one-to-one with the failed job"
                )

            replacement_grant = db.fetchone(
                """
                SELECT rg.agenda_id, rg.idea_id, rg.stage, rg.status,
                       rg.backend_allowlist_json, rg.max_gpu_hours,
                       arl.status AS ledger_status
                FROM resource_grants AS rg
                JOIN agenda_resource_ledger AS arl ON arl.id=rg.reservation_id
                WHERE rg.id=? AND rg.expires_at > CURRENT_TIMESTAMP
                FOR UPDATE OF rg, arl
                """,
                (request.replacement_resource_grant_id,),
            )
            replacement_allowlist = set(
                json.loads(
                    (replacement_grant or {}).get("backend_allowlist_json")
                    or "[]"
                )
            )
            if (
                not replacement_grant
                or int(replacement_grant.get("agenda_id") or 0)
                != request.agenda_id
                or int(replacement_grant.get("idea_id") or 0)
                != request.idea_id
                or str(replacement_grant.get("stage") or "") != request.stage
                or str(replacement_grant.get("status") or "") != "active"
                or str(replacement_grant.get("ledger_status") or "")
                != "reserved"
                or float(replacement_grant.get("max_gpu_hours") or 0.0) != 0.0
                or replacement_allowlist != {"llm"}
            ):
                raise ScopedLLMError(
                    "replacement ResourceGrant does not match ingestion scope"
                )

            existing_key = db.fetchone(
                """
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE agenda_id=? AND idempotency_key=? FOR UPDATE
                """,
                (request.agenda_id, request.idempotency_key),
            )
            if existing_key:
                raise ScopedLLMError(
                    "replacement ingestion idempotency key already exists"
                )
            sibling = db.fetchone(
                """
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? FOR UPDATE
                """,
                (request.replacement_resource_grant_id,),
            )
            if sibling:
                raise ScopedLLMError(
                    "replacement ResourceGrant already owns an ingestion job"
                )
            paper_count = db.fetchone(
                "SELECT COUNT(*) AS count FROM papers WHERE id = ANY(?)",
                (list(papers),),
            )
            if int((paper_count or {}).get("count") or 0) != len(papers):
                raise ScopedLLMError(
                    "all reconciliation paper_ids must already exist"
                )

            source_open_usage = db.fetchall(
                """
                SELECT id FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND agenda_id=? AND status='reserved'
                ORDER BY id FOR UPDATE
                """,
                (
                    request.source_resource_grant_id,
                    request.agenda_id,
                ),
            )
            if source_open_usage:
                raise ScopedLLMError(
                    "source ingestion grant has ambiguous open usage; "
                    "an exact usage disposition is required before replacement"
                )
            released_count = 0
            recorded_at = datetime.now(timezone.utc).isoformat()
            replacement_origin = {
                **expected_audit,
                "recorded_at": recorded_at,
            }
            replacement_id = db.insert_returning_id(
                """
                INSERT INTO scoped_ingestion_jobs_v1
                    (agenda_id, idea_id, resource_grant_id, stage,
                     idempotency_key, paper_ids_json, status, max_attempts,
                     result_json)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                RETURNING id
                """,
                (
                    request.agenda_id,
                    request.idea_id,
                    request.replacement_resource_grant_id,
                    request.stage,
                    request.idempotency_key,
                    _dump(list(papers)),
                    request.max_attempts,
                    _dump({"recovery_origin": replacement_origin}),
                ),
            )
            source_result["reconciliation"] = {
                **replacement_origin,
                "replacement_job_id": int(replacement_id),
                "released_source_reservations": released_count,
            }
            changed = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET result_json=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=?
                  AND status IN ('failed', 'manual_reconciliation')
                """,
                (
                    _dump(source_result),
                    request.source_job_id,
                    request.agenda_id,
                ),
            )
            _expect_one(changed, operation="audit_ingestion_reconciliation")
            db.commit()
            return {
                "status": "queued",
                "source_job_id": request.source_job_id,
                "replacement_job_id": int(replacement_id),
                "released_source_reservations": released_count,
            }
        except Exception:
            db.rollback()
            raise

    def dispose_open_usage(
        self,
        request: ScopedIngestionUsageDispositionRequest,
    ) -> dict[str, Any]:
        """Irreversibly settle or release one ambiguous ingestion LLM call.

        The operator must echo the complete job and reservation identity plus
        an evidence reference.  This method is deliberately unable to select
        a collection or infer whether a provider delivered the call.
        """

        request.validate()
        use_pg = db._use_pg()  # noqa: SLF001
        lock = " FOR UPDATE" if use_pg else ""
        lock_join = " FOR UPDATE OF rg, arl" if use_pg else ""
        papers = _paper_ids(request.paper_ids)
        audit_request = {
            "schema": "scoped-ingestion-usage-disposition-v1",
            "operator_request_id": request.operator_request_id.strip(),
            "job_id": request.job_id,
            "agenda_id": request.agenda_id,
            "idea_id": request.idea_id,
            "resource_grant_id": request.resource_grant_id,
            "stage": request.stage,
            "paper_ids": list(papers),
            "usage_reservation_id": request.usage_reservation_id,
            "operation": request.operation.strip(),
            "usage_idempotency_key": request.usage_idempotency_key.strip(),
            "expected_token_reserved": request.expected_token_reserved,
            "disposition": request.disposition,
            "tokens_used": (
                int(request.tokens_used)
                if request.disposition == "settle_measured"
                else None
            ),
            "cost_usd": (
                float(request.cost_usd)
                if request.disposition == "settle_measured"
                and request.cost_usd is not None
                else None
            ),
            "actor": request.actor.strip(),
            "reason": request.reason.strip(),
            "evidence_ref": request.evidence_ref.strip(),
            "resume": bool(request.resume),
        }
        try:
            job = db.fetchone(
                f"""
                SELECT * FROM scoped_ingestion_jobs_v1
                WHERE id=? AND agenda_id=?{lock}
                """,
                (request.job_id, request.agenda_id),
            )
            if (
                not job
                or int(job.get("idea_id") or 0) != request.idea_id
                or int(job.get("resource_grant_id") or 0)
                != request.resource_grant_id
                or str(job.get("stage") or "") != request.stage
                or _paper_ids(job.get("paper_ids_json")) != papers
                or str(job.get("status") or "")
                not in {"manual_reconciliation", "failed", "retryable"}
            ):
                raise ScopedLLMError("usage disposition job scope mismatch")
            require_ingestion_grant_stage(str(job.get("stage") or ""))
            grant = db.fetchone(
                f"""
                SELECT rg.*, arl.status AS ledger_status,
                       CASE WHEN rg.expires_at > CURRENT_TIMESTAMP
                            THEN 1 ELSE 0 END AS unexpired
                FROM resource_grants AS rg
                JOIN agenda_resource_ledger AS arl
                  ON arl.id=rg.reservation_id AND arl.agenda_id=rg.agenda_id
                WHERE rg.id=? AND rg.agenda_id=?{lock_join}
                """,
                (request.resource_grant_id, request.agenda_id),
            )
            if (
                not grant
                or int(grant.get("idea_id") or 0) != request.idea_id
                or str(grant.get("stage") or "") != request.stage
                or str(grant.get("status") or "") != "active"
                or str(grant.get("ledger_status") or "") != "reserved"
                or float(grant.get("max_gpu_hours") or 0.0) != 0.0
                or set(json.loads(grant.get("backend_allowlist_json") or "[]"))
                != {"llm"}
            ):
                raise ScopedLLMError(
                    "usage disposition ResourceGrant scope is not open and LLM-only"
                )
            siblings = db.fetchall(
                f"""
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? ORDER BY id{lock}
                """,
                (request.resource_grant_id,),
            )
            if (
                len(siblings) != 1
                or int(siblings[0].get("id") or 0) != request.job_id
            ):
                raise ScopedLLMError(
                    "usage disposition ResourceGrant is not one-to-one with its job"
                )
            gpu_attempts = db.fetchall(
                f"""
                SELECT id FROM experiment_attempt_gpu_reservations_v1
                WHERE resource_grant_id=? ORDER BY id{lock}
                """,
                (request.resource_grant_id,),
            )
            if gpu_attempts:
                raise ScopedLLMError(
                    "usage disposition ingestion grant has GPU attempt reservations"
                )
            usage = db.fetchone(
                f"""
                SELECT * FROM resource_grant_usage_reservations
                WHERE id=? AND agenda_id=? AND resource_grant_id=?{lock}
                """,
                (
                    request.usage_reservation_id,
                    request.agenda_id,
                    request.resource_grant_id,
                ),
            )
            if (
                not usage
                or str(usage.get("operation") or "") != request.operation
                or str(usage.get("idempotency_key") or "")
                != request.usage_idempotency_key
                or int(usage.get("token_reserved") or 0)
                != request.expected_token_reserved
            ):
                raise ScopedLLMError("usage disposition reservation scope mismatch")

            if str(usage.get("status") or "") != "reserved":
                persisted = _mapping(usage.get("release_reason"))
                comparable = {key: persisted.get(key) for key in audit_request}
                expected_status = (
                    "settled"
                    if request.disposition == "settle_measured"
                    else "released"
                )
                if (
                    comparable != audit_request
                    or str(usage.get("status") or "") != expected_status
                    or (
                        request.resume
                        and str(job.get("status") or "") != "retryable"
                    )
                ):
                    raise ScopedLLMError(
                        "usage reservation was already disposed differently"
                    )
                db.commit()
                return {
                    "status": "already_disposed",
                    "usage_status": expected_status,
                    "job_status": str(job.get("status") or ""),
                }

            if str(job.get("status") or "") not in {
                "manual_reconciliation",
                "failed",
            }:
                raise ScopedLLMError(
                    "new usage disposition requires a parked ingestion job"
                )
            if request.resume and str(job.get("status") or "") != "manual_reconciliation":
                raise ScopedLLMError("only manual_reconciliation work may resume")
            terminal_status = (
                "settled"
                if request.disposition == "settle_measured"
                else "released"
            )
            audit = {
                **audit_request,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            changed = db.execute(
                """
                UPDATE resource_grant_usage_reservations
                SET tokens_used=?, cost_usd=?, status=?, release_reason=?,
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND resource_grant_id=?
                  AND operation=? AND idempotency_key=?
                  AND token_reserved=? AND status='reserved'
                """,
                (
                    audit_request["tokens_used"],
                    audit_request["cost_usd"],
                    terminal_status,
                    _dump(audit),
                    request.usage_reservation_id,
                    request.agenda_id,
                    request.resource_grant_id,
                    request.operation,
                    request.usage_idempotency_key,
                    request.expected_token_reserved,
                ),
            )
            _expect_one(changed, operation="dispose_exact_ingestion_usage")

            job_status = str(job.get("status") or "")
            if request.resume:
                if not bool(grant.get("unexpired")):
                    raise ScopedLLMError(
                        "expired ingestion authority cannot resume; close and replace it"
                    )
                if int(job.get("attempt_count") or 0) >= int(
                    job.get("max_attempts") or 0
                ):
                    raise ScopedLLMError("ingestion attempts are exhausted")
                other_open = db.fetchone(
                    f"""
                    SELECT id FROM resource_grant_usage_reservations
                    WHERE resource_grant_id=? AND agenda_id=?
                      AND status='reserved' ORDER BY id LIMIT 1{lock}
                    """,
                    (request.resource_grant_id, request.agenda_id),
                )
                if other_open:
                    raise ScopedLLMError(
                        "another open usage reservation still blocks resume"
                    )
                changed = db.execute(
                    """
                    UPDATE scoped_ingestion_jobs_v1
                    SET status='retryable', lease_owner=NULL,
                        lease_expires_at=NULL, completed_at=NULL,
                        failure_reason='operator_usage_disposition_completed',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=?
                      AND status='manual_reconciliation'
                      AND attempt_count < max_attempts
                    """,
                    (request.job_id, request.agenda_id),
                )
                _expect_one(changed, operation="resume_disposed_ingestion_job")
                job_status = "retryable"
            db.commit()
            return {
                "status": "disposed",
                "usage_status": terminal_status,
                "job_status": job_status,
            }
        except ResourceGrantStageError as exc:
            db.rollback()
            raise ScopedLLMError(str(exc)) from exc
        except Exception:
            db.rollback()
            raise

    def release_dead_worker_claims(self, *, agenda_id: int) -> int:
        """Reclaim jobs held by a worker process that no longer exists.

        The lease is a timeout, which is a guess about whether the worker is
        alive. worker_id carries host:pid, so on this host a claim whose PID
        is gone is provably abandoned -- a stronger signal, available
        immediately. On 2026-08-19 a 30-minute lease outlived the web
        restarts that killed the worker, so a 20-paper batch spent most of a
        38-minute window waiting for leases to expire rather than working.

        The attempt is still charged. A process that dies because of the job
        it is running (rather than because of a restart) must not be able to
        retry forever, and this method cannot tell the two apart; charging the
        attempt keeps the exhaustion path (manual_reconciliation) reachable.
        """
        if int(agenda_id or 0) <= 0:
            raise ScopedLLMError(
                "ingestion recovery requires an explicit agenda scope"
            )
        if not db._use_pg():  # noqa: SLF001
            raise ScopedLLMError("durable scoped ingestion requires PostgreSQL")
        released = 0
        try:
            host = socket.gethostname()
            rows = db.fetchall(
                """
                SELECT id, resource_grant_id, lease_owner,
                       attempt_count, max_attempts
                FROM scoped_ingestion_jobs_v1
                WHERE status='running' AND agenda_id=? AND lease_owner LIKE ?
                FOR UPDATE
                """,
                (int(agenda_id), f"{host}:%"),
            )
            for row in rows:
                parts = str(row.get("lease_owner") or "").split(":")
                if len(parts) < 2 or not parts[1].isdigit():
                    continue
                pid = int(parts[1])
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, 0)
                    continue  # the claiming process is alive; leave it alone
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue  # exists under another user; not ours to reclaim
                grant_jobs = db.fetchall(
                    """
                    SELECT id, status FROM scoped_ingestion_jobs_v1
                    WHERE resource_grant_id=? ORDER BY id FOR UPDATE
                    """,
                    (int(row["resource_grant_id"]),),
                )
                one_to_one = (
                    len(grant_jobs) == 1
                    and int(grant_jobs[0].get("id") or 0) == int(row["id"])
                )
                if not one_to_one:
                    changed = db.execute(
                        """
                        UPDATE scoped_ingestion_jobs_v1
                        SET status='manual_reconciliation', lease_owner=NULL,
                            lease_expires_at=NULL,
                            failure_reason='dead_worker_grant_not_one_to_one',
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=? AND agenda_id=? AND status='running'
                        """,
                        (int(row["id"]), int(agenda_id)),
                    )
                    _expect_one(
                        changed,
                        operation="park_ambiguous_dead_worker_ingestion",
                    )
                    released += 1
                    continue

                # A dead PID proves that no worker can finish the lease.  It
                # does *not* prove whether a provider accepted an already
                # reserved call.  Automatically releasing such a reservation
                # could under-report real spend and then buy the same call
                # again, so ambiguous usage is parked for the exact operator
                # reconciliation path.
                open_usage = db.fetchall(
                    """
                    SELECT id FROM resource_grant_usage_reservations
                    WHERE resource_grant_id=? AND agenda_id=?
                      AND status='reserved'
                    ORDER BY id FOR UPDATE
                    """,
                    (int(row["resource_grant_id"]), int(agenda_id)),
                )
                if open_usage:
                    changed = db.execute(
                        """
                        UPDATE scoped_ingestion_jobs_v1
                        SET status='manual_reconciliation', lease_owner=NULL,
                            lease_expires_at=NULL,
                            failure_reason=
                                'dead_worker_open_usage_reconciliation_required',
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=? AND agenda_id=? AND status='running'
                        """,
                        (int(row["id"]), int(agenda_id)),
                    )
                    _expect_one(
                        changed,
                        operation="park_dead_worker_open_usage",
                    )
                    released += 1
                    continue
                exhausted = int(row.get("attempt_count") or 0) >= int(
                    row.get("max_attempts") or 0
                )
                changed = db.execute(
                    """
                    UPDATE scoped_ingestion_jobs_v1
                    SET status=?, lease_owner=NULL, lease_expires_at=NULL,
                        failure_reason=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='running'
                    """,
                    (
                        "manual_reconciliation" if exhausted else "retryable",
                        "worker_process_gone_attempts_exhausted"
                        if exhausted
                        else "worker_process_gone_checkpoint_resume",
                        int(row["id"]),
                        int(agenda_id),
                    ),
                )
                _expect_one(changed, operation="reclaim_dead_worker_ingestion")
                released += 1
            db.commit()
            return released
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
        # A dead claiming process is knowable now; the lease below only
        # catches workers that are alive but stuck, or that died elsewhere.
        dead = self.release_dead_worker_claims(agenda_id=agenda_id)
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
                  AND NOT EXISTS (
                      SELECT 1
                      FROM resource_grant_usage_reservations AS rgu
                      WHERE rgu.resource_grant_id=
                            scoped_ingestion_jobs_v1.resource_grant_id
                        AND rgu.agenda_id=scoped_ingestion_jobs_v1.agenda_id
                        AND rgu.status='reserved'
                  )
                """,
                (int(agenda_id),),
            )
            ambiguous = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='manual_reconciliation', lease_owner=NULL,
                    lease_expires_at=NULL,
                    failure_reason=
                        'worker_lease_expired_open_usage_reconciliation_required',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status='running'
                  AND lease_expires_at <= CURRENT_TIMESTAMP
                  AND attempt_count < max_attempts
                  AND agenda_id=?
                  AND EXISTS (
                      SELECT 1
                      FROM resource_grant_usage_reservations AS rgu
                      WHERE rgu.resource_grant_id=
                            scoped_ingestion_jobs_v1.resource_grant_id
                        AND rgu.agenda_id=scoped_ingestion_jobs_v1.agenda_id
                        AND rgu.status='reserved'
                  )
                """,
                (int(agenda_id),),
            )
            exhausted = db.execute(
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
                    getattr(ambiguous, "rowcount", 0) or 0
                )
                + int(
                    getattr(exhausted, "rowcount", 0) or 0
                ),
                "dead_worker_claims": dead,
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
                SELECT sij.*, rg.token_cap, rg.backend_allowlist_json,
                       rg.max_gpu_hours, arl.status AS ledger_status
                FROM scoped_ingestion_jobs_v1 AS sij
                JOIN resource_grants AS rg ON rg.id=sij.resource_grant_id
                JOIN agenda_resource_ledger AS arl
                  ON arl.id=rg.reservation_id AND arl.agenda_id=sij.agenda_id
                JOIN research_agendas AS ra ON ra.id=sij.agenda_id
                WHERE sij.status IN ('queued', 'retryable')
                  AND sij.attempt_count < sij.max_attempts
                  AND rg.agenda_id=sij.agenda_id
                  AND rg.idea_id=sij.idea_id
                  AND rg.stage=sij.stage
                  AND rg.status='active'
                  AND rg.expires_at > CURRENT_TIMESTAMP
                  AND arl.status='reserved'
                  AND ra.is_active=1
                  AND ra.status='active'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM scoped_ingestion_jobs_v1 AS active
                      WHERE active.agenda_id=sij.agenda_id
                        AND active.status='running'
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM resource_grant_usage_reservations AS rgu
                      WHERE rgu.resource_grant_id=sij.resource_grant_id
                        AND rgu.agenda_id=sij.agenda_id
                        AND rgu.status='reserved'
                  )
                ORDER BY sij.created_at, sij.id
                LIMIT 1 FOR UPDATE OF sij, ra SKIP LOCKED
                """
            )
            if not row:
                db.commit()
                return None
            try:
                require_ingestion_grant_stage(str(row.get("stage") or ""))
            except ResourceGrantStageError as exc:
                raise ScopedLLMError(str(exc)) from exc
            if (
                set(json.loads(row.get("backend_allowlist_json") or "[]"))
                != {"llm"}
                or float(row.get("max_gpu_hours") or 0.0) != 0.0
            ):
                raise ScopedLLMError(
                    "persisted ingestion ResourceGrant is not token-only and LLM-only"
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
    ) -> dict[str, Any]:
        """Finish a job and settle its entire ResourceGrant atomically.

        Scoped ingestion has no experiment OutcomeRecord, so its successful
        job is the terminal artifact against which usage is settled.  Until
        this method existed, successful ingestion grants stayed active until
        expiry and the agenda ledger did not reflect their spend.
        """

        if int(job_id or 0) <= 0 or int(agenda_id or 0) <= 0:
            raise ScopedLLMError("ingestion completion ids must be positive")
        if not str(worker_id or "").strip():
            raise ScopedLLMError("ingestion completion worker_id is required")
        result_rows = list(results)
        result_payload = {"papers": result_rows}
        try:
            job = db.fetchone(
                """
                SELECT * FROM scoped_ingestion_jobs_v1
                WHERE id=? AND agenda_id=? FOR UPDATE
                """,
                (int(job_id), int(agenda_id)),
            )
            if not job:
                raise ScopedLLMError("ingestion job was not found")
            expected_papers = _paper_ids(job.get("paper_ids_json"))
            result_papers = tuple(
                str(result.get("paper_id") or "").strip()
                for result in result_rows
                if isinstance(result, dict)
            )
            if (
                len(result_papers) != len(result_rows)
                or any(not paper_id for paper_id in result_papers)
                or len(set(result_papers)) != len(result_papers)
                or set(result_papers) != set(expected_papers)
                or len(result_papers) != len(expected_papers)
                or any(result.get("error") for result in result_rows)
            ):
                raise ScopedLLMError(
                    "ingestion completion results do not match the exact paper set"
                )
            result_by_paper = {
                str(result["paper_id"]): result for result in result_rows
            }
            incomplete_results = [
                paper_id
                for paper_id in expected_papers
                if int(result_by_paper[paper_id].get("claims") or 0) <= 0
                or int(result_by_paper[paper_id].get("graph_entities") or 0) <= 0
                or int(result_by_paper[paper_id].get("graph_relations") or 0) <= 0
            ]
            if incomplete_results:
                raise ScopedLLMError(
                    "ingestion completion caller did not report fresh or replayed "
                    "claims/graph lifecycle for papers:"
                    + ",".join(incomplete_results)
                )
            paper_rows = db.fetchall(
                """
                SELECT p.id, p.status, p.processing_stage,
                       (SELECT COUNT(*) FROM claims AS c
                        WHERE c.paper_id=p.id) AS claim_count,
                       (SELECT COUNT(*) FROM paper_entity_mentions AS pem
                        WHERE pem.paper_id=p.id) AS graph_entity_count,
                       (SELECT COUNT(*) FROM graph_relations AS gr
                        WHERE gr.paper_id=p.id) AS graph_relation_count
                FROM papers AS p
                WHERE p.id = ANY(?)
                ORDER BY p.id FOR UPDATE OF p
                """,
                (list(expected_papers),),
            )
            persisted_papers = {
                str(row.get("id") or ""): row for row in paper_rows
            }
            if (
                set(persisted_papers) != set(expected_papers)
                or any(
                    str(row.get("status") or "") != "reasoned"
                    or str(row.get("processing_stage") or "") != "reasoned"
                    or int(row.get("claim_count") or 0) <= 0
                    or int(row.get("graph_entity_count") or 0) <= 0
                    or int(row.get("graph_relation_count") or 0) <= 0
                    for row in persisted_papers.values()
                )
            ):
                raise ScopedLLMError(
                    "ingestion completion papers are not durably reasoned with "
                    "persisted claims and graph rows"
                )
            checkpoint_rows = db.fetchall(
                """
                SELECT paper_id, stage, payload, error_message
                FROM paper_stage_checkpoints
                WHERE paper_id = ANY(?)
                  AND stage IN ('extracted', 'graph_written', 'reasoned')
                ORDER BY paper_id, stage FOR UPDATE
                """,
                (list(expected_papers),),
            )
            checkpoints = {
                (str(row.get("paper_id") or ""), str(row.get("stage") or "")): row
                for row in checkpoint_rows
            }
            incomplete: list[str] = []
            for paper_id in expected_papers:
                extracted = checkpoints.get((paper_id, "extracted"))
                graph_written = checkpoints.get((paper_id, "graph_written"))
                reasoned = checkpoints.get((paper_id, "reasoned"))
                graph_payload = _mapping(
                    (graph_written or {}).get("payload")
                )
                reasoned_payload = _mapping((reasoned or {}).get("payload"))
                if (
                    not extracted
                    or extracted.get("error_message")
                    or not graph_written
                    or graph_written.get("error_message")
                    or not reasoned
                    or reasoned.get("error_message")
                    or int(graph_payload.get("claim_count") or 0) <= 0
                    or int(graph_payload.get("graph_entities") or 0) <= 0
                    or int(graph_payload.get("graph_relations") or 0) <= 0
                    or int(reasoned_payload.get("claims") or 0) <= 0
                ):
                    incomplete.append(paper_id)
            if incomplete:
                raise ScopedLLMError(
                    "ingestion completion lacks the full extraction/claims/graph "
                    "lifecycle for papers:"
                    + ",".join(incomplete)
                )
            persisted_result = _mapping(job.get("result_json"))
            if "recovery_origin" in persisted_result:
                result_payload["recovery_origin"] = persisted_result[
                    "recovery_origin"
                ]

            grant = db.fetchone(
                """
                SELECT *, CASE WHEN expires_at > CURRENT_TIMESTAMP
                               THEN 1 ELSE 0 END AS unexpired
                FROM resource_grants
                WHERE id=? FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            try:
                require_ingestion_grant_stage(str(job.get("stage") or ""))
            except ResourceGrantStageError as exc:
                raise ScopedLLMError(str(exc)) from exc
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != int(agenda_id)
                or int(grant.get("idea_id") or 0)
                != int(job.get("idea_id") or 0)
                or str(grant.get("stage") or "")
                != str(job.get("stage") or "")
                or float(grant.get("max_gpu_hours") or 0.0) != 0.0
                or set(json.loads(grant.get("backend_allowlist_json") or "[]"))
                != {"llm"}
            ):
                raise ScopedLLMError(
                    "ingestion completion ResourceGrant scope mismatch"
                )

            siblings = db.fetchall(
                """
                SELECT id, status FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            if (
                len(siblings) != 1
                or int(siblings[0].get("id") or 0) != int(job_id)
            ):
                raise ScopedLLMError(
                    "ingestion ResourceGrant is not one-to-one with its job"
                )

            usage_rows = db.fetchall(
                """
                SELECT id, status, token_reserved, tokens_used
                FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND agenda_id=?
                ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]), int(agenda_id)),
            )
            nonterminal_usage = [
                int(row["id"])
                for row in usage_rows
                if str(row.get("status") or "")
                not in {"settled", "released"}
            ]
            if nonterminal_usage:
                raise ScopedLLMError(
                    "ingestion grant has open child usage reservations:"
                    + ",".join(str(value) for value in nonterminal_usage)
                )
            metered_tokens = sum(
                int(row.get("tokens_used") or 0)
                for row in usage_rows
                if str(row.get("status") or "") == "settled"
            )
            if metered_tokens > int(grant.get("token_cap") or 0):
                raise ScopedLLMError(
                    "ingestion metered tokens exceed ResourceGrant cap"
                )

            gpu_attempts = db.fetchall(
                """
                SELECT id, status FROM experiment_attempt_gpu_reservations_v1
                WHERE resource_grant_id=? ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            if gpu_attempts:
                raise ScopedLLMError(
                    "scoped ingestion ResourceGrant has GPU attempt reservations"
                )

            ledger = db.fetchone(
                """
                SELECT * FROM agenda_resource_ledger
                WHERE id=? AND agenda_id=? FOR UPDATE
                """,
                (int(grant["reservation_id"]), int(agenda_id)),
            )
            if not ledger:
                raise ScopedLLMError(
                    "ingestion grant agenda reservation was not found"
                )
            token_reserved = int(ledger.get("token_reserved") or 0)
            gpu_reserved = float(ledger.get("gpu_hours_reserved") or 0.0)
            gpu_used = float(ledger.get("gpu_hours_used") or 0.0)
            if (
                token_reserved != int(grant.get("token_cap") or 0)
                or not math.isfinite(gpu_reserved)
                or abs(gpu_reserved) > 1e-9
            ):
                raise ScopedLLMError(
                    "ingestion parent reservation does not exactly match "
                    "the ResourceGrant cap"
                )

            if str(job.get("status") or "") == "succeeded":
                if _mapping(job.get("result_json")) != result_payload:
                    raise ScopedLLMError(
                        "duplicate ingestion completion result mismatch"
                    )
                if (
                    str(grant.get("status") or "") != "consumed"
                    or str(ledger.get("status") or "") != "settled"
                    or int(ledger.get("tokens_used") or 0) != metered_tokens
                    or not math.isfinite(gpu_used)
                    or abs(gpu_used) > 1e-9
                ):
                    raise ScopedLLMError(
                        "succeeded ingestion job has inconsistent settlement"
                    )
                db.commit()
                return {
                    "status": "already_succeeded",
                    "resource_grant_id": int(job["resource_grant_id"]),
                    "tokens_used": metered_tokens,
                }

            if (
                str(job.get("status") or "") != "running"
                or str(job.get("lease_owner") or "") != worker_id
            ):
                raise ScopedLLMError(
                    "ingestion completion does not own the active lease"
                )
            if (
                str(grant.get("status") or "") != "active"
                or not bool(grant.get("unexpired"))
            ):
                raise ScopedLLMError(
                    "ingestion completion ResourceGrant is not active"
                )
            if str(ledger.get("status") or "") != "reserved":
                raise ScopedLLMError(
                    "ingestion grant reservation is not settleable"
                )
            if metered_tokens > token_reserved:
                raise ScopedLLMError(
                    "ingestion metered tokens exceed agenda reservation"
                )
            if not math.isfinite(gpu_used) or abs(gpu_used) > 1e-9:
                raise ScopedLLMError(
                    "scoped ingestion agenda reservation contains GPU usage"
                )

            changed = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='succeeded', result_json=?, failure_reason=NULL,
                    lease_owner=NULL, lease_expires_at=NULL,
                    completed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                  AND lease_owner=?
                """,
                (
                    _dump(result_payload),
                    int(job_id),
                    int(agenda_id),
                    worker_id,
                ),
            )
            _expect_one(changed, operation="complete_ingestion_job")
            changed = db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved-?,
                    token_spent=token_spent+?,
                    gpu_hours_reserved=gpu_hours_reserved-?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND token_reserved>=?
                  AND gpu_hours_reserved>=?
                  AND token_spent+? <= token_budget
                """,
                (
                    token_reserved,
                    metered_tokens,
                    gpu_reserved,
                    int(agenda_id),
                    token_reserved,
                    gpu_reserved,
                    metered_tokens,
                ),
            )
            _expect_one(changed, operation="settle_ingestion_agenda_budget")
            changed = db.execute(
                """
                UPDATE agenda_resource_ledger
                SET tokens_used=?, gpu_hours_used=0, status='settled',
                    release_reason=NULL, settled_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='reserved'
                """,
                (metered_tokens, int(grant["reservation_id"]), int(agenda_id)),
            )
            _expect_one(changed, operation="settle_ingestion_grant_ledger")
            changed = db.execute(
                """
                UPDATE resource_grants
                SET status='consumed'
                WHERE id=? AND agenda_id=? AND status='active'
                """,
                (int(job["resource_grant_id"]), int(agenda_id)),
            )
            _expect_one(changed, operation="consume_ingestion_grant")
            db.commit()
            return {
                "status": "succeeded",
                "resource_grant_id": int(job["resource_grant_id"]),
                "tokens_used": metered_tokens,
            }
        except Exception:
            db.rollback()
            raise

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
            SELECT attempt_count, max_attempts, resource_grant_id
            FROM scoped_ingestion_jobs_v1
            WHERE id=? AND agenda_id=? AND status='running'
              AND lease_owner=?
            """,
            (int(job_id), int(agenda_id), worker_id),
        )
        if not row:
            raise ScopedLLMError("ingestion failure does not own the active lease")
        if (
            retryable
            and int(row.get("attempt_count") or 0)
            < int(row.get("max_attempts") or 0)
        ):
            open_usage = db.fetchone(
                """
                SELECT id FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND agenda_id=? AND status='reserved'
                ORDER BY id LIMIT 1
                """,
                (int(row["resource_grant_id"]), int(agenda_id)),
            )
            if open_usage:
                self._finish(
                    job_id,
                    agenda_id=agenda_id,
                    worker_id=worker_id,
                    status="manual_reconciliation",
                    result={"papers": list(partial_results)},
                    failure_reason=(
                        "retry_blocked_open_usage_disposition_required:"
                        + str(reason)
                    )[:1000],
                )
                return "manual_reconciliation"
            self._finish(
                job_id,
                agenda_id=agenda_id,
                worker_id=worker_id,
                status="retryable",
                result={"papers": list(partial_results)},
                failure_reason=reason,
            )
            return "retryable"
        return self._settle_terminal_failure(
            job_id,
            agenda_id=agenda_id,
            worker_id=worker_id,
            result={"papers": list(partial_results)},
            failure_reason=reason,
        )

    def _settle_terminal_failure(
        self,
        job_id: int,
        *,
        agenda_id: int,
        worker_id: str,
        result: dict,
        failure_reason: str,
    ) -> str:
        """Close a terminal failed job and its metered grant atomically.

        Failed inference still consumed tokens.  Leaving the parent grant
        active until expiry makes the job terminal while its authority and
        agenda reservation remain live.  Conversely, an open child usage row
        cannot be guessed away: that path is parked for exact operator
        reconciliation and keeps the parent reservation intact.
        """

        try:
            job = db.fetchone(
                """
                SELECT * FROM scoped_ingestion_jobs_v1
                WHERE id=? AND agenda_id=? AND status='running'
                  AND lease_owner=? FOR UPDATE
                """,
                (int(job_id), int(agenda_id), worker_id),
            )
            if not job:
                raise ScopedLLMError(
                    "ingestion failure does not own the active lease"
                )
            try:
                require_ingestion_grant_stage(str(job.get("stage") or ""))
            except ResourceGrantStageError as exc:
                raise ScopedLLMError(str(exc)) from exc
            prior_result = _mapping(job.get("result_json"))
            if "recovery_origin" in prior_result:
                result = {
                    **result,
                    "recovery_origin": prior_result["recovery_origin"],
                }
            grant = db.fetchone(
                """
                SELECT *, CASE WHEN expires_at > CURRENT_TIMESTAMP
                               THEN 1 ELSE 0 END AS unexpired
                FROM resource_grants WHERE id=? FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            if (
                not grant
                or int(grant.get("agenda_id") or 0) != int(agenda_id)
                or int(grant.get("idea_id") or 0)
                != int(job.get("idea_id") or 0)
                or str(grant.get("stage") or "")
                != str(job.get("stage") or "")
                or str(grant.get("status") or "") != "active"
                or not bool(grant.get("unexpired"))
                or float(grant.get("max_gpu_hours") or 0.0) != 0.0
                or set(json.loads(grant.get("backend_allowlist_json") or "[]"))
                != {"llm"}
            ):
                raise ScopedLLMError(
                    "terminal ingestion failure ResourceGrant scope mismatch"
                )
            siblings = db.fetchall(
                """
                SELECT id FROM scoped_ingestion_jobs_v1
                WHERE resource_grant_id=? ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            if (
                len(siblings) != 1
                or int(siblings[0].get("id") or 0) != int(job_id)
            ):
                raise ScopedLLMError(
                    "ingestion ResourceGrant is not one-to-one with its job"
                )
            usage_rows = db.fetchall(
                """
                SELECT id, status, tokens_used
                FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND agenda_id=?
                ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]), int(agenda_id)),
            )
            open_usage = [
                int(row["id"])
                for row in usage_rows
                if str(row.get("status") or "")
                not in {"settled", "released"}
            ]
            if open_usage:
                changed = db.execute(
                    """
                    UPDATE scoped_ingestion_jobs_v1
                    SET status='manual_reconciliation', result_json=?,
                        failure_reason=?, lease_owner=NULL,
                        lease_expires_at=NULL,
                        completed_at=CURRENT_TIMESTAMP,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=? AND agenda_id=? AND status='running'
                      AND lease_owner=?
                    """,
                    (
                        _dump(result),
                        (
                            "terminal_failure_open_usage:"
                            + ",".join(str(value) for value in open_usage)
                            + ":"
                            + str(failure_reason)
                        )[:1000],
                        int(job_id),
                        int(agenda_id),
                        worker_id,
                    ),
                )
                _expect_one(
                    changed,
                    operation="park_terminal_ingestion_open_usage",
                )
                db.commit()
                return "manual_reconciliation"
            metered_tokens = sum(
                int(row.get("tokens_used") or 0)
                for row in usage_rows
                if str(row.get("status") or "") == "settled"
            )
            if metered_tokens > int(grant.get("token_cap") or 0):
                raise ScopedLLMError(
                    "terminal ingestion usage exceeds ResourceGrant cap"
                )
            gpu_attempts = db.fetchall(
                """
                SELECT id FROM experiment_attempt_gpu_reservations_v1
                WHERE resource_grant_id=? ORDER BY id FOR UPDATE
                """,
                (int(job["resource_grant_id"]),),
            )
            if gpu_attempts:
                raise ScopedLLMError(
                    "scoped ingestion ResourceGrant has GPU attempt reservations"
                )
            ledger = db.fetchone(
                """
                SELECT * FROM agenda_resource_ledger
                WHERE id=? AND agenda_id=? FOR UPDATE
                """,
                (int(grant["reservation_id"]), int(agenda_id)),
            )
            if not ledger or str(ledger.get("status") or "") != "reserved":
                raise ScopedLLMError(
                    "terminal ingestion grant reservation is not settleable"
                )
            token_reserved = int(ledger.get("token_reserved") or 0)
            gpu_reserved = float(ledger.get("gpu_hours_reserved") or 0.0)
            gpu_used = float(ledger.get("gpu_hours_used") or 0.0)
            if (
                token_reserved != int(grant.get("token_cap") or 0)
                or not math.isfinite(gpu_reserved)
                or abs(gpu_reserved) > 1e-9
            ):
                raise ScopedLLMError(
                    "terminal ingestion parent reservation does not exactly "
                    "match the ResourceGrant cap"
                )
            if metered_tokens > token_reserved:
                raise ScopedLLMError(
                    "terminal ingestion usage exceeds agenda reservation"
                )
            if not math.isfinite(gpu_used) or abs(gpu_used) > 1e-9:
                raise ScopedLLMError(
                    "terminal ingestion agenda reservation contains GPU usage"
                )
            changed = db.execute(
                """
                UPDATE scoped_ingestion_jobs_v1
                SET status='failed', result_json=?, failure_reason=?,
                    lease_owner=NULL, lease_expires_at=NULL,
                    completed_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='running'
                  AND lease_owner=?
                """,
                (
                    _dump(result),
                    str(failure_reason)[:1000],
                    int(job_id),
                    int(agenda_id),
                    worker_id,
                ),
            )
            _expect_one(changed, operation="fail_terminal_ingestion_job")
            changed = db.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved-?,
                    token_spent=token_spent+?,
                    gpu_hours_reserved=gpu_hours_reserved-?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND token_reserved>=?
                  AND gpu_hours_reserved>=?
                  AND token_spent+? <= token_budget
                """,
                (
                    token_reserved,
                    metered_tokens,
                    gpu_reserved,
                    int(agenda_id),
                    token_reserved,
                    gpu_reserved,
                    metered_tokens,
                ),
            )
            _expect_one(changed, operation="settle_failed_ingestion_budget")
            changed = db.execute(
                """
                UPDATE agenda_resource_ledger
                SET tokens_used=?, gpu_hours_used=0, status='settled',
                    release_reason='terminal_ingestion_failure',
                    settled_at=CURRENT_TIMESTAMP
                WHERE id=? AND agenda_id=? AND status='reserved'
                """,
                (metered_tokens, int(grant["reservation_id"]), int(agenda_id)),
            )
            _expect_one(changed, operation="settle_failed_ingestion_ledger")
            changed = db.execute(
                """
                UPDATE resource_grants SET status='consumed'
                WHERE id=? AND agenda_id=? AND status='active'
                """,
                (int(job["resource_grant_id"]), int(agenda_id)),
            )
            _expect_one(changed, operation="consume_failed_ingestion_grant")
            db.commit()
            return "failed"
        except Exception:
            db.rollback()
            raise

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
            persisted = db.fetchone(
                """
                SELECT result_json FROM scoped_ingestion_jobs_v1
                WHERE id=? AND agenda_id=? AND status='running'
                  AND lease_owner=? FOR UPDATE
                """,
                (int(job_id), int(agenda_id), worker_id),
            )
            if not persisted:
                raise ScopedLLMError("ingestion completion lost its worker lease")
            prior_result = _mapping(persisted.get("result_json"))
            if "recovery_origin" in prior_result:
                result = {
                    **result,
                    "recovery_origin": prior_result["recovery_origin"],
                }
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
