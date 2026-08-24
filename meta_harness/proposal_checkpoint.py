"""Durable delivery checkpoints for one explicitly bounded proposal.

The generic LLM router settles usage before returning to its caller.  For a
controlled proposal canary, a process crash between those two actions must not
cause the same delivered method or experiment to be bought twice.  This
repository stores the exact provider output in the existing durable pipeline
event store *before* settlement.  Replay can then settle that exact reservation
and reuse the output without another provider call.

No table is created here.  ``pipeline_events`` is an existing application
contract, so deploying this recovery path does not perform an implicit schema
mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from db import database as db
from meta_harness.grant_usage import GrantUsageLedger


CHECKPOINT_EVENT = "bounded_proposal_llm_delivered_v1"
CHECKPOINT_SCHEMA = "bounded-proposal-llm-delivery-v1"


class ProposalCheckpointError(RuntimeError):
    """Exact proposal replay cannot safely continue automatically."""


@dataclass(frozen=True)
class ProposalCheckpointScope:
    job_id: int
    agenda_id: int
    idea_id: int
    resource_grant_id: int
    operation: str
    input_digest: str

    def validate(self) -> None:
        if min(
            int(self.job_id),
            int(self.agenda_id),
            int(self.idea_id),
            int(self.resource_grant_id),
        ) <= 0:
            raise ProposalCheckpointError("proposal checkpoint scope ids must be positive")
        if not str(self.operation or "").strip():
            raise ProposalCheckpointError("proposal checkpoint operation is required")
        digest = str(self.input_digest or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ProposalCheckpointError("proposal checkpoint input digest is invalid")


def proposal_input_digest(
    *,
    job_id: int,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
    operation: str,
    system_prompt: str,
    user_prompt: str,
    prompt_version: str,
    token_cap: int,
) -> str:
    """Fingerprint all authority and prompt inputs for one billed operation."""

    payload = {
        "job_id": int(job_id),
        "agenda_id": int(agenda_id),
        "idea_id": int(idea_id),
        "resource_grant_id": int(resource_grant_id),
        "operation": str(operation),
        "system_prompt": str(system_prompt),
        "user_prompt": str(user_prompt),
        "prompt_version": str(prompt_version),
        "token_cap": int(token_cap),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ProposalCheckpointRepository:
    """Repository path for exact proposal delivery/replay state."""

    @staticmethod
    def idempotency_base(scope: ProposalCheckpointScope) -> str:
        scope.validate()
        return (
            f"bounded-proposal:{scope.operation}:{scope.job_id}:"
            f"{scope.agenda_id}:{scope.idea_id}:{scope.resource_grant_id}:"
            f"i{scope.input_digest}"
        )

    @staticmethod
    def checkpoint_key(scope: ProposalCheckpointScope) -> str:
        return "proposal-llm-checkpoint:" + ProposalCheckpointRepository.idempotency_base(
            scope
        )

    def _decode(
        self,
        row: dict[str, Any] | None,
        scope: ProposalCheckpointScope,
    ) -> dict | None:
        if not row:
            return None
        try:
            payload = json.loads(row.get("payload") or "{}")
        except (TypeError, ValueError) as exc:
            raise ProposalCheckpointError("proposal checkpoint payload is malformed") from exc
        expected = {
            "schema": CHECKPOINT_SCHEMA,
            "job_id": int(scope.job_id),
            "agenda_id": int(scope.agenda_id),
            "idea_id": int(scope.idea_id),
            "resource_grant_id": int(scope.resource_grant_id),
            "operation": str(scope.operation),
            "input_digest": str(scope.input_digest),
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ProposalCheckpointError("proposal checkpoint scope/fingerprint mismatch")
        if not str(payload.get("output") or "").strip():
            raise ProposalCheckpointError("proposal checkpoint has no delivered output")
        if int(payload.get("reservation_id") or 0) <= 0:
            raise ProposalCheckpointError("proposal checkpoint has no reservation identity")
        if int(payload.get("tokens_used") or 0) < 0:
            raise ProposalCheckpointError("proposal checkpoint token usage is invalid")
        cost = payload.get("cost_usd")
        if cost is not None and (not math.isfinite(float(cost)) or float(cost) < 0):
            raise ProposalCheckpointError("proposal checkpoint cost is invalid")
        return payload

    def load(self, scope: ProposalCheckpointScope) -> dict | None:
        scope.validate()
        row = db.fetchone(
            """
            SELECT payload
            FROM pipeline_events
            WHERE event_type=? AND entity_type='proposal_candidate'
              AND entity_id=? AND dedupe_key=?
            """,
            (
                CHECKPOINT_EVENT,
                str(scope.idea_id),
                self.checkpoint_key(scope),
            ),
        )
        return self._decode(row, scope)

    def _refuse_input_drift(self, scope: ProposalCheckpointScope) -> None:
        rows = db.fetchall(
            """
            SELECT payload
            FROM pipeline_events
            WHERE event_type=? AND entity_type='proposal_candidate'
              AND entity_id=?
            ORDER BY id
            """,
            (CHECKPOINT_EVENT, str(scope.idea_id)),
        )
        for row in rows or []:
            try:
                payload = json.loads(row.get("payload") or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            same_operation = (
                int(payload.get("job_id") or 0) == scope.job_id
                and int(payload.get("agenda_id") or 0) == scope.agenda_id
                and int(payload.get("idea_id") or 0) == scope.idea_id
                and int(payload.get("resource_grant_id") or 0)
                == scope.resource_grant_id
                and str(payload.get("operation") or "") == scope.operation
            )
            if same_operation and str(payload.get("input_digest") or "") != scope.input_digest:
                raise ProposalCheckpointError(
                    "proposal input fingerprint changed after a delivered result; "
                    "automatic re-billing is refused"
                )

    def save_delivery(
        self,
        scope: ProposalCheckpointScope,
        delivery: dict[str, Any],
    ) -> None:
        """Persist one delivered output before the router settles its usage."""

        scope.validate()
        expected_base = self.idempotency_base(scope)
        if (
            int(delivery.get("agenda_id") or 0) != scope.agenda_id
            or int(delivery.get("idea_id") or 0) != scope.idea_id
            or int(delivery.get("resource_grant_id") or 0)
            != scope.resource_grant_id
            or str(delivery.get("operation") or "") != scope.operation
            or not str(delivery.get("idempotency_key") or "").startswith(
                expected_base + ":t"
            )
        ):
            raise ProposalCheckpointError("delivered proposal escaped checkpoint scope")
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "job_id": scope.job_id,
            "agenda_id": scope.agenda_id,
            "idea_id": scope.idea_id,
            "resource_grant_id": scope.resource_grant_id,
            "operation": scope.operation,
            "input_digest": scope.input_digest,
            "idempotency_key": str(delivery["idempotency_key"]),
            "reservation_id": int(delivery.get("reservation_id") or 0),
            "tokens_used": int(delivery.get("tokens_used") or 0),
            "cost_usd": delivery.get("cost_usd"),
            "route": dict(delivery.get("route") or {}),
            "output": str(delivery.get("output") or ""),
        }
        # A duplicate callback may only confirm the exact same delivery.  Never
        # use pipeline_events' generic upsert here: overwriting a checkpoint
        # would erase the evidence needed to prevent double billing.
        existing = self.load(scope)
        if existing is not None:
            if existing != payload:
                raise ProposalCheckpointError("conflicting proposal delivery checkpoint")
            return
        row = db.fetchone(
            """
            INSERT INTO pipeline_events
                (event_type, entity_type, entity_id, dedupe_key, payload)
            VALUES (?, 'proposal_candidate', ?, ?, ?)
            ON CONFLICT(dedupe_key) DO NOTHING
            RETURNING id
            """,
            (
                CHECKPOINT_EVENT,
                str(scope.idea_id),
                self.checkpoint_key(scope),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        db.commit()
        if not row:
            concurrent = self.load(scope)
            if concurrent != payload:
                raise ProposalCheckpointError("conflicting concurrent proposal checkpoint")

    def recover_or_refuse(self, scope: ProposalCheckpointScope) -> dict | None:
        """Return a delivered result, or prove that issuing a call is safe.

        A reserved attempt without a delivery checkpoint is ambiguous: the
        process may have died before or after the provider accepted the call.
        A settled attempt without a checkpoint proves spend but not a replayable
        result.  Both states stop exact recovery rather than risking a duplicate
        charge; an operator can reconcile the named reservation through the
        ResourceGrant usage repository.
        """

        checkpoint = self.load(scope)
        if checkpoint is not None:
            reservation = db.fetchone(
                """
                SELECT r.id, r.status, r.operation, r.idempotency_key,
                       r.token_reserved, r.tokens_used, r.cost_usd
                FROM resource_grant_usage_reservations r
                JOIN resource_grants g ON g.id=r.resource_grant_id
                WHERE r.id=? AND r.resource_grant_id=? AND r.agenda_id=?
                  AND g.idea_id=?
                """,
                (
                    int(checkpoint["reservation_id"]),
                    scope.resource_grant_id,
                    scope.agenda_id,
                    scope.idea_id,
                ),
            )
            if (
                not reservation
                or str(reservation.get("operation") or "") != scope.operation
                or str(reservation.get("idempotency_key") or "")
                != str(checkpoint.get("idempotency_key") or "")
            ):
                raise ProposalCheckpointError(
                    "proposal checkpoint reservation scope is inconsistent"
                )
            status = str(reservation.get("status") or "")
            tokens_used = int(checkpoint.get("tokens_used") or 0)
            if tokens_used > int(reservation.get("token_reserved") or 0):
                raise ProposalCheckpointError(
                    "proposal checkpoint exceeds its reservation"
                )
            if status == "reserved":
                GrantUsageLedger(scope.resource_grant_id).settle(
                    int(checkpoint["reservation_id"]),
                    tokens_used=tokens_used,
                    cost_usd=checkpoint.get("cost_usd"),
                )
            elif status == "settled":
                if int(reservation.get("tokens_used") or 0) != tokens_used:
                    raise ProposalCheckpointError(
                        "settled proposal usage disagrees with its checkpoint"
                    )
                expected_cost = checkpoint.get("cost_usd")
                actual_cost = reservation.get("cost_usd")
                if (expected_cost is None) != (actual_cost is None) or (
                    expected_cost is not None
                    and not math.isclose(
                        float(expected_cost),
                        float(actual_cost),
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ProposalCheckpointError(
                        "settled proposal cost disagrees with its checkpoint"
                    )
            else:
                raise ProposalCheckpointError(
                    f"proposal checkpoint reservation is {status or 'unknown'}"
                )
            return checkpoint

        self._refuse_input_drift(scope)
        base = self.idempotency_base(scope)
        attempts = db.fetchall(
            """
            SELECT id, status
            FROM resource_grant_usage_reservations
            WHERE resource_grant_id=? AND operation=?
              AND (idempotency_key=? OR idempotency_key LIKE ?)
            ORDER BY id
            """,
            (
                scope.resource_grant_id,
                scope.operation,
                base,
                f"{base}:t%",
            ),
        )
        ambiguous = [
            row
            for row in attempts or []
            if str(row.get("status") or "") in {"reserved", "settled"}
        ]
        if ambiguous:
            row = ambiguous[-1]
            status = str(row.get("status") or "")
            raise ProposalCheckpointError(
                f"proposal usage reservation {int(row['id'])} is {status} "
                "without a delivered checkpoint; exact proposal halted for "
                "operator reconciliation"
            )
        return None
