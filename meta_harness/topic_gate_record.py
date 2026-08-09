"""The missing writer for a candidate's recorded topic-gate answers.

``agents.topic_gate`` reads ``deep_insights.topic_gate_json`` and deliberately
refuses to invent a prediction with an ungranted model call. Migration
``0002_topic_gate_and_frontier_authority`` added the column and the gate reads
it -- but nothing in the tree ever wrote it, so every agenda whose candidates
were not hand-edited in SQL was stuck at ``topic_gate_prediction_missing``
with no sanctioned way forward. That is the same "one side of the wire is
missing" defect class as ``stage='portfolio_granted'`` having no reader; this
module is the other half of this wire, and deliberately nothing more.

What it will not do, because a pre-registration that can be edited afterwards
is not a pre-registration:

* it never writes a record the gate would reject -- a stored answer that cannot
  pass is noise that later reads have to second-guess;
* it never changes a record once the candidate has bought resources. Any
  ResourceGrant or OutcomeRecord for the idea freezes the answers; re-writing
  the identical record stays a no-op so retries and replays are safe;
* it never crosses an agenda: the candidate must already be bound to the
  agenda named by the caller;
* it never calls a model. The prediction is the researcher's commitment, and
  eliciting it from an LLM here would reintroduce exactly the ungranted call
  the gate exists to refuse.

Provenance travels inside the record itself rather than in a side table: the
row is the thing later reads trust, so who drafted and who authorized the
prediction has to be readable from the same JSON the gate reads.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping

from agents.topic_gate import (
    TopicGateDecision,
    TopicGatePolicy,
    policy_from_config,
    screen_candidate,
)
from db import database as db


class TopicGateRecordError(PermissionError):
    """Raised before any write when the recorded answers cannot be trusted."""


REQUIRED_PREDICTION_FIELDS = (
    "predicted_outcome",
    "confidence",
    "action_if_confirmed",
    "action_if_refuted",
)
REQUIRED_EXPERIMENT_FIELDS = ("metric", "baseline", "decisive_comparison")
REQUIRED_PROVENANCE_FIELDS = ("drafted_by", "authorized_by", "review_status")


def canonical(record: Mapping[str, Any]) -> str:
    """One byte-for-byte form, so "same record" is a decidable question."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)


def _shape_blockers(record: Mapping[str, Any]) -> list[str]:
    """Structural checks the gate itself cannot make.

    The gate reads a candidate row and answers pass/fail; it has no opinion on
    whether a *record* is complete enough to be worth persisting. Missing keys
    would simply read back as a missing prediction later, which is indis-
    tinguishable from never having recorded one at all.
    """
    missing: list[str] = []
    prediction = record.get("prediction")
    if not isinstance(prediction, Mapping):
        return ["prediction"]
    for name in REQUIRED_PREDICTION_FIELDS:
        value = prediction.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(f"prediction.{name}")
    experiment = record.get("minimum_falsification_experiment")
    if not isinstance(experiment, Mapping):
        missing.append("minimum_falsification_experiment")
    else:
        for name in REQUIRED_EXPERIMENT_FIELDS:
            if not str(experiment.get(name) or "").strip():
                missing.append(f"minimum_falsification_experiment.{name}")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        missing.append("provenance")
    else:
        for name in REQUIRED_PROVENANCE_FIELDS:
            if not str(provenance.get(name) or "").strip():
                missing.append(f"provenance.{name}")
    return missing


def _frozen_by(*, agenda_id: int, idea_id: int) -> list[str]:
    """Commitments that must outlive any later change of mind."""
    frozen: list[str] = []
    grants = db.fetchone(
        "SELECT COUNT(*) AS count FROM resource_grants "
        "WHERE agenda_id=? AND idea_id=?",
        (int(agenda_id), int(idea_id)),
    )
    if int((grants or {}).get("count") or 0) > 0:
        frozen.append("resource_grant_issued")
    outcomes = db.fetchone(
        "SELECT COUNT(*) AS count FROM outcome_records "
        "WHERE agenda_id=? AND idea_id=?",
        (int(agenda_id), int(idea_id)),
    )
    if int((outcomes or {}).get("count") or 0) > 0:
        frozen.append("outcome_recorded")
    return frozen


def evaluate_record(
    *,
    agenda_id: int,
    idea_id: int,
    record: Mapping[str, Any],
    policy: TopicGatePolicy | None = None,
) -> tuple[TopicGateDecision, dict[str, Any]]:
    """Screen the candidate as if the record were already stored. No writes.

    This is what makes the CLI's ``--dry-run`` meaningful: the verdict it
    prints is produced by the same pure function that will re-run from the
    persisted row when a portfolio decision is attempted.
    """
    from agents.agenda_repository import row_to_agenda

    if int(agenda_id) <= 0 or int(idea_id) <= 0:
        raise TopicGateRecordError("topic gate record scope ids must be positive")
    missing = _shape_blockers(record)
    if missing:
        raise TopicGateRecordError(
            "topic gate record is incomplete:" + ",".join(sorted(missing))
        )
    candidate = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (int(idea_id), int(agenda_id)),
    )
    if not candidate:
        raise TopicGateRecordError(
            "candidate is not bound to this agenda; explicit import is required"
        )
    agenda_row = db.fetchone(
        "SELECT * FROM research_agendas WHERE id=?", (int(agenda_id),)
    )
    if not agenda_row:
        raise TopicGateRecordError("agenda does not exist")
    merged = dict(candidate)
    merged["topic_gate_json"] = canonical(record)
    decision = screen_candidate(
        merged,
        row_to_agenda(agenda_row),
        policy=policy or policy_from_config(),
    )
    return decision, dict(candidate)


def record_prediction(
    *,
    agenda_id: int,
    idea_id: int,
    record: Mapping[str, Any],
    actor: str,
    policy: TopicGatePolicy | None = None,
) -> dict[str, Any]:
    """Persist one candidate's gate answers, or refuse with reasons.

    Returns the outcome (``recorded`` / ``unchanged``) together with the gate
    verdict, so the caller never has to guess whether the candidate can now
    reach a portfolio decision.
    """
    if not str(actor or "").strip():
        raise TopicGateRecordError("actor is required so the write is auditable")
    stamped = dict(record)
    provenance = dict(stamped.get("provenance") or {})
    provenance["recorded_by"] = str(actor).strip()
    provenance["recorded_at"] = datetime.now(timezone.utc).isoformat()
    stamped["provenance"] = provenance

    decision, candidate = evaluate_record(
        agenda_id=agenda_id, idea_id=idea_id, record=stamped, policy=policy
    )
    payload = canonical(stamped)
    stored = str(candidate.get("topic_gate_json") or "").strip()

    # Compared without the stamp: a replay that only differs by its recording
    # timestamp is the same pre-registration, and must not count as an edit.
    def _without_stamp(text: str) -> str:
        try:
            loaded = json.loads(text)
        except (TypeError, ValueError):
            return text
        if isinstance(loaded, dict) and isinstance(loaded.get("provenance"), dict):
            loaded["provenance"] = {
                key: value
                for key, value in loaded["provenance"].items()
                if key not in {"recorded_at", "recorded_by"}
            }
        return canonical(loaded)

    unchanged = bool(stored) and _without_stamp(stored) == _without_stamp(payload)
    if unchanged:
        return {
            "status": "unchanged",
            "agenda_id": int(agenda_id),
            "idea_id": int(idea_id),
            "gate_passed": decision.passed,
            "gate_reason_codes": list(decision.reason_codes),
        }

    frozen = _frozen_by(agenda_id=int(agenda_id), idea_id=int(idea_id))
    if frozen and stored:
        raise TopicGateRecordError(
            "pre-registration is frozen and cannot be rewritten:"
            + ",".join(sorted(frozen))
        )
    if not decision.passed:
        raise TopicGateRecordError(
            "topic gate would reject this record; it is not persisted:"
            + ",".join(decision.reason_codes)
        )

    try:
        cursor = db.execute(
            "UPDATE deep_insights SET topic_gate_json=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=? AND agenda_id=?",
            (payload, int(idea_id), int(agenda_id)),
        )
        if int(getattr(cursor, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise TopicGateRecordError("candidate scope changed during the write")
        if db._use_pg():  # noqa: SLF001 - durable stage history is PostgreSQL-only.
            stage = (
                "proposal"
                if str(candidate.get("status") or "") == "proposal_pending"
                else "experiment"
            )
            db.execute(
                """
                INSERT INTO candidate_stage_gate_records_v1
                    (agenda_id, idea_id, stage, record_json, content_hash, actor)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (agenda_id, idea_id, stage, content_hash) DO NOTHING
                """,
                (
                    int(agenda_id),
                    int(idea_id),
                    stage,
                    payload,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    str(actor).strip(),
                ),
            )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {
        "status": "recorded",
        "agenda_id": int(agenda_id),
        "idea_id": int(idea_id),
        "gate_passed": decision.passed,
        "gate_reason_codes": list(decision.reason_codes),
        "expected_bits": decision.expected_bits,
        "refute_bits": decision.refute_bits,
        "confidence": decision.confidence,
    }
