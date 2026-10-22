"""Agenda-scoped persistence for Harness Evolution evaluation artifacts."""

from __future__ import annotations

import json
from typing import Any

from db import database as db
from meta_harness.harness_evolution import (
    EvaluationRun,
)


class HarnessPersistenceError(RuntimeError):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class HarnessRepository:
    def save_evaluation(
        self,
        evaluation: EvaluationRun,
        *,
        candidate_id: int,
        patch_id: int,
    ) -> int:
        if evaluation.agenda_id <= 0:
            raise HarnessPersistenceError("EvaluationRun requires agenda_id")
        if evaluation.suite not in {"held_in", "held_out", "canary"}:
            raise HarnessPersistenceError("invalid evaluation suite")
        if evaluation.status not in {"passed", "failed"}:
            raise HarnessPersistenceError("invalid evaluation status")
        if evaluation.status == "failed" and not evaluation.failure_reason:
            raise HarnessPersistenceError(
                "failed evaluation requires a failure reason"
            )
        if not evaluation.evaluator_ref or not evaluation.evaluator_hash:
            raise HarnessPersistenceError("evaluator ref/hash are required")
        if not evaluation.artifact_manifest:
            raise HarnessPersistenceError("evaluation artifacts are required")
        scope = db.fetchone(
            """
            SELECT hc.agenda_id, hp.agenda_id AS patch_agenda_id
            FROM harness_candidates hc
            JOIN harness_patches hp ON hp.candidate_id=hc.id
            WHERE hc.id=? AND hp.id=?
            """,
            (int(candidate_id), int(patch_id)),
        )
        if (
            not scope
            or int(scope.get("agenda_id") or 0) != evaluation.agenda_id
            or int(scope.get("patch_agenda_id") or 0) != evaluation.agenda_id
        ):
            raise HarnessPersistenceError("evaluation scope mismatch")
        existing = db.fetchone(
            """
            SELECT id FROM harness_evaluation_runs
            WHERE candidate_id=? AND patch_id=? AND suite=?
            """,
            (candidate_id, patch_id, evaluation.suite),
        )
        if existing:
            raise HarnessPersistenceError(
                "evaluation suite is immutable; create a new patch/candidate"
            )
        evaluation_id = db.insert_returning_id(
            """
            INSERT INTO harness_evaluation_runs
                (agenda_id, candidate_id, patch_id, suite, evaluator_ref,
                 evaluator_hash, status, result_json, artifact_manifest_json,
                 completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (
                evaluation.agenda_id,
                candidate_id,
                patch_id,
                evaluation.suite,
                evaluation.evaluator_ref,
                evaluation.evaluator_hash,
                evaluation.status,
                _dump({"failure_reason": evaluation.failure_reason}),
                _dump(evaluation.artifact_manifest),
            ),
        )
        db.commit()
        return evaluation_id
