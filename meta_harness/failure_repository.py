"""Durable failure fingerprinting and recovery decisions for experiment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from db import database as db
from meta_harness.attempt_gpu_usage import GrantGPUUsageControl
from meta_harness.failure_policy import (
    FailureContext,
    RecoveryDecision,
    classify_failure,
    decide_recovery,
)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _code_hash(workdir: str) -> str:
    root = Path(workdir) / "code"
    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class FailureRecoveryRepository:
    def decide_for_run(
        self,
        *,
        experiment_run_id: int,
        execution_result: Mapping[str, Any],
        retry_count: int = 0,
    ) -> tuple[RecoveryDecision, str, int]:
        run = db.fetchone(
            """
            SELECT er.id, er.agenda_id, er.deep_insight_id, er.resource_grant_id,
                   er.workdir, er.error_message, rg.preflight_result_id,
                   p.adapter_id, p.adapter_version, p.dataset_revision,
                   p.model_revision, p.environment_json
            FROM experiment_runs er
            LEFT JOIN resource_grants rg ON rg.id=er.resource_grant_id
            LEFT JOIN candidate_preflight_results_v1 p
              ON p.id=rg.preflight_result_id
            WHERE er.id=?
            """,
            (int(experiment_run_id),),
        )
        if not run:
            raise RuntimeError("failure_run_not_found")
        detail = str(
            execution_result.get("error")
            or execution_result.get("failure_type")
            or run.get("error_message")
            or "unknown execution failure"
        )[:4000]
        reason_code = str(execution_result.get("reason_code") or "").strip()
        if not reason_code:
            reason_code = classify_failure(
                message=" ".join(
                    [detail, str(execution_result.get("failure_type") or "")]
                ),
                returncode=execution_result.get("returncode"),
                final_results_present=bool(
                    execution_result.get("final_results_present")
                ),
            )
        code_hash = _code_hash(str(run.get("workdir") or ""))
        environment_hash = _hash_payload(
            {
                "adapter_id": run.get("adapter_id"),
                "adapter_version": run.get("adapter_version"),
                "dataset_revision": run.get("dataset_revision"),
                "model_revision": run.get("model_revision"),
                "preflight_environment": run.get("environment_json"),
                "backend": execution_result.get("backend"),
                "worker_id": execution_result.get("worker_id"),
                "visible_device": execution_result.get("visible_device"),
            }
        )
        remaining = 0.0
        grant_id = int(run.get("resource_grant_id") or 0)
        if grant_id > 0:
            remaining = GrantGPUUsageControl().grant_usage(grant_id).remaining_gpu_seconds
        context = FailureContext(
            reason_code=reason_code,
            detail=detail,
            code_hash=code_hash,
            environment_hash=environment_hash,
            remaining_gpu_seconds=remaining,
            retry_count=max(0, int(retry_count)),
        )
        fingerprint = context.fingerprint()
        existing = db.fetchone(
            """
            SELECT id, occurrences FROM experiment_failure_fingerprints_v1
            WHERE agenda_id=? AND idea_id=? AND fingerprint=?
            """,
            (
                int(run["agenda_id"]),
                int(run["deep_insight_id"]),
                fingerprint,
            ),
        )
        decision = decide_recovery(context, fingerprint_seen=bool(existing))
        decision_payload = {
            "action": decision.action,
            "retryable": decision.retryable,
            "invoke_llm_repair": decision.invoke_llm_repair,
            "reason_code": decision.reason_code,
            "backoff_seconds": decision.backoff_seconds,
            "adjustments": dict(decision.adjustments),
        }
        if existing:
            db.execute(
                """
                UPDATE experiment_failure_fingerprints_v1
                SET occurrences=occurrences + 1, last_seen_at=CURRENT_TIMESTAMP,
                    recovery_action=?, recovery_json=?
                WHERE id=?
                """,
                (
                    decision.action,
                    json.dumps(decision_payload, sort_keys=True),
                    int(existing["id"]),
                ),
            )
            record_id = int(existing["id"])
        else:
            record_id = int(
                db.insert_returning_id(
                    """
                    INSERT INTO experiment_failure_fingerprints_v1
                        (agenda_id, idea_id, experiment_run_id,
                         resource_grant_id, reason_code, fingerprint,
                         code_hash, environment_hash, detail,
                         recovery_action, recovery_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
                    """,
                    (
                        int(run["agenda_id"]),
                        int(run["deep_insight_id"]),
                        int(run["id"]),
                        grant_id or None,
                        reason_code,
                        fingerprint,
                        code_hash,
                        environment_hash,
                        detail,
                        decision.action,
                        json.dumps(decision_payload, sort_keys=True),
                    ),
                )
            )
        db.commit()
        return decision, fingerprint, record_id
