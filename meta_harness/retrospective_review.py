"""Retrospective (legacy) review: a human-signed path onto the evidence ladder
for historical runs that predate ResourceGrants.

This is deliberately a PARALLEL writer, not a modification of the native gate:
`MetaHarnessRepository.advance_experiment_state` keeps requiring a persisted
ResourceGrant, and nothing here relaxes it. For pre-grant history the missing
authorization chain is replaced by an explicit, HMAC-signed reviewer approval
(the same primitive the manuscript gate uses), and every record this module
writes is marked so provenance stays honest:

- transition actor is ``retrospective_review:<reviewer_id>``;
- every transition context carries ``legacy_review: true``;
- all content hashes are computed from real rows and real files on disk;
- there is no holdout for historical runs, so the verdict is CAPPED at
  ``inconclusive`` - supported/refuted are refused; the waiver itself is
  hashed and recorded as the holdout reference.

A run is eligible only if it is operationally completed with a real measured
baseline, is agenda-scoped, still sits at the bottom of the ladder, and has no
scientific decision recorded.
"""
from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import database as db
from meta_harness.reviewer_approval import (
    ReviewerApproval,
    ReviewerApprovalError,
    ReviewerApprovalVerifier,
)

PURPOSE = "retrospective_review"
EVALUATOR_REF = "retrospective-metrics-check-v1"
HOLDOUT_REF = "holdout_waived_retrospective_v1"
HOLDOUT_WAIVER_TEXT = (
    "No holdout set exists for this historical run. Under the retrospective "
    "review policy the scientific verdict is therefore capped at "
    "'inconclusive'; supported and refuted verdicts are not reachable."
)
LADDER = (
    "sanity_passed",
    "full_benchmark_complete",
    "evidence_audited",
    "scientifically_decided",
)


class RetrospectiveReviewError(RuntimeError):
    pass


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def retrospective_subject(*, agenda_id: int, experiment_run_id: int) -> str:
    return f"retrospective-review:{int(agenda_id)}:{int(experiment_run_id)}"


@dataclass
class RunEvidence:
    run: dict
    claims: list[dict]
    artifacts: list[dict]
    artifact_files_present: int
    artifact_files_missing: int
    raw_artifacts_hash: str
    claim_ledger_hash: str
    benchmark_contract_hash: str
    evaluator_hash: str
    holdout_hash: str
    evaluator_report: dict
    blockers: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not self.blockers


def eligible_run_rows() -> list[dict]:
    return [dict(row) for row in db.fetchall(
        """
        SELECT er.id, er.agenda_id, er.deep_insight_id, er.status,
               er.baseline_metric_value, er.best_metric_value, er.effect_pct,
               er.hypothesis_verdict, er.experiment_suite, er.created_at
        FROM experiment_runs er
        WHERE er.status='completed'
          AND er.agenda_id IS NOT NULL
          AND er.baseline_metric_value IS NOT NULL
          AND er.baseline_metric_value <> 0
          AND er.best_metric_value IS NOT NULL
          AND COALESCE(er.scientific_evidence_state, 'planned')
              IN ('', 'planned')
          AND NOT EXISTS (
              SELECT 1 FROM scientific_decision_records sdr
              WHERE sdr.experiment_run_id = er.id
          )
        ORDER BY er.id
        """
    )]


def _evaluator_source_hash() -> str:
    return _sha256_text(Path(__file__).read_text(encoding="utf-8"))


def _metrics_check(run: dict) -> dict:
    """Deterministic re-check of the run's recorded metrics.

    This is the retrospective 'independent evaluation': it does not re-run the
    experiment, it verifies internal consistency of the persisted evidence and
    says so explicitly.
    """
    baseline = float(run.get("baseline_metric_value") or 0)
    best = float(run.get("best_metric_value") or 0)
    effect_pct = run.get("effect_pct")
    checks = {
        "baseline_nonzero": baseline != 0,
        "best_present": best is not None,
        "effect_consistent": True,
    }
    if effect_pct is not None and baseline:
        derived = abs((best - baseline) / baseline) * 100.0
        checks["effect_consistent"] = abs(derived - abs(float(effect_pct))) < 0.51
    return {
        "evaluator": EVALUATOR_REF,
        "checks": checks,
        "passed": all(checks.values()),
        "note": "consistency re-check of persisted metrics; not a re-execution",
    }


def collect_evidence(run_id: int) -> RunEvidence:
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        raise RetrospectiveReviewError(f"run {run_id} does not exist")
    run = dict(run)
    blockers: list[str] = []
    if run.get("status") != "completed":
        blockers.append("run_not_completed")
    if not run.get("agenda_id"):
        blockers.append("run_not_agenda_scoped")
    baseline = run.get("baseline_metric_value")
    if baseline in (None, 0):
        blockers.append("baseline_missing_or_zero")
    if run.get("best_metric_value") is None:
        blockers.append("best_metric_missing")
    if db.fetchone(
        "SELECT 1 as x FROM scientific_decision_records WHERE experiment_run_id=?",
        (run_id,),
    ):
        blockers.append("already_decided")
    state = str(run.get("scientific_evidence_state") or "planned") or "planned"
    if state not in ("", "planned"):
        blockers.append(f"ladder_already_at_{state}")

    claims = [dict(row) for row in db.fetchall(
        "SELECT id, claim_text, verdict, created_at FROM experimental_claims WHERE run_id=? ORDER BY id",
        (run_id,),
    )]
    artifacts = [dict(row) for row in db.fetchall(
        "SELECT id, artifact_type, path, metric_key, metric_value FROM experiment_artifacts WHERE run_id=? ORDER BY id",
        (run_id,),
    )]
    if not artifacts:
        blockers.append("no_artifacts_recorded")

    digest = hashlib.sha256()
    present = missing = 0
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or ""))
        digest.update(_canonical_json(
            {"id": artifact["id"], "type": artifact.get("artifact_type"),
             "path_name": path.name}
        ).encode("utf-8"))
        try:
            if path.is_file():
                digest.update(path.read_bytes())
                present += 1
            else:
                digest.update(b"<missing>")
                missing += 1
        except OSError:
            digest.update(b"<unreadable>")
            missing += 1
    if artifacts and present == 0:
        blockers.append("no_artifact_files_on_disk")

    evaluator_report = _metrics_check(run)
    if not evaluator_report["passed"]:
        blockers.append("metrics_consistency_check_failed")

    return RunEvidence(
        run=run,
        claims=claims,
        artifacts=artifacts,
        artifact_files_present=present,
        artifact_files_missing=missing,
        raw_artifacts_hash=digest.hexdigest(),
        claim_ledger_hash=_sha256_text(_canonical_json(claims)),
        benchmark_contract_hash=_sha256_text(_canonical_json({
            "suite": run.get("experiment_suite"),
            "baseline_metric_value": run.get("baseline_metric_value"),
            "best_metric_value": run.get("best_metric_value"),
            "effect_pct": run.get("effect_pct"),
            "reconstructed": True,
        })),
        evaluator_hash=_evaluator_source_hash(),
        holdout_hash=_sha256_text(HOLDOUT_WAIVER_TEXT),
        evaluator_report=evaluator_report,
        blockers=blockers,
    )


def build_packet(run_id: int) -> dict:
    """One-page review packet for the human reviewer."""
    evidence = collect_evidence(run_id)
    run = evidence.run
    insight = db.fetchone(
        "SELECT id, title, tier FROM deep_insights WHERE id=?",
        (run.get("deep_insight_id"),),
    )
    return {
        "run_id": run["id"],
        "agenda_id": run.get("agenda_id"),
        "idea": dict(insight) if insight else None,
        "status": run.get("status"),
        "metrics": {
            "baseline": run.get("baseline_metric_value"),
            "best": run.get("best_metric_value"),
            "effect_pct": run.get("effect_pct"),
            "historical_informal_verdict": run.get("hypothesis_verdict"),
        },
        "claims": [
            {"id": c["id"], "text": c.get("claim_text"), "verdict": c.get("verdict")}
            for c in evidence.claims
        ],
        "artifacts": {
            "recorded": len(evidence.artifacts),
            "files_present": evidence.artifact_files_present,
            "files_missing": evidence.artifact_files_missing,
        },
        "hashes": {
            "raw_artifacts": evidence.raw_artifacts_hash,
            "claim_ledger": evidence.claim_ledger_hash,
            "benchmark_contract": evidence.benchmark_contract_hash,
            "evaluator": evidence.evaluator_hash,
            "holdout_waiver": evidence.holdout_hash,
        },
        "evaluator_report": evidence.evaluator_report,
        "policy": {
            "verdict_ceiling": "inconclusive",
            "reason": HOLDOUT_WAIVER_TEXT,
        },
        "blockers": evidence.blockers,
        "subject": retrospective_subject(
            agenda_id=int(run.get("agenda_id") or 0),
            experiment_run_id=int(run["id"]),
        ),
    }


def sign_approval(
    *, reviewer_id: str, key_id: str, subject: str, secret: str,
    issued_at: str | None = None,
) -> ReviewerApproval:
    """Create a signed approval envelope. Possession of the reviewer secret is
    the authorization; verification happens again inside apply_review."""
    envelope = ReviewerApproval(
        reviewer_id=reviewer_id,
        key_id=key_id,
        purpose=PURPOSE,
        subject=subject,
        issued_at=issued_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        signature="",
    )
    signature = hmac_lib.new(
        secret.encode("utf-8"), envelope.signing_payload(), hashlib.sha256
    ).hexdigest()
    return ReviewerApproval(
        reviewer_id=reviewer_id,
        key_id=key_id,
        purpose=PURPOSE,
        subject=subject,
        issued_at=envelope.issued_at,
        signature=signature,
    )


def apply_review(
    *,
    run_id: int,
    approval: ReviewerApproval | dict,
    verdict: str = "inconclusive",
) -> dict:
    """Walk one eligible historical run up the ladder to a capped decision.

    Writes, in one transaction: four evidence_state_transitions (each marked
    legacy_review), one evidence_audit_records row, one
    scientific_decision_records row, one reviewer_approval_records row, and
    the run's scientific_evidence_state.
    """
    if verdict != "inconclusive":
        raise RetrospectiveReviewError(
            "retrospective verdicts are capped at 'inconclusive' (no holdout)"
        )
    evidence = collect_evidence(run_id)
    if evidence.blockers:
        raise RetrospectiveReviewError(
            "run not eligible: " + ",".join(evidence.blockers)
        )
    run = evidence.run
    agenda_id = int(run["agenda_id"])
    subject = retrospective_subject(agenda_id=agenda_id, experiment_run_id=run_id)
    envelope = ReviewerApprovalVerifier.from_environment().verify(
        approval, purpose=PURPOSE, subject=subject
    )
    actor = f"retrospective_review:{envelope.reviewer_id}"
    verdict_hash = _sha256_text(_canonical_json({
        "run_id": run_id,
        "verdict": verdict,
        "raw_artifacts_hash": evidence.raw_artifacts_hash,
        "claim_ledger_hash": evidence.claim_ledger_hash,
        "policy": HOLDOUT_REF,
    }))

    base_context = {
        "legacy_review": True,
        "reviewer": envelope.public_record(),
        "holdout_waiver": HOLDOUT_REF,
        "raw_artifacts_hash": evidence.raw_artifacts_hash,
        "claim_ledger_hash": evidence.claim_ledger_hash,
        "benchmark_contract_hash": evidence.benchmark_contract_hash,
        "evaluator_ref": EVALUATOR_REF,
        "evaluator_hash": evidence.evaluator_hash,
        "holdout_ref": HOLDOUT_REF,
        "holdout_hash": evidence.holdout_hash,
    }
    try:
        current = "planned"
        for target in LADDER:
            context = dict(base_context)
            if target == "scientifically_decided":
                context["verdict"] = verdict
                context["verdict_hash"] = verdict_hash
            db.execute(
                """
                INSERT INTO evidence_state_transitions
                    (agenda_id, experiment_run_id, from_state, to_state, actor,
                     context_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (agenda_id, run_id, current, target, actor,
                 _canonical_json(context)),
            )
            current = target
        db.execute(
            """
            INSERT INTO evidence_audit_records
                (agenda_id, experiment_run_id, raw_artifacts_hash,
                 claim_ledger_hash, benchmark_contract_hash, evaluator_ref,
                 evaluator_hash, holdout_ref, holdout_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (agenda_id, run_id, evidence.raw_artifacts_hash,
             evidence.claim_ledger_hash, evidence.benchmark_contract_hash,
             EVALUATOR_REF, evidence.evaluator_hash, HOLDOUT_REF,
             evidence.holdout_hash),
        )
        db.execute(
            """
            INSERT INTO reviewer_approval_records
                (agenda_id, purpose, subject, reviewer_id, key_id, issued_at,
                 signature_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (agenda_id, PURPOSE, subject, envelope.reviewer_id,
             envelope.key_id, envelope.issued_at, envelope.signature_hash()),
        )
        db.execute(
            """
            INSERT INTO scientific_decision_records
                (agenda_id, experiment_run_id, verdict, verdict_hash,
                 evidence_decision_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agenda_id, run_id, verdict, verdict_hash,
             _canonical_json({
                 "path": "retrospective_review",
                 "evaluator_report": evidence.evaluator_report,
                 "holdout": HOLDOUT_REF,
             })),
        )
        db.execute(
            "UPDATE experiment_runs SET scientific_evidence_state=? "
            "WHERE id=? AND agenda_id=?",
            ("scientifically_decided", run_id, agenda_id),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {
        "run_id": run_id,
        "agenda_id": agenda_id,
        "verdict": verdict,
        "verdict_hash": verdict_hash,
        "actor": actor,
    }
