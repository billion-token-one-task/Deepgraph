"""Drive the last ladder rung: scientifically_decided -> manuscript_allowed.

That rung had no executor. `advance_experiment_state(target="manuscript_allowed")`
was never called anywhere in production, the only `sign_approval` in the tree is
bound to `PURPOSE = "retrospective_review"` and reachable only from an
operator-run script, and no run has ever reached the state. The verifier was
fully built and waiting for an input nobody produced -- so V1's success terminal
state was unreachable no matter what the experiments found.

This module produces that input. It is deliberately the smallest thing that can:

* A live evidence audit funds its review before that grant settles. Historical
  recovery instead uses a dedicated, token-only ``manuscript`` grant, so it
  never has to reopen a consumed audit grant or rewrite its outcome.

* It asks a DIFFERENT question from the evidence audit. The audit asks whether
  the measurement supports the verdict, and it has already passed by the time we
  get here; asking it again would be the same judge twice and the second gate
  would be worth nothing. This gate asks whether the evidence is sufficient to
  justify writing a paper at all -- sample size, holdout agreement, effect
  size against the baseline, and whether anything in the ledger is missing.

* It fails closed everywhere. No concurrence, an unparseable answer, a missing
  ledger, a mismatched verdict hash, an exhausted attempt budget: the run stays
  at scientifically_decided. Nothing here can move a run forward except an
  explicit `concur: true` against a complete ledger.

The signature is provenance, not authorization: this service holds the signing
secret, so an AI-minted approval proves which reviewer identity and which
evidence produced it, not that a human agreed. The identity is deliberately
distinct (`ai-reviewer-v1`, its own key id) so that an AI approval and an
operator approval are distinguishable in `reviewer_approval_records` forever.
The intended operating model is AI-approves plus human-audits-and-revokes, not
"there is a signature, therefore it is trustworthy".
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agents.llm_client import (
    call_llm_for_role,
    configured_role_prompt_version,
    parse_llm_json_text,
)
from db import database as db
from meta_harness.evidence_state import EvidenceTransitionContext
from meta_harness.repository import MetaHarnessRepository
from meta_harness.reviewer_approval import (
    ReviewerApproval,
    scientific_manuscript_subject,
)

# Registered in docs/internal/V1_SCAFFOLD_REGISTER.md.
MANUSCRIPT_REVIEWER_ID = "ai-reviewer-v1"
MANUSCRIPT_REVIEWER_KEY_ID = "ai-reviewer-20260820"
MANUSCRIPT_REVIEWER_SECRET_ENV = "DEEPGRAPH_REVIEWER_SECRET_AI_REVIEWER_20260820"
MANUSCRIPT_PURPOSE = "scientific_manuscript"
# Same shape and reasoning as the audit evaluator's ceiling: the reviewer
# reasons before it answers and that reasoning is output tokens, so a ceiling
# small enough to truncate the JSON turns a judgement into an unparseable
# answer. The audit evaluator settled at exactly its 4096 reservation three
# times running before that cap was retired (2026-08-20).
MANUSCRIPT_REVIEW_MAX_TOKENS = 16384
MAX_MANUSCRIPT_REVIEW_ATTEMPTS = 2


class ManuscriptGateError(RuntimeError):
    """The gate could not reach a decision. The run does not advance."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sign_manuscript_approval(
    *,
    subject: str,
    secret: str,
    reviewer_id: str = MANUSCRIPT_REVIEWER_ID,
    key_id: str = MANUSCRIPT_REVIEWER_KEY_ID,
    issued_at: str | None = None,
) -> ReviewerApproval:
    """Mint a manuscript approval. Possession of the secret is the authority.

    Separate from retrospective_review.sign_approval, which hard-binds
    PURPOSE = "retrospective_review" and therefore cannot express this one.
    """
    envelope = ReviewerApproval(
        reviewer_id=reviewer_id,
        key_id=key_id,
        purpose=MANUSCRIPT_PURPOSE,
        subject=subject,
        issued_at=issued_at or _utc_now(),
        signature="",
    )
    signature = hmac_lib.new(
        secret.encode("utf-8"), envelope.signing_payload(), hashlib.sha256
    ).hexdigest()
    return ReviewerApproval(
        reviewer_id=reviewer_id,
        key_id=key_id,
        purpose=MANUSCRIPT_PURPOSE,
        subject=subject,
        issued_at=envelope.issued_at,
        signature=signature,
    )


def build_review_prompt(
    *,
    decision: Mapping[str, Any],
    ledger: Mapping[str, Any],
    holdout: Mapping[str, Any] | None,
) -> str:
    """The gate's question, stated so the answer is checkable against numbers."""
    return (
        "A completed experiment has been adjudicated 'supported' by an "
        "independent evidence audit. You are a SEPARATE reviewer, and you are "
        "deciding one thing only: is this evidence sufficient to justify "
        "writing a paper about it?\n\n"
        "You are NOT re-judging whether the measurement supports the verdict; "
        "that has already been audited. Judge sufficiency and completeness:\n"
        "  - Is the sample large enough for the effect claimed?\n"
        "  - Does the held-out result agree with the main result?\n"
        "  - Is the effect meaningful against the baseline, not merely "
        "significant?\n"
        "  - Is anything missing from the ledger that a reviewer would demand?\n\n"
        "Refusing is a normal outcome and costs nothing. Approving evidence "
        "that cannot support a paper is the expensive mistake. Dissent freely; "
        "your concurrence is not assumed.\n\n"
        f"DECISION RECORD:\n{json.dumps(dict(decision), indent=2, sort_keys=True, default=str)}\n\n"
        f"CLAIM LEDGER:\n{json.dumps(dict(ledger), indent=2, sort_keys=True, default=str)}\n\n"
        f"HELD-OUT RESULT:\n{json.dumps(dict(holdout or {}), indent=2, sort_keys=True, default=str)}\n\n"
        'Answer with one JSON object only: {"concur": true|false, '
        '"reasons": ["..."]}'
    )


def review_manuscript_readiness(
    *,
    agenda_id: int,
    idea_id: int,
    resource_grant_id: int,
    grant_stage: str,
    attempt_key: str,
    prompt_ref: str,
    decision: Mapping[str, Any],
    ledger: Mapping[str, Any],
    holdout: Mapping[str, Any] | None,
    verdict_hash: str,
) -> dict[str, Any]:
    """Ask the reviewer role. Raises rather than guessing on any failure."""
    prompt = build_review_prompt(decision=decision, ledger=ledger, holdout=holdout)
    raw, _tokens, route = call_llm_for_role(
        "Review whether measured evidence justifies a manuscript. "
        "Judge only from the numbers provided.",
        prompt,
        agenda_id=agenda_id,
        idea_id=idea_id,
        role="reviewer",
        stage=str(grant_stage),
        resource_grant_id=resource_grant_id,
        operation="manuscript_gate_review",
        # The append-only attempt record was committed before this call.  The
        # same key therefore identifies both the logical attempt and any real
        # usage reservation, without fabricating usage when routing fails.
        idempotency_key=attempt_key,
        prompt_version=prompt_ref,
        max_tokens=MANUSCRIPT_REVIEW_MAX_TOKENS,
    )
    parsed, _how = parse_llm_json_text(raw)
    if not isinstance(parsed, dict) or "concur" not in parsed:
        raise ManuscriptGateError("manuscript reviewer returned no judgement")
    usage = db.fetchone(
        """
        SELECT id FROM resource_grant_usage_reservations
        WHERE resource_grant_id=? AND idempotency_key=?
          AND operation='manuscript_gate_review' AND status='settled'
        """,
        (int(resource_grant_id), attempt_key),
    )
    if not usage:
        raise ManuscriptGateError("manuscript reviewer usage was not settled")
    return {
        "judgement": parsed,
        "reviewer_ref": f"{route.get('provider')}:{route.get('model')}",
        "reviewer_hash": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(),
        "prompt_ref": prompt_ref,
        "grant_usage_reservation_id": int(usage["id"]),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _finish_terminal_result(
    *,
    repo: MetaHarnessRepository,
    run: Mapping[str, Any],
    terminal: Mapping[str, Any],
    verdict_hash: str,
    secret: str,
    log,
    record_kwargs: dict[str, Any] | None = None,
) -> str:
    """Persist and finish one terminal result in a single transaction.

    ``record_kwargs`` is present for a new result.  A cached terminal omits it
    and uses this same transaction to resume any transition/settlement that a
    prior process did not finish.
    """

    agenda_id = int(run["agenda_id"])
    run_id = int(run["id"])
    grant_id = int(
        terminal.get("resource_grant_id") or run.get("resource_grant_id") or 0
    )
    disposition = str(terminal.get("disposition") or "")
    if disposition not in {"approved", "refused", "technical_failed"}:
        raise ManuscriptGateError("unknown manuscript terminal disposition")

    try:
        if record_kwargs is not None:
            repo.record_manuscript_gate_result(commit=False, **record_kwargs)

        current_run = db.fetchone(
            """
            SELECT resource_grant_id, scientific_evidence_state
            FROM experiment_runs WHERE id=? AND agenda_id=?
            """,
            (run_id, agenda_id),
        )
        if (
            not current_run
            or int(current_run.get("resource_grant_id") or 0) != grant_id
        ):
            raise ManuscriptGateError("manuscript terminal grant is not run-scoped")
        state = str(current_run.get("scientific_evidence_state") or "")

        if disposition == "approved":
            if state == "scientifically_decided":
                if not str(secret or "").strip():
                    raise ManuscriptGateError(
                        "approved manuscript result cannot resume without signing secret"
                    )
                subject = scientific_manuscript_subject(
                    agenda_id=agenda_id,
                    experiment_run_id=run_id,
                    verdict_hash=verdict_hash,
                )
                approval = sign_manuscript_approval(subject=subject, secret=secret)
                context = EvidenceTransitionContext(
                    resource_grant_valid=True,
                    resource_grant_id=grant_id,
                    execution_succeeded=True,
                    verdict="supported",
                    verdict_hash=verdict_hash,
                    # The verifier needs the signed envelope, not public_record().
                    reviewer_approval={
                        "reviewer_id": approval.reviewer_id,
                        "key_id": approval.key_id,
                        "purpose": approval.purpose,
                        "subject": approval.subject,
                        "issued_at": approval.issued_at,
                        "signature": approval.signature,
                    },
                )
                repo.advance_experiment_state(
                    agenda_id=agenda_id,
                    experiment_run_id=run_id,
                    target="manuscript_allowed",
                    context=context,
                    actor=MANUSCRIPT_REVIEWER_ID,
                    commit=False,
                )
            elif state != "manuscript_allowed":
                raise ManuscriptGateError(
                    "approved manuscript record has an incompatible run state"
                )
            status = "manuscript_allowed"
        else:
            if state != "scientifically_decided":
                raise ManuscriptGateError(
                    "non-approval manuscript record has an incompatible run state"
                )
            status = disposition

        grant = db.fetchone(
            "SELECT stage FROM resource_grants WHERE id=? AND agenda_id=?",
            (grant_id, agenda_id),
        )
        grant_stage = str(dict(grant or {}).get("stage") or "")
        if grant_stage == "manuscript":
            used = repo.complete_manuscript_grant(
                agenda_id=agenda_id,
                experiment_run_id=run_id,
                resource_grant_id=grant_id,
                commit=False,
            )
            log(
                f"[MANUSCRIPT] run {run_id} terminal={status}; "
                f"grant {grant_id} settled with {used} tokens"
            )
        elif grant_stage != "evidence_audit":
            raise ManuscriptGateError(
                "manuscript gate used an invalid grant stage"
            )
        db.commit()
        return status
    except Exception:
        db.rollback()
        raise


def _finish_technical_failure(
    *,
    repo: MetaHarnessRepository,
    run: Mapping[str, Any],
    verdict_hash: str,
    prompt_ref: str,
    failure: Exception,
    secret: str,
    log,
) -> str:
    """Return retryable failure, or atomically close the second attempt."""

    grant_id = int(run.get("resource_grant_id") or 0)
    if repo.count_manuscript_gate_attempts(
        resource_grant_id=grant_id
    ) < MAX_MANUSCRIPT_REVIEW_ATTEMPTS:
        return "review_failed"
    terminal = {
        "disposition": "technical_failed",
        "resource_grant_id": grant_id,
    }
    return _finish_terminal_result(
        repo=repo,
        run=run,
        terminal=terminal,
        verdict_hash=verdict_hash,
        secret=secret,
        log=log,
        record_kwargs={
            "agenda_id": int(run["agenda_id"]),
            "idea_id": int(run["deep_insight_id"]),
            "experiment_run_id": int(run["id"]),
            "resource_grant_id": grant_id,
            "verdict_hash": verdict_hash,
            "disposition": "technical_failed",
            "prompt_ref": prompt_ref,
            "failure_reason": f"{type(failure).__name__}: {failure}"[:1000],
        },
    )


def run_manuscript_gate(
    run: Mapping[str, Any],
    *,
    secret: str,
    log=print,
) -> str:
    """Decide whether one supported run may proceed to a manuscript.

    Returns a status string. Every path that is not an explicit concurrence
    against a complete ledger leaves the run at scientifically_decided.
    """
    agenda_id = int(run["agenda_id"])
    run_id = int(run["id"])
    idea_id = int(run["deep_insight_id"])
    grant_id = int(run.get("resource_grant_id") or 0)
    if not grant_id:
        return "no_grant"

    decision_row = db.fetchone(
        """
        SELECT verdict, verdict_hash, evidence_decision_json
        FROM scientific_decision_records
        WHERE agenda_id=? AND experiment_run_id=?
        ORDER BY id DESC LIMIT 1
        """,
        (agenda_id, run_id),
    )
    decision = dict(decision_row or {})
    if str(decision.get("verdict") or "") != "supported":
        # Not an error: this is the normal state of every run so far.
        return "not_supported"
    verdict_hash = str(decision.get("verdict_hash") or "")
    if not verdict_hash:
        log(f"[MANUSCRIPT] run {run_id} has no verdict hash; not advancing")
        return "no_verdict_hash"

    repo = MetaHarnessRepository()
    terminal = repo.load_manuscript_gate_record(
        agenda_id=agenda_id,
        experiment_run_id=run_id,
        verdict_hash=verdict_hash,
    )
    if terminal:
        return _finish_terminal_result(
            repo=repo,
            run=run,
            terminal=terminal,
            verdict_hash=verdict_hash,
            secret=secret,
            log=log,
        )

    prompt_error: Exception | None = None
    try:
        prompt_ref = configured_role_prompt_version("reviewer")
    except Exception as exc:
        # Still record the logical attempt.  No usage is claimed, and the
        # terminal reason retains the configuration failure if it repeats.
        prompt_ref = "reviewer:configuration-unavailable"
        prompt_error = exc

    try:
        attempt = repo.begin_manuscript_gate_attempt(
            agenda_id=agenda_id,
            idea_id=idea_id,
            experiment_run_id=run_id,
            resource_grant_id=grant_id,
            verdict_hash=verdict_hash,
            prompt_ref=prompt_ref,
            max_attempts=MAX_MANUSCRIPT_REVIEW_ATTEMPTS,
        )
    except Exception as exc:
        log(
            f"[MANUSCRIPT] run {run_id} could not begin review attempt: "
            f"{type(exc).__name__}: {exc}"
        )
        return _finish_technical_failure(
            repo=repo,
            run=run,
            verdict_hash=verdict_hash,
            prompt_ref=prompt_ref,
            failure=exc,
            secret=secret,
            log=log,
        )

    try:
        if prompt_error is not None:
            raise prompt_error
        if not str(secret or "").strip():
            raise ManuscriptGateError("manuscript reviewer signing secret is absent")
        results = Path(str(run["workdir"])) / "results"
        ledger = _load_json(results / "claim_ledger.json")
        if ledger is None:
            raise ManuscriptGateError("claim ledger is unreadable")
        holdout = _load_json(
            Path(str(run["workdir"]))
            / "results_holdout"
            / "final_results.json"
        )
        review = review_manuscript_readiness(
            agenda_id=agenda_id,
            idea_id=idea_id,
            resource_grant_id=grant_id,
            grant_stage=str(attempt["grant_stage"]),
            attempt_key=str(attempt["idempotency_key"]),
            prompt_ref=prompt_ref,
            decision=decision,
            ledger=ledger,
            holdout=holdout,
            verdict_hash=verdict_hash,
        )
    except Exception as exc:
        # Loud, and the run stays put. An unreachable reviewer must never
        # read as approval.
        log(
            f"[MANUSCRIPT] run {run_id} review failed, staying at "
            f"scientifically_decided: {type(exc).__name__}: {exc}"
        )
        return _finish_technical_failure(
            repo=repo,
            run=run,
            verdict_hash=verdict_hash,
            prompt_ref=prompt_ref,
            failure=exc,
            secret=secret,
            log=log,
        )

    judgement = review["judgement"]
    disposition = "approved" if bool(judgement.get("concur")) else "refused"
    if disposition == "refused":
        log(
            f"[MANUSCRIPT] run {run_id} REFUSED by {review['reviewer_ref']}: "
            f"{judgement.get('reasons')}"
        )
    terminal = {"disposition": disposition, "resource_grant_id": grant_id}
    return _finish_terminal_result(
        repo=repo,
        run=run,
        terminal=terminal,
        verdict_hash=verdict_hash,
        secret=secret,
        log=log,
        record_kwargs={
            "agenda_id": agenda_id,
            "idea_id": idea_id,
            "experiment_run_id": run_id,
            "resource_grant_id": grant_id,
            "verdict_hash": verdict_hash,
            "disposition": disposition,
            "prompt_ref": str(review["prompt_ref"]),
            "judgement": dict(judgement),
            "grant_usage_reservation_id": int(
                review["grant_usage_reservation_id"]
            ),
            "reviewer_ref": str(review["reviewer_ref"]),
            "reviewer_response_hash": str(review["reviewer_hash"]),
        },
    )
