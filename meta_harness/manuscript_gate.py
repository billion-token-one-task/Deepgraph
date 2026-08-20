"""Drive the last ladder rung: scientifically_decided -> manuscript_allowed.

That rung had no executor. `advance_experiment_state(target="manuscript_allowed")`
was never called anywhere in production, the only `sign_approval` in the tree is
bound to `PURPOSE = "retrospective_review"` and reachable only from an
operator-run script, and no run has ever reached the state. The verifier was
fully built and waiting for an input nobody produced -- so V1's success terminal
state was unreachable no matter what the experiments found.

This module produces that input. It is deliberately the smallest thing that can:

* It reuses the run's existing active grant rather than issuing a new stage.
  A separate manuscript grant would take another concurrency slot at the agenda
  cap and would be a fresh source of the superseded-shell-grant hazard that
  still has no settling path. The audit grant is already active at exactly this
  point in the run's life, which is the window this gate needs.

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

from agents.llm_client import call_llm_for_role, parse_llm_json_text
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
MANUSCRIPT_PROMPT_REF = "manuscript_gate_reviewer_v1"
# Same shape and reasoning as the audit evaluator's ceiling: the reviewer
# reasons before it answers and that reasoning is output tokens, so a ceiling
# small enough to truncate the JSON turns a judgement into an unparseable
# answer. The audit evaluator settled at exactly its 4096 reservation three
# times running before that cap was retired (2026-08-20).
MANUSCRIPT_REVIEW_MAX_TOKENS = 16384
MAX_MANUSCRIPT_REVIEW_ATTEMPTS = 3


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


def _review_attempts(resource_grant_id: int) -> int:
    row = db.fetchone(
        """
        SELECT COUNT(*) AS n
        FROM llm_route_observations
        WHERE role='reviewer' AND prompt_version=?
          AND grant_usage_reservation_id IN (
            SELECT id FROM resource_grant_usage_reservations
            WHERE resource_grant_id=?
          )
        """,
        (MANUSCRIPT_PROMPT_REF, int(resource_grant_id)),
    )
    return int(dict(row or {}).get("n") or 0)


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
    decision: Mapping[str, Any],
    ledger: Mapping[str, Any],
    holdout: Mapping[str, Any] | None,
    verdict_hash: str,
) -> dict[str, Any]:
    """Ask the reviewer role. Raises rather than guessing on any failure."""
    attempt = _review_attempts(resource_grant_id)
    if attempt >= MAX_MANUSCRIPT_REVIEW_ATTEMPTS:
        raise ManuscriptGateError("manuscript review attempts exhausted")
    prompt = build_review_prompt(decision=decision, ledger=ledger, holdout=holdout)
    raw, _tokens, route = call_llm_for_role(
        "Review whether measured evidence justifies a manuscript. "
        "Judge only from the numbers provided.",
        prompt,
        agenda_id=agenda_id,
        idea_id=idea_id,
        role="reviewer",
        stage="manuscript_gate",
        resource_grant_id=resource_grant_id,
        operation="manuscript_gate_review",
        # Bound to the verdict hash: a re-review is only meaningful when the
        # evidence changed, and the attempt suffix keeps a retry after a lost
        # answer from colliding with a settled reservation (the failure that
        # stranded run 191's audit, 2026-08-20).
        idempotency_key=(
            f"manuscript-gate:{agenda_id}:{idea_id}:"
            f"{str(verdict_hash)[:16]}:{attempt}"
        ),
        prompt_version=MANUSCRIPT_PROMPT_REF,
        max_tokens=MANUSCRIPT_REVIEW_MAX_TOKENS,
    )
    parsed, _how = parse_llm_json_text(raw)
    if not isinstance(parsed, dict) or "concur" not in parsed:
        raise ManuscriptGateError("manuscript reviewer returned no judgement")
    return {
        "judgement": parsed,
        "reviewer_ref": f"{route.get('provider')}:{route.get('model')}",
        "reviewer_hash": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(),
        "prompt_ref": MANUSCRIPT_PROMPT_REF,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


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

    results = Path(str(run["workdir"])) / "results"
    ledger = _load_json(results / "claim_ledger.json")
    if ledger is None:
        log(f"[MANUSCRIPT] run {run_id} claim ledger unreadable; not advancing")
        return "no_ledger"
    holdout = _load_json(
        Path(str(run["workdir"])) / "results_holdout" / "final_results.json"
    )

    try:
        review = review_manuscript_readiness(
            agenda_id=agenda_id,
            idea_id=idea_id,
            resource_grant_id=grant_id,
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
        return "review_failed"

    judgement = review["judgement"]
    if not bool(judgement.get("concur")):
        log(
            f"[MANUSCRIPT] run {run_id} REFUSED by {review['reviewer_ref']}: "
            f"{judgement.get('reasons')}"
        )
        return "refused"

    subject = scientific_manuscript_subject(
        agenda_id=agenda_id, experiment_run_id=run_id, verdict_hash=verdict_hash
    )
    approval = sign_manuscript_approval(subject=subject, secret=secret)
    context = EvidenceTransitionContext(
        verdict="supported",
        verdict_hash=verdict_hash,
        # The envelope the verifier re-signs, not public_record() -- that one
        # carries signature_hash instead of the signature and would verify as
        # an incomplete envelope.
        reviewer_approval={
            "reviewer_id": approval.reviewer_id,
            "key_id": approval.key_id,
            "purpose": approval.purpose,
            "subject": approval.subject,
            "issued_at": approval.issued_at,
            "signature": approval.signature,
        },
    )
    MetaHarnessRepository().advance_experiment_state(
        agenda_id=agenda_id,
        experiment_run_id=run_id,
        target="manuscript_allowed",
        context=context,
        # The verifier requires actor == the signed reviewer id, so the audit
        # trail names the AI reviewer rather than an operator.
        actor=MANUSCRIPT_REVIEWER_ID,
    )
    log(
        f"[MANUSCRIPT] run {run_id} manuscript_allowed, approved by "
        f"{MANUSCRIPT_REVIEWER_ID} via {review['reviewer_ref']}"
    )
    return "manuscript_allowed"
