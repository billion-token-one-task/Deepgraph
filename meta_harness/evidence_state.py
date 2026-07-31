"""The single scientific evidence state machine for meta-harness-v1."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from contracts.meta_harness import EVIDENCE_STATES


class EvidenceTransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceTransitionContext:
    resource_grant_valid: bool = False
    resource_grant_id: int | None = None
    execution_succeeded: bool = False
    pilot_only: bool = False
    raw_artifacts_present: bool = False
    claim_ledger_present: bool = False
    full_benchmark_complete: bool = False
    evaluator_passed: bool = False
    holdout_passed: bool = False
    verdict: str = "inconclusive"
    reviewer_approved: bool = False
    raw_artifacts_hash: str = ""
    claim_ledger_hash: str = ""
    benchmark_contract_hash: str = ""
    evaluator_ref: str = ""
    evaluator_hash: str = ""
    holdout_ref: str = ""
    holdout_hash: str = ""
    verdict_hash: str = ""
    evidence_decision_passed: bool = False
    p_value: float | None = None
    metric_value: float | None = None
    baseline_value: float | None = None
    alpha: float = 0.05
    blockers: tuple[str, ...] = field(default_factory=tuple)


_CONTENT_HASH = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _require_content_hash(
    blockers: list[str],
    value: str,
    *,
    name: str,
) -> None:
    if not _CONTENT_HASH.fullmatch(str(value or "").strip().lower()):
        blockers.append(f"{name}_missing_or_invalid")


def advance(
    current: str,
    target: str,
    context: EvidenceTransitionContext,
) -> str:
    if current not in EVIDENCE_STATES or target not in EVIDENCE_STATES:
        raise EvidenceTransitionError("unknown_evidence_state")
    current_index = EVIDENCE_STATES.index(current)
    target_index = EVIDENCE_STATES.index(target)
    if target_index != current_index + 1:
        raise EvidenceTransitionError("only_single_forward_transition_allowed")
    blockers = list(context.blockers)
    if not context.resource_grant_valid:
        blockers.append("resource_grant_invalid")
    if int(context.resource_grant_id or 0) <= 0:
        blockers.append("resource_grant_id_missing")
    if not context.execution_succeeded:
        blockers.append("execution_not_successful")
    if target == "sanity_passed":
        if not context.raw_artifacts_present:
            blockers.append("raw_artifacts_missing")
        _require_content_hash(
            blockers,
            context.raw_artifacts_hash,
            name="raw_artifacts_hash",
        )
    if target == "full_benchmark_complete":
        if context.pilot_only:
            blockers.append("pilot_cannot_complete_full_benchmark")
        if not context.full_benchmark_complete:
            blockers.append("full_benchmark_incomplete")
        _require_content_hash(
            blockers,
            context.benchmark_contract_hash,
            name="benchmark_contract_hash",
        )
    if target == "evidence_audited":
        if not context.claim_ledger_present:
            blockers.append("claim_ledger_missing")
        if not context.evaluator_passed or not context.holdout_passed:
            blockers.append("independent_evaluation_incomplete")
        for name, value in (
            ("raw_artifacts_hash", context.raw_artifacts_hash),
            ("claim_ledger_hash", context.claim_ledger_hash),
            ("benchmark_contract_hash", context.benchmark_contract_hash),
            ("evaluator_hash", context.evaluator_hash),
            ("holdout_hash", context.holdout_hash),
        ):
            _require_content_hash(blockers, value, name=name)
        if not context.evaluator_ref.strip():
            blockers.append("evaluator_ref_missing")
        if not context.holdout_ref.strip():
            blockers.append("holdout_ref_missing")
    if target == "scientifically_decided":
        if context.verdict not in {
            "supported",
            "refuted",
            "inconclusive",
        }:
            blockers.append("scientific_verdict_missing")
        _require_content_hash(
            blockers,
            context.verdict_hash,
            name="verdict_hash",
        )
        for name, value in (
            ("raw_artifacts_hash", context.raw_artifacts_hash),
            ("claim_ledger_hash", context.claim_ledger_hash),
            ("benchmark_contract_hash", context.benchmark_contract_hash),
            ("evaluator_hash", context.evaluator_hash),
            ("holdout_hash", context.holdout_hash),
        ):
            _require_content_hash(blockers, value, name=name)
        if not context.evaluator_ref.strip():
            blockers.append("evaluator_ref_missing")
        if not context.holdout_ref.strip():
            blockers.append("holdout_ref_missing")
        if context.verdict == "supported" and not context.evidence_decision_passed:
            blockers.append("positive_evidence_decision_failed")
    if target == "manuscript_allowed":
        if context.verdict != "supported":
            blockers.append("positive_manuscript_requires_supported_verdict")
        if not context.reviewer_approved:
            blockers.append("reviewer_approval_required")
        _require_content_hash(
            blockers,
            context.verdict_hash,
            name="verdict_hash",
        )
    if blockers:
        raise EvidenceTransitionError(",".join(dict.fromkeys(blockers)))
    return target
