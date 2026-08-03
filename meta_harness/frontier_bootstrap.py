"""Run exactly one authority-scoped Frontier evaluation.

This is the only path that may call a model without a ResourceGrant, and it is
narrower than a grant in every dimension: one agenda, one research problem, one
pinned route, one operation, one hard token ceiling, one short TTL, one run.

It is also *not* the proposer. The evaluator route must differ from the route
that proposed the idea, and the runner refuses to label an operator or proposer
response as an independent evaluation.

Every failure mode closes: unavailable provider, missing scope, expired
authority, malformed output, or missing linked evidence all settle the ledger,
close the authority, and raise. None of them fall back to an unscoped call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from meta_harness.frontier import evaluate_frontier
from meta_harness.frontier_authority import (
    FrontierAuthorityError,
    FrontierAuthorityRepository,
    FrontierEvaluationRequest,
    assessment_schema,
    authorize,
)
from meta_harness.frontier_builder import FrontierBuildError
from meta_harness.frontier_source import EvidenceGraphFrontierSource, FrontierAssessment
from meta_harness.repository import MetaHarnessRepository


OPERATION = "frontier_assessment"


class FrontierBootstrapError(RuntimeError):
    """Raised when a bootstrap evaluation cannot produce a usable packet."""


@dataclass(frozen=True)
class BootstrapUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return max(0, int(self.input_tokens)) + max(0, int(self.output_tokens))


@dataclass(frozen=True)
class BootstrapCall:
    provider: str
    model: str
    model_family: str
    prompt_version: str
    system_prompt: str
    user_prompt: str
    token_cap: int


Executor = Callable[[BootstrapCall], "tuple[Any, BootstrapUsage]"]


def _system_prompt(prompt_version: str) -> str:
    return "\n".join(
        (
            "You are an INDEPENDENT Frontier evaluator for one research problem.",
            "You are not the proposer. Do not advocate for the idea.",
            "",
            "Rules:",
            "1. Judge only from the evidence briefing supplied in the user message.",
            "2. Do not invent papers, benchmarks, numbers, or citations.",
            "3. If the evidence cannot decide a field, say so in that field;"
            " do not guess.",
            "4. Negative and refuting evidence is evidence. Do not discount it.",
            "5. Return ONE JSON object and nothing else.",
            "",
            f"prompt_version: {prompt_version}",
            "Schema:",
            assessment_schema(),
        )
    )


def _user_prompt(briefing: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# EVIDENCE BRIEFING (the only evidence you may use)",
            json.dumps(briefing, ensure_ascii=False, sort_keys=True, default=str),
        )
    )


def default_executor(call: BootstrapCall) -> tuple[Any, BootstrapUsage]:
    """Call the pinned provider route once. No retry, no fallback route.

    The provider adapter reports a single combined token count; it is recorded
    as ``output_tokens`` so the authority's ledger never understates usage.
    """
    from agents.llm_client import call_llm_with_provider, parse_llm_json_text

    text, tokens, route = call_llm_with_provider(
        call.system_prompt,
        call.user_prompt,
        provider_name=call.provider,
        max_tokens=call.token_cap,
    )
    if str(route.get("model") or "") != call.model:
        raise FrontierAuthorityError("pinned evaluator model does not match provider")
    parsed, _how = parse_llm_json_text(text)
    return parsed, BootstrapUsage(output_tokens=int(tokens))


def _assessment_from_output(output: Any, authority) -> FrontierAssessment:
    """Malformed output is a closed failure, never a partially filled packet.

    Provenance fields are taken from the authority, never from the model: an
    evaluator cannot claim to be a different evaluator, provider, or prompt
    version than the one it was authorized to be.
    """
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if not isinstance(output, dict):
        raise FrontierBootstrapError("evaluator output is not a JSON object")
    try:
        assessment = FrontierAssessment(
            problem_status=str(output.get("problem_status") or ""),
            contribution_delta=dict(output.get("contribution_delta") or {}),
            why_not_obsolete=str(output.get("why_not_obsolete") or ""),
            minimum_falsification_experiment=dict(
                output.get("minimum_falsification_experiment") or {}
            ),
            evaluator=authority.evaluator,
            provider=authority.provider,
            model=authority.model,
            prompt_version=authority.prompt_version,
            coverage_start=str(output.get("coverage_start") or ""),
            coverage_end=str(output.get("coverage_end") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise FrontierBootstrapError("evaluator output is malformed") from exc
    assessment.validate()
    return assessment


def run_bootstrap_evaluation(
    *,
    authority_id: int,
    agenda_id: int,
    research_problem_id: int,
    proposer_provider: str | None = None,
    proposer_model_family: str | None = None,
    executor: Executor | None = None,
    authority_repository: FrontierAuthorityRepository | None = None,
    source: EvidenceGraphFrontierSource | None = None,
    frontier_repository: MetaHarnessRepository | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Produce one Frontier packet under one authority, or fail closed."""
    authorities = authority_repository or FrontierAuthorityRepository()
    evidence_source = source or EvidenceGraphFrontierSource()
    packets = frontier_repository or MetaHarnessRepository()
    run_executor = executor or default_executor
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    authority = authorities.load(
        int(authority_id),
        agenda_id=int(agenda_id),
        research_problem_id=int(research_problem_id),
    )
    if authority.status == "consumed":
        # Replay of a completed bootstrap: return the packet it produced rather
        # than spending the budget again.
        existing = authorities.completed_packet_id(
            int(authority_id), agenda_id=int(agenda_id)
        )
        if existing:
            return {
                "status": "already_completed",
                "frontier_packet_id": existing,
                "authority_id": int(authority_id),
            }
    authorize(
        authority,
        FrontierEvaluationRequest(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
            operation=OPERATION,
            token_cap=authority.token_cap,
            proposer_provider=proposer_provider,
            proposer_model_family=proposer_model_family,
        ),
        now=current,
    )

    def _fail(reason: str, usage: BootstrapUsage, query_ref: str = "") -> None:
        authorities.record_usage(
            authority=authority,
            operation=OPERATION,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            status="failed",
            failure_reason=reason,
            evidence_query_ref=query_ref,
        )
        authorities.settle(
            authority,
            tokens_used=usage.total_tokens,
            cost_usd=usage.cost_usd,
            outcome="revoked",
        )

    try:
        briefing = evidence_source.evidence_briefing(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
        )
    except FrontierBuildError as exc:
        _fail(f"evidence_unavailable:{exc}", BootstrapUsage())
        raise FrontierBootstrapError(f"linked evidence is unusable:{exc}") from exc

    query_ref = str(briefing.get("query_ref") or "")
    call = BootstrapCall(
        provider=authority.provider,
        model=authority.model,
        model_family=authority.model_family,
        prompt_version=authority.prompt_version,
        system_prompt=_system_prompt(authority.prompt_version),
        user_prompt=_user_prompt(briefing),
        token_cap=authority.token_cap,
    )
    try:
        output, usage = run_executor(call)
    except Exception as exc:
        _fail(f"provider_unavailable:{type(exc).__name__}", BootstrapUsage(), query_ref)
        raise FrontierBootstrapError(
            "frontier evaluator route is unavailable; no fallback is permitted"
        ) from exc

    if not isinstance(usage, BootstrapUsage):
        usage = BootstrapUsage()
    if usage.total_tokens > authority.token_cap:
        _fail("token_cap_exceeded", usage, query_ref)
        raise FrontierBootstrapError("evaluator usage exceeded the authority cap")

    try:
        assessment = _assessment_from_output(output, authority)
        packet = evidence_source.build(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
            assessment=assessment,
        )
    except (FrontierBootstrapError, FrontierBuildError, ValueError) as exc:
        _fail(f"malformed_assessment:{type(exc).__name__}", usage, query_ref)
        raise FrontierBootstrapError(f"evaluator output rejected:{exc}") from exc

    gate = evaluate_frontier(packet)
    packet_id = packets.save_frontier(packet)
    authorities.record_usage(
        authority=authority,
        operation=OPERATION,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        status="succeeded",
        frontier_packet_id=packet_id,
        evidence_query_ref=query_ref,
    )
    authorities.settle(
        authority,
        tokens_used=usage.total_tokens,
        cost_usd=usage.cost_usd,
        outcome="consumed",
    )
    return {
        "status": "completed",
        "authority_id": int(authority_id),
        "frontier_packet_id": packet_id,
        "gate_allowed": gate.allowed,
        "gate_reason_codes": list(gate.reason_codes),
        "evidence_query_ref": query_ref,
        "tokens_used": usage.total_tokens,
        "evaluator": authority.evaluator,
        "provider": authority.provider,
        "model": authority.model,
        "prompt_version": authority.prompt_version,
    }
