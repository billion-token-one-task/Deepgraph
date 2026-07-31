"""Frontier admission rules that reject obsolete or duplicate problems."""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.meta_harness import FrontierPacket


@dataclass(frozen=True)
class FrontierGateDecision:
    allowed: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def evaluate_frontier(packet: FrontierPacket) -> FrontierGateDecision:
    packet.validate()
    reasons: list[str] = []
    if packet.problem_status in {"duplicate", "obsolete", "solved"}:
        reasons.append(f"frontier_{packet.problem_status}")
    if not packet.strongest_recent_work:
        reasons.append("recent_work_not_covered")
    if not packet.latest_benchmarks:
        reasons.append("benchmark_frontier_not_covered")
    if not packet.nearest_prior_art:
        reasons.append("nearest_prior_art_missing")
    if not packet.contribution_delta:
        reasons.append("contribution_delta_missing")
    if packet.obsolete_or_duplicate_evidence and not packet.contribution_delta:
        reasons.append("obsolete_evidence_unanswered")
    return FrontierGateDecision(allowed=not reasons, reason_codes=tuple(reasons))
