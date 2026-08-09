"""Transparent best-of-N portfolio policy with diversity and budget buckets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable

from contracts.meta_harness import IdeaDecisionPacket, ResourceGrant


@dataclass(frozen=True)
class PortfolioPolicy:
    version: str = "portfolio_heuristic_v1"
    promote_count: int = 1
    exploration_fraction: float = 0.20
    falsification_fraction: float = 0.15
    surprise_fraction: float = 0.10
    kill_obsolescence_threshold: float = 0.80
    kill_execution_risk_threshold: float = 0.92
    max_family_share: float = 0.50
    park_hours: int = 168
    impact_weight: float = 1.2
    success_weight: float = 1.0
    novelty_weight: float = 0.9
    information_weight: float = 1.0
    falsification_weight: float = 0.65
    reuse_weight: float = 0.45
    obsolescence_weight: float = 1.1
    risk_weight: float = 0.8
    token_cost_weight: float = 0.15
    gpu_cost_weight: float = 0.30
    feedback_time_weight: float = 0.15
    token_normalizer: float = 100_000.0
    gpu_hour_normalizer: float = 8.0
    feedback_hour_normalizer: float = 168.0

    def validate(self) -> None:
        if self.promote_count <= 0:
            raise ValueError("promote_count must be positive")
        for name in (
            "exploration_fraction",
            "falsification_fraction",
            "surprise_fraction",
            "max_family_share",
        ):
            value = float(getattr(self, name))
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            self.exploration_fraction
            + self.falsification_fraction
            + self.surprise_fraction
            > 1
        ):
            raise ValueError("portfolio reserve fractions cannot sum above 1")
        if min(
            self.token_normalizer,
            self.gpu_hour_normalizer,
            self.feedback_hour_normalizer,
        ) <= 0:
            raise ValueError("portfolio normalizers must be positive")


@dataclass(frozen=True)
class RankedIdea:
    packet: IdeaDecisionPacket
    score: float
    bucket: str
    penalties: tuple[str, ...] = field(default_factory=tuple)


def _base_score(packet: IdeaDecisionPacket, policy: PortfolioPolicy) -> float:
    return (
        policy.impact_weight * packet.expected_impact.value
        + policy.success_weight * packet.success_probability.value
        + policy.novelty_weight * packet.novelty.value
        + policy.information_weight * packet.information_value.value
        + policy.falsification_weight * packet.falsification_value.value
        + policy.reuse_weight * packet.reuse_value.value
        - policy.obsolescence_weight * packet.obsolescence_probability.value
        - policy.risk_weight * packet.execution_risk.value
        - policy.token_cost_weight
        * (packet.expected_token_cost.value / policy.token_normalizer)
        - policy.gpu_cost_weight
        * (packet.expected_gpu_cost.value / policy.gpu_hour_normalizer)
        - policy.feedback_time_weight
        * (packet.time_to_feedback.value / policy.feedback_hour_normalizer)
    )


def _bucket(packet: IdeaDecisionPacket) -> str:
    if packet.falsification_value.value >= 0.70:
        return "falsification"
    if packet.information_value.value >= 0.75 and packet.success_probability.value < 0.50:
        return "exploration"
    if packet.novelty.value >= 0.85 and packet.execution_risk.value >= 0.55:
        return "surprise"
    return "exploitation"


def rank_candidates(
    packets: Iterable[IdeaDecisionPacket],
    *,
    policy: PortfolioPolicy | None = None,
) -> list[RankedIdea]:
    selected_policy = policy or PortfolioPolicy()
    selected_policy.validate()
    family_seen: dict[str, int] = {}
    correlation_seen: set[str] = set()
    ranked: list[RankedIdea] = []
    prelim = []
    for packet in packets:
        packet.validate()
        prelim.append((_base_score(packet, selected_policy), packet))
    for score, packet in sorted(prelim, key=lambda item: item[0], reverse=True):
        penalties: list[str] = []
        family_count = family_seen.get(packet.candidate_family, 0)
        if family_count:
            score -= 0.25 * family_count
            penalties.append("candidate_family_correlation")
        overlap = correlation_seen.intersection(packet.correlation_keys)
        if overlap:
            score -= min(0.75, 0.15 * len(overlap))
            penalties.append("correlation_key_overlap")
        ranked.append(
            RankedIdea(
                packet=packet,
                score=round(score, 6),
                bucket=_bucket(packet),
                penalties=tuple(penalties),
            )
        )
        family_seen[packet.candidate_family] = family_count + 1
        correlation_seen.update(packet.correlation_keys)
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def decide_portfolio(
    packets: Iterable[IdeaDecisionPacket],
    *,
    killed_signatures: set[str] | None = None,
    policy: PortfolioPolicy | None = None,
    now: datetime | None = None,
) -> list[IdeaDecisionPacket]:
    """Assign promote/kill/park with explicit reasons and revisit conditions."""
    selected_policy = policy or PortfolioPolicy()
    selected_policy.validate()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    killed = killed_signatures or set()
    ranked = rank_candidates(packets, policy=selected_policy)
    decisions: list[IdeaDecisionPacket] = []
    promotable: list[RankedIdea] = []
    for item in ranked:
        packet = item.packet
        repeated = bool(killed.intersection(packet.correlation_keys))
        if repeated:
            decisions.append(
                replace(
                    packet,
                    decision="kill",
                    reason_codes=["similar_to_previously_killed_idea"],
                    policy_version=selected_policy.version,
                )
            )
        elif (
            packet.obsolescence_probability.value
            >= selected_policy.kill_obsolescence_threshold
        ):
            decisions.append(
                replace(
                    packet,
                    decision="kill",
                    reason_codes=["obsolescence_probability_too_high"],
                    policy_version=selected_policy.version,
                )
            )
        elif packet.execution_risk.value >= selected_policy.kill_execution_risk_threshold:
            decisions.append(
                replace(
                    packet,
                    decision="kill",
                    reason_codes=["execution_risk_too_high"],
                    policy_version=selected_policy.version,
                )
            )
        else:
            promotable.append(item)

    chosen: list[RankedIdea] = []
    bucket_targets = {
        "exploration": round(selected_policy.promote_count * selected_policy.exploration_fraction),
        "falsification": round(
            selected_policy.promote_count * selected_policy.falsification_fraction
        ),
        "surprise": round(selected_policy.promote_count * selected_policy.surprise_fraction),
    }
    for bucket, target in bucket_targets.items():
        chosen.extend(
            item
            for item in promotable
            if item.bucket == bucket and item not in chosen
        )
        chosen = chosen[: min(selected_policy.promote_count, len(chosen))]
        if len([item for item in chosen if item.bucket == bucket]) > target:
            excess = len([item for item in chosen if item.bucket == bucket]) - target
            for item in reversed(chosen):
                if excess <= 0:
                    break
                if item.bucket == bucket:
                    chosen.remove(item)
                    excess -= 1
    for item in promotable:
        if len(chosen) >= selected_policy.promote_count:
            break
        if item not in chosen:
            chosen.append(item)

    chosen_ids = {item.packet.idea_id for item in chosen}
    for item in promotable:
        if item.packet.idea_id in chosen_ids:
            reasons = ["portfolio_score_selected", f"bucket:{item.bucket}"]
            reasons.extend(item.penalties)
            decisions.append(
                replace(
                    item.packet,
                    decision="promote",
                    reason_codes=reasons,
                    revisit_condition={},
                    revisit_after=None,
                    policy_version=selected_policy.version,
                )
            )
        else:
            decisions.append(
                replace(
                    item.packet,
                    decision="park",
                    reason_codes=["opportunity_cost", f"bucket:{item.bucket}"],
                    revisit_condition={
                        "on": [
                            "budget_bucket_refilled",
                            "frontier_changed",
                            "correlated_outcome_recorded",
                        ]
                    },
                    revisit_after=(current + timedelta(hours=selected_policy.park_hours)).isoformat(),
                    policy_version=selected_policy.version,
                )
            )
    for decision in decisions:
        decision.validate()
    return sorted(decisions, key=lambda item: item.idea_id)


def issue_resource_grant(
    decision: IdeaDecisionPacket,
    *,
    stage: str,
    token_cap: int,
    gpu_class: str,
    max_gpu_hours: float,
    backend_allowlist: list[str],
    artifact_requirements: list[str],
    expires_at: str,
    idempotency_key: str,
    preflight_result_id: int | None = None,
) -> ResourceGrant:
    decision.validate()
    if decision.decision not in {"promote", "revisit"}:
        raise PermissionError("only promoted or revisited ideas can receive a grant")
    if not decision.decision_packet_id:
        raise ValueError("persisted decision_packet_id is required before issuing a grant")
    grant = ResourceGrant(
        agenda_id=decision.agenda_id,
        idea_id=decision.idea_id,
        decision_packet_id=decision.decision_packet_id,
        stage=stage,
        token_cap=token_cap,
        gpu_class=gpu_class,
        max_gpu_hours=max_gpu_hours,
        backend_allowlist=backend_allowlist,
        artifact_requirements=artifact_requirements,
        expires_at=expires_at,
        grant_reason=";".join(decision.reason_codes),
        idempotency_key=idempotency_key,
        preflight_result_id=preflight_result_id,
    )
    grant.validate()
    return grant
