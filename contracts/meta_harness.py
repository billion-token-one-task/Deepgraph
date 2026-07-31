"""Versioned contracts for the meta-harness-v1 decision and execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from contracts.base import (
    ContractRecord,
    ContractValidationError,
    ensure_dict,
    ensure_string_list,
    require_non_empty,
)


DECISIONS = {"promote", "kill", "park", "revisit"}
FRONTIER_STATES = {"open", "uncertain", "duplicate", "obsolete", "solved"}
GRANT_STATES = {"active", "consumed", "expired", "revoked"}
VERDICTS = {"supported", "refuted", "inconclusive", "invalid"}
EVIDENCE_STATES = (
    "planned",
    "sanity_passed",
    "full_benchmark_complete",
    "evidence_audited",
    "scientifically_decided",
    "manuscript_allowed",
)


def _positive_id(name: str, value: int | None) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ContractValidationError(f"{name} must be a positive integer")
    return parsed


def _utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ContractValidationError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass
class Estimate(ContractRecord):
    """A point estimate with traceable uncertainty and evaluator provenance."""

    value: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    evaluator: str = ""
    provider: str = ""
    model: str = ""
    evidence_sources: list[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        self.value = float(self.value)
        self.lower = float(self.lower)
        self.upper = float(self.upper)
        if not self.lower <= self.value <= self.upper:
            raise ContractValidationError("estimate must satisfy lower <= value <= upper")
        require_non_empty("evaluator", self.evaluator)
        require_non_empty("provider", self.provider)
        require_non_empty("model", self.model)
        self.evidence_sources = ensure_string_list(self.evidence_sources)
        if not self.evidence_sources:
            raise ContractValidationError("estimate requires at least one evidence source")

    def validate_probability(self, name: str) -> None:
        self.validate()
        if self.lower < 0 or self.upper > 1:
            raise ContractValidationError(f"{name} interval must be within [0, 1]")


@dataclass
class FrontierPacket(ContractRecord):
    agenda_id: int = 0
    research_problem_id: int = 0
    retrieved_at: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    problem_status: str = "uncertain"
    strongest_recent_work: list[dict[str, Any]] = field(default_factory=list)
    latest_benchmarks: list[dict[str, Any]] = field(default_factory=list)
    nearest_prior_art: list[dict[str, Any]] = field(default_factory=list)
    contribution_delta: dict[str, Any] = field(default_factory=dict)
    obsolete_or_duplicate_evidence: list[dict[str, Any]] = field(default_factory=list)
    counterevidence_and_negative_results: list[dict[str, Any]] = field(default_factory=list)
    why_not_obsolete: str = ""
    minimum_falsification_experiment: dict[str, Any] = field(default_factory=dict)
    evaluator: str = ""
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    frontier_packet_id: int | None = None

    def validate(self) -> None:
        super().validate()
        self.agenda_id = _positive_id("agenda_id", self.agenda_id)
        self.research_problem_id = _positive_id(
            "research_problem_id", self.research_problem_id
        )
        _utc(self.retrieved_at, "retrieved_at")
        self.coverage = ensure_dict(self.coverage)
        if not self.coverage:
            raise ContractValidationError("FrontierPacket requires retrieval coverage")
        if self.problem_status not in FRONTIER_STATES:
            raise ContractValidationError("invalid frontier problem_status")
        require_non_empty("why_not_obsolete", self.why_not_obsolete)
        self.minimum_falsification_experiment = ensure_dict(
            self.minimum_falsification_experiment
        )
        if not self.minimum_falsification_experiment:
            raise ContractValidationError(
                "FrontierPacket requires a minimum falsification experiment"
            )
        require_non_empty("evaluator", self.evaluator)
        require_non_empty("provider", self.provider)
        require_non_empty("model", self.model)
        require_non_empty("prompt_version", self.prompt_version)


@dataclass
class IdeaDecisionPacket(ContractRecord):
    agenda_id: int = 0
    idea_id: int = 0
    frontier_packet_id: int = 0
    expected_impact: Estimate = field(default_factory=Estimate)
    success_probability: Estimate = field(default_factory=Estimate)
    novelty: Estimate = field(default_factory=Estimate)
    obsolescence_probability: Estimate = field(default_factory=Estimate)
    falsification_value: Estimate = field(default_factory=Estimate)
    reuse_value: Estimate = field(default_factory=Estimate)
    expected_token_cost: Estimate = field(default_factory=Estimate)
    expected_gpu_cost: Estimate = field(default_factory=Estimate)
    time_to_feedback: Estimate = field(default_factory=Estimate)
    execution_risk: Estimate = field(default_factory=Estimate)
    information_value: Estimate = field(default_factory=Estimate)
    candidate_family: str = ""
    correlation_keys: list[str] = field(default_factory=list)
    decision: str = "park"
    reason_codes: list[str] = field(default_factory=list)
    revisit_condition: dict[str, Any] = field(default_factory=dict)
    revisit_after: str | None = None
    policy_version: str = "portfolio_heuristic_v1"
    decision_packet_id: int | None = None

    def validate(self) -> None:
        super().validate()
        self.agenda_id = _positive_id("agenda_id", self.agenda_id)
        self.idea_id = _positive_id("idea_id", self.idea_id)
        self.frontier_packet_id = _positive_id(
            "frontier_packet_id", self.frontier_packet_id
        )
        for name in (
            "expected_impact",
            "success_probability",
            "novelty",
            "obsolescence_probability",
            "falsification_value",
            "reuse_value",
            "expected_token_cost",
            "expected_gpu_cost",
            "time_to_feedback",
            "execution_risk",
            "information_value",
        ):
            estimate = getattr(self, name)
            if isinstance(estimate, dict):
                estimate = Estimate.from_partial_dict(estimate)
                setattr(self, name, estimate)
            estimate.validate()
        for name in (
            "success_probability",
            "novelty",
            "obsolescence_probability",
            "execution_risk",
        ):
            getattr(self, name).validate_probability(name)
        require_non_empty("candidate_family", self.candidate_family)
        self.correlation_keys = ensure_string_list(self.correlation_keys)
        if not self.correlation_keys:
            raise ContractValidationError("correlation_keys cannot be empty")
        if self.decision not in DECISIONS:
            raise ContractValidationError("invalid portfolio decision")
        self.reason_codes = ensure_string_list(self.reason_codes)
        if not self.reason_codes:
            raise ContractValidationError("portfolio decision requires reason codes")
        if self.decision == "park":
            self.revisit_condition = ensure_dict(self.revisit_condition)
            if not self.revisit_condition and not self.revisit_after:
                raise ContractValidationError(
                    "parked ideas require a revisit condition or expiry"
                )
        if self.revisit_after:
            _utc(self.revisit_after, "revisit_after")
        require_non_empty("policy_version", self.policy_version)


@dataclass
class ResourceGrant(ContractRecord):
    agenda_id: int = 0
    idea_id: int = 0
    decision_packet_id: int = 0
    stage: str = ""
    token_cap: int = 0
    gpu_class: str = "none"
    max_gpu_hours: float = 0.0
    backend_allowlist: list[str] = field(default_factory=list)
    artifact_requirements: list[str] = field(default_factory=list)
    expires_at: str = ""
    grant_reason: str = ""
    idempotency_key: str = ""
    status: str = "active"
    grant_id: int | None = None
    reservation_id: int | None = None

    def validate(self) -> None:
        super().validate()
        self.agenda_id = _positive_id("agenda_id", self.agenda_id)
        self.idea_id = _positive_id("idea_id", self.idea_id)
        self.decision_packet_id = _positive_id(
            "decision_packet_id", self.decision_packet_id
        )
        require_non_empty("stage", self.stage)
        self.token_cap = int(self.token_cap)
        self.max_gpu_hours = float(self.max_gpu_hours)
        if self.token_cap < 0 or self.max_gpu_hours < 0:
            raise ContractValidationError("grant caps cannot be negative")
        if self.token_cap == 0 and self.max_gpu_hours == 0:
            raise ContractValidationError("ResourceGrant must grant a bounded resource")
        self.backend_allowlist = ensure_string_list(self.backend_allowlist)
        if not self.backend_allowlist:
            raise ContractValidationError("ResourceGrant backend_allowlist cannot be empty")
        self.artifact_requirements = ensure_string_list(self.artifact_requirements)
        if not self.artifact_requirements:
            raise ContractValidationError("ResourceGrant requires artifact outputs")
        _utc(self.expires_at, "expires_at")
        require_non_empty("grant_reason", self.grant_reason)
        require_non_empty("idempotency_key", self.idempotency_key)
        if self.status not in GRANT_STATES:
            raise ContractValidationError("invalid ResourceGrant status")


@dataclass
class OutcomeRecord(ContractRecord):
    agenda_id: int = 0
    idea_id: int = 0
    resource_grant_id: int = 0
    experiment_run_id: int | None = None
    actual_tokens: int = 0
    actual_gpu_hours: float = 0.0
    wall_seconds: float = 0.0
    execution_result: str = ""
    effect: float | None = None
    baseline: float | None = None
    verdict: str = "inconclusive"
    new_information: dict[str, Any] = field(default_factory=dict)
    state_decision: str = "planned"
    prediction_error: dict[str, Any] = field(default_factory=dict)
    artifact_manifest: dict[str, Any] = field(default_factory=dict)
    outcome_record_id: int | None = None

    def validate(self) -> None:
        super().validate()
        self.agenda_id = _positive_id("agenda_id", self.agenda_id)
        self.idea_id = _positive_id("idea_id", self.idea_id)
        self.resource_grant_id = _positive_id(
            "resource_grant_id", self.resource_grant_id
        )
        if self.experiment_run_id is not None:
            self.experiment_run_id = _positive_id(
                "experiment_run_id", self.experiment_run_id
            )
        self.actual_tokens = int(self.actual_tokens)
        self.actual_gpu_hours = float(self.actual_gpu_hours)
        self.wall_seconds = float(self.wall_seconds)
        if min(self.actual_tokens, self.actual_gpu_hours, self.wall_seconds) < 0:
            raise ContractValidationError("OutcomeRecord usage cannot be negative")
        require_non_empty("execution_result", self.execution_result)
        if self.verdict not in VERDICTS:
            raise ContractValidationError("invalid OutcomeRecord verdict")
        if self.state_decision not in EVIDENCE_STATES:
            raise ContractValidationError("invalid OutcomeRecord state_decision")
        self.new_information = ensure_dict(self.new_information)
        self.prediction_error = ensure_dict(self.prediction_error)
        self.artifact_manifest = ensure_dict(self.artifact_manifest)
        if not self.artifact_manifest:
            raise ContractValidationError("OutcomeRecord requires an artifact manifest")
